import os
import numpy as np
import torch
from tqdm import tqdm

from utils.utils import AverageMeter, log_datasetloss, write_to_tensorboard


def compute_sequence_metrics(loss_module,
                             predicted: torch.Tensor,
                             target: torch.Tensor,
                             template: torch.Tensor,
                             annot_type: str):
    device = predicted.device
    zero = torch.tensor(0.0, device=device)
    mouth_metric_L2 = zero
    mouth_metric_L2_norm = zero
    mve = zero
    ufve = zero
    fdd = zero

    try:
        mouth_metric_L2, mouth_metric_L2_norm = loss_module.mouth_metric(
            predicted.detach(), target.detach(), annot_type)
    except NotImplementedError:
        pass

    if hasattr(loss_module, 'get_vertices_sequence'):
        try:
            pred_vertices = loss_module.get_vertices_sequence(
                predicted.detach(), annot_type)
            target_vertices = loss_module.get_vertices_sequence(
                target.detach(), annot_type)
            diff = pred_vertices - target_vertices
            mve = diff.norm(dim=-1).mean()

            upper_idx = None
            if hasattr(loss_module, 'get_upper_face_indices'):
                upper_idx = loss_module.get_upper_face_indices(annot_type)
            if isinstance(upper_idx, torch.Tensor) and upper_idx.numel() > 0:
                upper_idx = upper_idx.to(pred_vertices.device)
                pred_upper = pred_vertices.index_select(2, upper_idx)
                target_upper = target_vertices.index_select(2, upper_idx)
                ufve = (pred_upper - target_upper).norm(dim=-1).mean()

                # FDD（official / CodeTalker-style）：
                # 1) motion = vertices - template_vertices
                # 2) 对上半脸顶点：energy(t, v) = ||motion(t, v)||^2
                # 3) 对每个顶点 v：std_t(energy(:, v))
                # 4) motion_std = mean_v(std_t(energy(:, v)))
                # 5) FDD = mean_b(|motion_std_gt - motion_std_pred|)
                if template is not None and pred_upper.size(1) > 1:
                    # template: [B, 1, D] or [B, D]
                    if template.dim() == 3 and template.size(1) == 1:
                        template_flat = template[:, 0]
                    elif template.dim() == 2:
                        template_flat = template
                    else:
                        template_flat = template.reshape(template.size(0), -1)

                    template_vertices = None
                    if hasattr(loss_module, 'get_vertices'):
                        template_vertices = loss_module.get_vertices(
                            template_flat.detach(), annot_type)

                    if isinstance(template_vertices, torch.Tensor) and template_vertices.numel() > 0:
                        template_vertices = template_vertices.to(
                            pred_vertices.device).unsqueeze(1)  # [B, 1, V, 3]
                        motion_pred = pred_vertices - template_vertices
                        motion_gt = target_vertices - template_vertices
                        motion_pred = motion_pred.index_select(2, upper_idx)
                        motion_gt = motion_gt.index_select(2, upper_idx)
                        energy_pred = motion_pred.pow(2).sum(
                            dim=-1)  # [B, T, V]
                        energy_gt = motion_gt.pow(2).sum(dim=-1)
                        pred_motion_std = energy_pred.std(
                            dim=1, unbiased=False).mean(dim=-1)  # [B]
                        gt_motion_std = energy_gt.std(
                            dim=1, unbiased=False).mean(dim=-1)  # [B]
                        fdd = (gt_motion_std - pred_motion_std).abs().mean()
        except NotImplementedError:
            pass

    return (
        mouth_metric_L2.detach(),
        mouth_metric_L2_norm.detach(),
        mve.detach(),
        ufve.detach(),
        fdd.detach(),
    )


def compute_region_vertex_losses(loss_module, predicted: torch.Tensor,
                                 target: torch.Tensor, annot_type: str):
    """Lightweight region vertex L1 losses aligned with eval mouth/upper indices."""
    device = predicted.device
    zero = torch.tensor(0.0, device=device)
    lve_loss = zero
    ufve_loss = zero
    if not hasattr(loss_module, 'get_vertices_sequence'):
        return lve_loss, ufve_loss
    if not hasattr(loss_module, 'loss_module_dict'):
        return lve_loss, ufve_loss
    if annot_type not in loss_module.loss_module_dict:
        return lve_loss, ufve_loss
    try:
        pred_vertices = loss_module.get_vertices_sequence(predicted, annot_type)
        target_vertices = loss_module.get_vertices_sequence(target, annot_type)
        module = loss_module.loss_module_dict[annot_type]
        mouth_idx = getattr(module, 'mouth_idx', None)
        if isinstance(mouth_idx, torch.Tensor) and mouth_idx.numel() > 0:
            mouth_idx = mouth_idx.to(pred_vertices.device)
            pred_mouth = pred_vertices.index_select(2, mouth_idx)
            target_mouth = target_vertices.index_select(2, mouth_idx)
            lve_loss = torch.nn.functional.l1_loss(pred_mouth, target_mouth)
        if hasattr(loss_module, 'get_upper_face_indices'):
            upper_idx = loss_module.get_upper_face_indices(annot_type)
            if isinstance(upper_idx, torch.Tensor) and upper_idx.numel() > 0:
                upper_idx = upper_idx.to(pred_vertices.device)
                pred_upper = pred_vertices.index_select(2, upper_idx)
                target_upper = target_vertices.index_select(2, upper_idx)
                ufve_loss = torch.nn.functional.l1_loss(
                    pred_upper, target_upper)
    except NotImplementedError:
        return lve_loss, ufve_loss
    return lve_loss, ufve_loss


def train_epoch(data_loader, model, loss_module, optimizer, epoch, args):
    """单个训练轮：更新模型参数并统计训练指标。"""
    model.train()
    train_meter = args.train_meter
    metric_name_list = args.train_metric_name_list
    use_audio = getattr(args, 'use_audio', True)
    mixed_dataset_name = 'mixed_dataset'
    grad_acc_steps = int(getattr(args, 'gradient_accumulation_steps', 1) or 1)
    grad_acc_steps = max(1, grad_acc_steps)
    optimizer.zero_grad(set_to_none=True)
    # tqdm 进度条方便观测当前训练进度
    pbar = tqdm(data_loader)
    pbar.set_description(f'Epoch: {epoch} train')
    micro_step = 0
    for batch in pbar:
        data = batch['data'].cuda(non_blocking=True)
        zero_loss = torch.tensor(0.0, device=data.device)
        template = batch['template'].cuda(non_blocking=True)
        audio = None
        audio_cached_feat = batch.get('audio_cached_feat', None)
        if isinstance(audio_cached_feat, torch.Tensor):
            audio_cached_feat = audio_cached_feat.cuda(non_blocking=True)
        else:
            audio_cached_feat = None
        if use_audio:
            audio_tensor = batch.get('audio', None)
            if isinstance(audio_tensor, torch.Tensor):
                audio = audio_tensor.cuda(non_blocking=True)
        identity = batch['id'].cuda(non_blocking=True)
        use_eeg = getattr(args, 'use_eeg', True)
        eeg = batch.get('eeg', None)
        eeg_cached_feat = batch.get('eeg_cached_feat', None)
        if isinstance(eeg_cached_feat, torch.Tensor):
            eeg_cached_feat = eeg_cached_feat.cuda(non_blocking=True)
        else:
            eeg_cached_feat = None
        if use_eeg and isinstance(eeg, torch.Tensor):
            eeg = eeg.cuda(non_blocking=True)
        else:
            eeg = None
        fps_tensor = batch.get('fps', None)
        fps_value = None
        if fps_tensor is not None:
            if isinstance(fps_tensor, torch.Tensor):
                fps_value = float(fps_tensor[0].item())
            else:
                fps_value = float(fps_tensor[0])

        if args.random_id_prob > 0.0:
            if np.random.random() < args.random_id_prob:
                identity = identity.fill_(args.identity_num - 1)
        annot_type = batch['annot_type'][0]
        dataset_name = batch['dataset_name'][0]

        out_motion, out_pca, gt_pca = model(
            audio=audio,
            eeg=eeg,
            audio_cached_feat=audio_cached_feat,
            eeg_cached_feat=eeg_cached_feat,
            template=template,
            face_motion=data,
            style_idx=identity,
            annot_type=annot_type,
            fps=fps_value)

        rec_loss = loss_module(out_motion, data, annot_type)
        # 速度正则：相邻帧差的 MSE，缓解抖动、鼓励平滑动态
        speed_weight = getattr(args, 'velocity_loss_weight', 0.0)
        if out_motion.shape[1] > 1:
            pred_vel = out_motion[:, 1:] - out_motion[:, :-1]
            gt_vel = data[:, 1:] - data[:, :-1]
            speed_loss = torch.nn.functional.mse_loss(pred_vel, gt_vel)
        else:
            speed_loss = zero_loss

        # 加速度正则：速度差（即加速度）的 MSE，进一步平滑动态，防止突变
        acc_weight = getattr(args, 'acceleration_loss_weight', 0.0)
        if out_motion.shape[1] > 2:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
            acc_loss = torch.nn.functional.mse_loss(pred_acc, gt_acc)
        else:
            acc_loss = zero_loss

        # 口型动态（BS 空间）：只约束口型相关维度的速度/加速度 + 欠幅惩罚。
        # 目的：在不改 GT 的前提下，缓解“嘴动幅度小 / 细节（开合次数）缺失”的问题。
        mouth_speed_weight = float(
            getattr(args, 'mouth_velocity_loss_weight', 0.0) or 0.0)
        mouth_acc_weight = float(
            getattr(args, 'mouth_acceleration_loss_weight', 0.0) or 0.0)
        mouth_under_std_weight = float(
            getattr(args, 'mouth_under_std_weight', 0.0) or 0.0)
        mouth_speed_loss = zero_loss
        mouth_acc_loss = zero_loss
        mouth_under_std_loss = zero_loss

        mouth_idx = getattr(loss_module, 'mouth_bs_idx', None)
        if isinstance(mouth_idx, torch.Tensor) and mouth_idx.numel() > 0:
            mouth_idx = mouth_idx.to(out_motion.device)
            pred_mouth = out_motion.index_select(2, mouth_idx)
            gt_mouth = data.index_select(2, mouth_idx)

            if mouth_under_std_weight > 0.0:
                pred_std = pred_mouth.std(dim=1, unbiased=False)
                gt_std = gt_mouth.std(dim=1, unbiased=False)
                mouth_under_std_loss = torch.relu(gt_std -
                                                  pred_std).mean()

            if out_motion.shape[1] > 1:
                pred_m_vel = pred_mouth[:, 1:] - pred_mouth[:, :-1]
                gt_m_vel = gt_mouth[:, 1:] - gt_mouth[:, :-1]
                if mouth_speed_weight > 0.0:
                    mouth_speed_loss = torch.nn.functional.mse_loss(
                        pred_m_vel, gt_m_vel)
                if out_motion.shape[1] > 2 and mouth_acc_weight > 0.0:
                    pred_m_acc = pred_m_vel[:, 1:] - pred_m_vel[:, :-1]
                    gt_m_acc = gt_m_vel[:, 1:] - gt_m_vel[:, :-1]
                    mouth_acc_loss = torch.nn.functional.mse_loss(
                        pred_m_acc, gt_m_acc)
        if out_pca is not None:
            pca_loss = loss_module.pca_loss(out_pca, gt_pca)
        else:
            pca_loss = zero_loss
        lve_region_loss = zero_loss
        ufve_region_loss = zero_loss
        w_lve = float(getattr(args, 'w_lve', 0.0) or 0.0)
        w_ufve = float(getattr(args, 'w_ufve', 0.0) or 0.0)
        if w_lve > 0.0 or w_ufve > 0.0:
            lve_region_loss, ufve_region_loss = compute_region_vertex_losses(
                loss_module, out_motion, data, annot_type)
        total_loss = (
            rec_loss + args.pca_weight * pca_loss + speed_weight * speed_loss +
            acc_weight * acc_loss + mouth_speed_weight * mouth_speed_loss +
            mouth_acc_weight * mouth_acc_loss + mouth_under_std_weight *
            mouth_under_std_loss + w_lve * lve_region_loss + w_ufve *
            ufve_region_loss)

        # 梯度累积：每个 micro-step 反传 total_loss / grad_acc_steps，保证累积后的梯度尺度不随步数变化。
        (total_loss / grad_acc_steps).backward()
        micro_step += 1

        # 仅在实际 step 前裁剪（裁剪累积后的总梯度），避免每个 micro-step 都裁剪导致梯度被过度缩小。
        if micro_step % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if dataset_name not in train_meter:
            train_meter[dataset_name] = {
                metric: AverageMeter() for metric in metric_name_list
            }
        with torch.no_grad():
            mouth_metric_L2, mouth_metric_L2_norm, mve, ufve, fdd = (
                compute_sequence_metrics(loss_module, out_motion, data,
                                          template, annot_type))
        metric_values = (
            rec_loss.detach(),
            mouth_metric_L2,
            mouth_metric_L2_norm,
            mve,
            ufve,
            fdd,
        )
        # 更新统计指标：注意这里用 total loss 替换了原来的 rec_loss
        train_meter[dataset_name]['rec_L2_loss'].update(total_loss.item())
        train_meter[mixed_dataset_name]['rec_L2_loss'].update(total_loss.item())
        # Update other metrics as before
        for i, k in enumerate(metric_name_list):
            if k == 'rec_L2_loss': # Skip rec_L2_loss as it's handled above
                continue
            v = metric_values[i]
            if isinstance(v, torch.Tensor):
                scalar = v.detach().cpu().item()
            else:
                scalar = float(v)
            train_meter[dataset_name][k].update(scalar)
            train_meter[mixed_dataset_name][k].update(scalar)

    # 处理最后不足 grad_acc_steps 的尾巴：确保不会把残留梯度带到下一个 epoch。
    if micro_step % grad_acc_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    log_datasetloss(args.logger, epoch, 'train', train_meter, only_mix=True)
    write_to_tensorboard(
        args.writer, epoch, 'train', train_meter, only_mix=False)

    for dataset_name, sub_metric_dict in train_meter.items():
        for metric_name in sub_metric_dict.keys():
            sub_metric_dict[metric_name].reset()
    return


def validate_epoch(data_loader, model, loss_module, epoch, phase, args):
    assert phase in ('val', 'test')
    model.eval()
    val_meter = args.val_meter
    metric_name_list = args.val_metric_name_list
    use_audio = getattr(args, 'use_audio', True)
    mixed_dataset_name = 'mixed_dataset'
    # 如果提供了保存目录，则额外记录每个序列的详细指标（用于 Fig.6 后续可视化）
    save_sequence_stats = getattr(args, 'save_path', None) is not None
    if save_sequence_stats:
        stats_dir = os.path.join(args.save_path, 'metrics', phase)
        os.makedirs(stats_dir, exist_ok=True)
        sequence_counter = 0
    with torch.no_grad():
        pbar = tqdm(data_loader)
        pbar.set_description(f'Epoch: {epoch} {phase:<5}')
        for batch in pbar:
            data = batch['data'].cuda(non_blocking=True)
            template = batch['template'].cuda(non_blocking=True)
            audio = None
            audio_cached_feat = batch.get('audio_cached_feat', None)
            if isinstance(audio_cached_feat, torch.Tensor):
                audio_cached_feat = audio_cached_feat.cuda(non_blocking=True)
            else:
                audio_cached_feat = None
            if use_audio:
                audio_tensor = batch.get('audio', None)
                if isinstance(audio_tensor, torch.Tensor):
                    audio = audio_tensor.cuda(non_blocking=True)
            identity = batch['id'].cuda(non_blocking=True)
            # DataLoader 会把字符串打包成列表，统一转成 list 方便索引
            annot_type_list = batch['annot_type']
            if not isinstance(annot_type_list, (list, tuple)):
                annot_type_list = [annot_type_list]
            dataset_name_list = batch['dataset_name']
            if not isinstance(dataset_name_list, (list, tuple)):
                dataset_name_list = [dataset_name_list]
            use_eeg = getattr(args, 'use_eeg', True)
            eeg = batch.get('eeg', None)
            eeg_cached_feat = batch.get('eeg_cached_feat', None)
            if isinstance(eeg_cached_feat, torch.Tensor):
                eeg_cached_feat = eeg_cached_feat.cuda(non_blocking=True)
            else:
                eeg_cached_feat = None
            if use_eeg and isinstance(eeg, torch.Tensor):
                eeg = eeg.cuda(non_blocking=True)
            else:
                eeg = None
            fps_tensor = batch.get('fps', None)
            fps_value = None
            if fps_tensor is not None:
                if isinstance(fps_tensor, torch.Tensor):
                    fps_value = float(fps_tensor[0].item())
                else:
                    fps_value = float(fps_tensor[0])
            annot_type = annot_type_list[0]
            dataset_name = dataset_name_list[0]
            annot_path_list = batch.get('annot_path', None)
            audio_path_list = batch.get('audio_path', None)
            if args.random_id_prob >= 1.0:
                identity = identity.fill_(args.identity_num - 1)
            if dataset_name not in val_meter:
                val_meter[dataset_name] = {
                    metric: AverageMeter() for metric in metric_name_list
                }
            out_motion, out_pca, gt_pca = model(
                audio=audio,
                eeg=eeg,
                audio_cached_feat=audio_cached_feat,
                eeg_cached_feat=eeg_cached_feat,
                template=template,
                face_motion=data,
                style_idx=identity,
                annot_type=annot_type,
                fps=fps_value)
            rec_loss = loss_module(out_motion, data, annot_type)
            mouth_metric_L2, mouth_metric_L2_norm = loss_module.mouth_metric(
                out_motion, data, annot_type)
            # 评估指标初始化
            mve = torch.tensor(0.0, device=out_motion.device)
            ufve = torch.tensor(0.0, device=out_motion.device)
            fdd = torch.tensor(0.0, device=out_motion.device)
            per_vertex_error = None
            velocity_mean_pred = velocity_std_pred = None
            velocity_mean_gt = velocity_std_gt = None
            zero_loss = torch.tensor(0.0, device=out_motion.device)
            # 可从 loss 模块恢复顶点序列时，计算新增的三项指标以及速度统计
            if hasattr(loss_module, 'get_vertices_sequence'):
                try: 
                    #全脸顶点序列
                    pred_vertices = loss_module.get_vertices_sequence(out_motion, annot_type)
                    target_vertices = loss_module.get_vertices_sequence(data, annot_type)
                    diff = pred_vertices - target_vertices
                    per_vertex_error = diff.norm(dim=-1).mean(dim=1)  # 逐顶点平均误差（Fig.6b 误差图）
                    mve = diff.norm(dim=-1).mean()
                    if pred_vertices.size(1) > 1:# 多帧样本才有速度
                        velocity_pred = pred_vertices[:, 1:] - pred_vertices[:, :-1]
                        velocity_gt = target_vertices[:, 1:] - target_vertices[:, :-1]
                        speed_pred = velocity_pred.norm(dim=-1)
                        speed_gt = velocity_gt.norm(dim=-1)
                        velocity_mean_pred = speed_pred.mean(dim=1)
                        velocity_std_pred = speed_pred.std(
                            dim=1, unbiased=False)
                        velocity_mean_gt = speed_gt.mean(dim=1)
                        velocity_std_gt = speed_gt.std(dim=1, unbiased=False)
                    else:
                        # 单帧样本没有速度，填充 0 避免后续保存时报错
                        vertex_count = pred_vertices.size(2)
                        zero_tensor = torch.zeros(pred_vertices.size(0),
                                                  vertex_count,
                                                  device=out_motion.device)
                        velocity_mean_pred = zero_tensor.clone()
                        velocity_std_pred = zero_tensor.clone()
                        velocity_mean_gt = zero_tensor.clone()
                        velocity_std_gt = zero_tensor.clone()

                    #上半脸顶点序列    
                    upper_idx = None 
                    if hasattr(loss_module, 'get_upper_face_indices'):
                        upper_idx = loss_module.get_upper_face_indices(annot_type)
                    if isinstance(upper_idx, torch.Tensor) and upper_idx.numel() > 0:
                        upper_idx = upper_idx.to(pred_vertices.device)
                        pred_upper = pred_vertices.index_select(2, upper_idx)
                        target_upper = target_vertices.index_select(2, upper_idx)
                        ufve = (pred_upper - target_upper).norm(dim=-1).mean()
                        # FDD（official / CodeTalker-style）：与 compute_objective_metrics.py 对齐
                        # motion = vertices - template_vertices
                        # energy(t, v) = ||motion(t, v)||^2
                        # motion_std = mean_v(std_t(energy(:, v)))
                        # FDD = mean_b(|motion_std_gt - motion_std_pred|)
                        if pred_upper.size(1) > 1:
                            template_flat = None
                            if template is not None:
                                if template.dim() == 3 and template.size(1) == 1:
                                    template_flat = template[:, 0]
                                elif template.dim() == 2:
                                    template_flat = template
                                else:
                                    template_flat = template.reshape(template.size(0), -1)

                            template_vertices = None
                            if template_flat is not None and hasattr(loss_module, 'get_vertices'):
                                template_vertices = loss_module.get_vertices(
                                    template_flat.detach(), annot_type)

                            if isinstance(template_vertices, torch.Tensor) and template_vertices.numel() > 0:
                                template_vertices = template_vertices.to(
                                    pred_vertices.device).unsqueeze(1)  # [B, 1, V, 3]
                                motion_pred = pred_vertices - template_vertices
                                motion_gt = target_vertices - template_vertices
                                motion_pred = motion_pred.index_select(2, upper_idx)
                                motion_gt = motion_gt.index_select(2, upper_idx)
                                energy_pred = motion_pred.pow(2).sum(dim=-1)
                                energy_gt = motion_gt.pow(2).sum(dim=-1)
                                pred_motion_std = energy_pred.std(
                                    dim=1, unbiased=False).mean(dim=-1)
                                gt_motion_std = energy_gt.std(
                                    dim=1, unbiased=False).mean(dim=-1)
                                fdd = (gt_motion_std -
                                       pred_motion_std).abs().mean()
                except NotImplementedError:
                    pass

            speed_weight = getattr(args, 'velocity_loss_weight', 0.0)
            if out_motion.shape[1] > 1:
                pred_vel = out_motion[:, 1:] - out_motion[:, :-1]
                gt_vel = data[:, 1:] - data[:, :-1]
                speed_loss = torch.nn.functional.mse_loss(pred_vel, gt_vel)
            else:
                speed_loss = zero_loss

            acc_weight = getattr(args, 'acceleration_loss_weight', 0.0)
            if out_motion.shape[1] > 2:
                pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
                gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
                acc_loss = torch.nn.functional.mse_loss(pred_acc, gt_acc)
            else:
                acc_loss = zero_loss

            # 口型动态（与 train_epoch 保持一致）
            mouth_speed_weight = float(
                getattr(args, 'mouth_velocity_loss_weight', 0.0) or 0.0)
            mouth_acc_weight = float(
                getattr(args, 'mouth_acceleration_loss_weight', 0.0) or 0.0)
            mouth_under_std_weight = float(
                getattr(args, 'mouth_under_std_weight', 0.0) or 0.0)
            mouth_speed_loss = zero_loss
            mouth_acc_loss = zero_loss
            mouth_under_std_loss = zero_loss

            mouth_idx = getattr(loss_module, 'mouth_bs_idx', None)
            if isinstance(mouth_idx, torch.Tensor) and mouth_idx.numel() > 0:
                mouth_idx = mouth_idx.to(out_motion.device)
                pred_mouth = out_motion.index_select(2, mouth_idx)
                gt_mouth = data.index_select(2, mouth_idx)

                if mouth_under_std_weight > 0.0:
                    pred_std = pred_mouth.std(dim=1, unbiased=False)
                    gt_std = gt_mouth.std(dim=1, unbiased=False)
                    mouth_under_std_loss = torch.relu(gt_std -
                                                      pred_std).mean()

                if out_motion.shape[1] > 1:
                    pred_m_vel = pred_mouth[:, 1:] - pred_mouth[:, :-1]
                    gt_m_vel = gt_mouth[:, 1:] - gt_mouth[:, :-1]
                    if mouth_speed_weight > 0.0:
                        mouth_speed_loss = torch.nn.functional.mse_loss(
                            pred_m_vel, gt_m_vel)
                    if out_motion.shape[1] > 2 and mouth_acc_weight > 0.0:
                        pred_m_acc = pred_m_vel[:, 1:] - pred_m_vel[:, :-1]
                        gt_m_acc = gt_m_vel[:, 1:] - gt_m_vel[:, :-1]
                        mouth_acc_loss = torch.nn.functional.mse_loss(
                            pred_m_acc, gt_m_acc)

            if out_pca is not None:
                pca_loss = loss_module.pca_loss(out_pca, gt_pca)
            else:
                pca_loss = zero_loss
            lve_region_loss = zero_loss
            ufve_region_loss = zero_loss
            w_lve = float(getattr(args, 'w_lve', 0.0) or 0.0)
            w_ufve = float(getattr(args, 'w_ufve', 0.0) or 0.0)
            if w_lve > 0.0 or w_ufve > 0.0:
                lve_region_loss, ufve_region_loss = compute_region_vertex_losses(
                    loss_module, out_motion, data, annot_type)

            total_loss = (
                rec_loss + args.pca_weight * pca_loss +
                speed_weight * speed_loss + acc_weight * acc_loss +
                mouth_speed_weight * mouth_speed_loss +
                mouth_acc_weight * mouth_acc_loss +
                mouth_under_std_weight * mouth_under_std_loss + w_lve *
                lve_region_loss + w_ufve * ufve_region_loss
            )

            # 将所有指标统一打包，便于循环入表
            metrics_values = (
                total_loss,
                mouth_metric_L2,
                mouth_metric_L2_norm,
                mve,
                ufve,
                fdd,
            )
            for k, v in zip(metric_name_list, metrics_values):
                val_meter[dataset_name][k].update(v.item())
                val_meter[mixed_dataset_name][k].update(v.item())

            if save_sequence_stats and per_vertex_error is not None:
                batch_size = per_vertex_error.shape[0]
                for idx in range(batch_size):
                    sample_dataset = dataset_name_list[idx] if idx < len(
                        dataset_name_list) else dataset_name
                    sample_annot_type = annot_type_list[idx] if idx < len(
                        annot_type_list) else annot_type
                    sample_annot_path = None
                    if annot_path_list is not None:
                        if isinstance(annot_path_list, (list, tuple)):
                            sample_annot_path = annot_path_list[idx]
                        else:
                            sample_annot_path = annot_path_list
                    sample_audio_path = None
                    if audio_path_list is not None:
                        if isinstance(audio_path_list, (list, tuple)):
                            sample_audio_path = audio_path_list[idx]
                        else:
                            sample_audio_path = audio_path_list
                    # 将当前序列的所有统计指标打包保存，便于后续绘制速度/误差热力图
                    payload = {
                        'dataset': np.array(sample_dataset),
                        'annot_type': np.array(sample_annot_type),
                        'per_vertex_error':
                        np.asarray(per_vertex_error[idx].detach().cpu().tolist(),
                                   dtype=np.float32),
                    }
                    if sample_annot_path is not None:
                        payload['annot_path'] = np.array(sample_annot_path)
                    if sample_audio_path is not None:
                        payload['audio_path'] = np.array(sample_audio_path)
                    if velocity_mean_pred is not None:
                        payload['velocity_mean_pred'] = np.asarray(
                            velocity_mean_pred[idx].detach().cpu().tolist(),
                            dtype=np.float32)
                        payload['velocity_std_pred'] = np.asarray(
                            velocity_std_pred[idx].detach().cpu().tolist(),
                            dtype=np.float32)
                        payload['velocity_mean_gt'] = np.asarray(
                            velocity_mean_gt[idx].detach().cpu().tolist(),
                            dtype=np.float32)
                        payload['velocity_std_gt'] = np.asarray(
                            velocity_std_gt[idx].detach().cpu().tolist(),
                            dtype=np.float32)
                    file_name = f'{phase}_epoch{epoch:03d}_{sequence_counter:05d}.npz'
                    np.savez_compressed(os.path.join(stats_dir, file_name),
                                        **payload)
                    sequence_counter += 1
        if getattr(args, 'logger', None) is not None:
            log_datasetloss(
                args.logger, epoch, phase, val_meter, only_mix=True)
        if getattr(args, 'writer', None) is not None:
            write_to_tensorboard(
                args.writer, epoch, phase, val_meter, only_mix=False)
        for dataset_name, sub_metric_dict in val_meter.items():
            for metric_name in sub_metric_dict.keys():
                sub_metric_dict[metric_name].reset()
    return
