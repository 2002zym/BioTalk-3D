#!/usr/bin/env python
# EATalker专用训练脚本
# 基于UniTalker训练脚本，支持EEG+音频双模态输入

import os

import sys
from pathlib import Path
import torch
import torch.optim
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# 确保项目根目录及其父目录在 Python 路径中，便于直接运行脚本
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_PARENT = PROJECT_ROOT.parent
for path in (PROJECT_PARENT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from EATalker.dataset.dataset import get_dataloaders
from EATalker.loss.loss import EATalkerLoss  # 使用EATalker专用损失函数
from EATalker.models.eatalker import EATalker  # 使用EATalker模型
from EATalker.utils.utils import (
    get_average_meter_dict, get_logger, get_parser, get_audio_encoder_dim,
    load_ckpt, seed_everything, filter_unitalker_state_dict
)
from EATalker.main.train_eval_loop import train_epoch, validate_epoch


class _DummyLogger:
    def info(self, *args, **kwargs): ...
    def warning(self, *args, **kwargs): ...
    def error(self, *args, **kwargs): ...
    def debug(self, *args, **kwargs): ...


class _DummyWriter:
    def add_scalar(self, *args, **kwargs): ...
    def close(self): ...


def _log_param_groups(logger, optimizer, prefix: str = ''):
    """打印优化器参数组的 LR 与规模，便于对齐训练日志。"""
    try:
        param_groups = optimizer.param_groups
    except AttributeError:
        return
    total = 0
    logger.info(f'{prefix}优化器参数组:')
    for idx, group in enumerate(param_groups):
        params = group.get('params', [])
        name = group.get('name', f'group{idx}')
        lr = group.get('lr', None)
        num_tensors = len(params)
        num_elements = sum(p.numel() for p in params if getattr(p, 'requires_grad', False))
        total += num_elements
        if lr is not None:
            logger.info(
                f'{prefix}  - {name}: LR={lr:.1e}, params={num_tensors}, elements={num_elements/1e6:.3f} M')
        else:
            logger.info(
                f'{prefix}  - {name}: params={num_tensors}, elements={num_elements/1e6:.3f} M')
    logger.info(f'{prefix}  总计可训练元素: {total/1e6:.3f} M')

def _build_optimizer(raw_model, args):
    """Build Adam optimizer with differential LR groups.

    Groups:
    - decoder / out_head: base LR
    - audio_encoder (wav2vec/wavlm): very small LR (optional, for middle-route fine-tune)
    - eeg_encoder / fusion / other: high LR (new modules)
    """
    base_lr = float(args.lr)
    eeg_fusion_lr = getattr(args, 'eeg_fusion_lr', None)
    eeg_fusion_lr_scale = float(getattr(args, 'eeg_fusion_lr_scale', 10.0))
    other_lr_scale = float(getattr(args, 'other_lr_scale',
                                   eeg_fusion_lr_scale))
    if eeg_fusion_lr is None:
        high_lr = base_lr * eeg_fusion_lr_scale
    else:
        high_lr = float(eeg_fusion_lr)
    other_lr = base_lr * other_lr_scale

    wav2vec_lr = getattr(args, 'wav2vec_lr', None)
    if wav2vec_lr is None:
        last_n = int(getattr(args, 'wav2vec_train_last_n_layers', 0) or 0)
        freeze_all = bool(getattr(args, 'freeze_wav2vec', False))
        # Default heuristic:
        # - middle-route (last_n>0): very small LR to avoid catastrophic forgetting
        # - full fine-tune (freeze_wav2vec=False and last_n==0): use base LR
        wav2vec_lr = base_lr * 0.01 if (last_n > 0 or freeze_all) else base_lr
    wav2vec_lr = float(wav2vec_lr)
    if wav2vec_lr <= 0:
        raise ValueError(f'wav2vec_lr must be > 0, got {wav2vec_lr}')
    if high_lr <= 0:
        raise ValueError(f'eeg_fusion_lr must be > 0, got {high_lr}')
    if other_lr <= 0:
        raise ValueError(f'other_lr must be > 0, got {other_lr}')

    param_groups = []

    # 1) Decoder - pre-trained, base LR
    decoder_params = list(
        filter(lambda p: p.requires_grad, raw_model.decoder.parameters()))
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr,
            'name': 'decoder'
        })

    # 2) Output head - pre-trained, base LR
    head_params = list(
        filter(lambda p: p.requires_grad, raw_model.out_head_dict.parameters()))
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'name': 'out_head'
        })

    # 3) Audio encoder (wav2vec/wavlm) - very small LR (optional)
    if hasattr(raw_model, 'audio_encoder'):
        audio_enc_params = list(
            filter(lambda p: p.requires_grad, raw_model.audio_encoder.parameters()))
        if audio_enc_params:
            param_groups.append({
                'params': audio_enc_params,
                'lr': wav2vec_lr,
                'name': 'audio_encoder'
            })

    # 4) EEG encoder - new module, high LR
    if hasattr(raw_model, 'eeg_encoder') and raw_model.use_eeg:
        eeg_params = list(
            filter(lambda p: p.requires_grad, raw_model.eeg_encoder.parameters()))
        if eeg_params:
            param_groups.append({
                'params': eeg_params,
                'lr': high_lr,
                'name': 'eeg_encoder'
            })

    # 5) Multimodal fusion - new module, high LR
    if hasattr(raw_model, 'multimodal_fusion') and raw_model.use_eeg:
        fusion_params = list(
            filter(lambda p: p.requires_grad, raw_model.multimodal_fusion.parameters()))
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': high_lr,
                'name': 'fusion'
            })

    # 6) Other trainable params - configurable LR
    named_param_ids = set()
    for group in param_groups:
        for p in group['params']:
            named_param_ids.add(id(p))
    other_params = [
        p for p in raw_model.parameters()
        if p.requires_grad and id(p) not in named_param_ids
    ]
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': other_lr,
            'name': 'other'
        })

    optimizer = torch.optim.Adam(param_groups)
    return optimizer, param_groups


def _set_requires_grad(module, flag: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(flag)


def _set_decoder_trainable_last_k(raw_model, k_last: int):
    """Freeze decoder then unfreeze last-k logical layers (k=0 => all)."""
    decoder = getattr(raw_model, 'decoder', None)
    if decoder is None:
        return
    k_last = int(k_last or 0)
    _set_requires_grad(decoder, False)
    if k_last == 0:
        _set_requires_grad(decoder, True)
        return

    # Conv decoder path
    if hasattr(decoder, 'tcn') and hasattr(decoder.tcn, 'first_net') and hasattr(
            decoder.tcn.first_net, 'conv_layers'):
        layers = list(decoder.tcn.first_net.conv_layers)
        if layers:
            k = min(k_last, len(layers))
            for layer in layers[-k:]:
                _set_requires_grad(layer, True)
            if hasattr(decoder, 'multimodal_feature_map'):
                _set_requires_grad(decoder.multimodal_feature_map, True)
            if hasattr(decoder, 'learnable_style_emb'):
                _set_requires_grad(decoder.learnable_style_emb, True)
            return

    # Transformer decoder path
    if hasattr(decoder, 'transformer_decoder') and hasattr(
            decoder.transformer_decoder, 'layers'):
        layers = list(decoder.transformer_decoder.layers)
        if layers:
            k = min(k_last, len(layers))
            for layer in layers[-k:]:
                _set_requires_grad(layer, True)
            if hasattr(decoder, 'multimodal_feature_map'):
                _set_requires_grad(decoder.multimodal_feature_map, True)
            if hasattr(decoder, 'learnable_style_emb'):
                _set_requires_grad(decoder.learnable_style_emb, True)
            return

    # Fallback if decoder topology is unknown.
    _set_requires_grad(decoder, True)


def _apply_train_stage(raw_model, args, logger):
    """Apply explicit stage-wise freezing strategy for fast/stable multimodal tuning."""
    stage = str(getattr(args, 'train_stage', 'joint') or 'joint').lower()
    if stage not in ('joint', 'stage1', 'stage2'):
        raise ValueError(f'Unknown train_stage: {stage}')

    if stage == 'joint':
        logger.info('=> train_stage=joint (use existing freeze_wav2vec strategy)')
        return

    # Stage1/2 both keep audio encoder frozen.
    if hasattr(raw_model, 'audio_encoder'):
        _set_requires_grad(raw_model.audio_encoder, False)
        if hasattr(raw_model.audio_encoder, '_freeze_wav2vec2_parameters'):
            raw_model.audio_encoder._freeze_wav2vec2_parameters(True)

    # EEG branch is always trainable in stage-wise multimodal tuning.
    if hasattr(raw_model, 'eeg_encoder'):
        _set_requires_grad(raw_model.eeg_encoder, True)
    if hasattr(raw_model, 'multimodal_fusion'):
        _set_requires_grad(raw_model.multimodal_fusion, True)
    if hasattr(raw_model, 'eeg_to_audio'):
        _set_requires_grad(raw_model.eeg_to_audio, True)

    if stage == 'stage1':
        _set_requires_grad(getattr(raw_model, 'decoder', None), False)
        _set_requires_grad(getattr(raw_model, 'out_head_dict', None), False)
        logger.info(
            '=> train_stage=stage1: freeze audio+decoder+head, train eeg+fusion(+alpha)')
        return

    # stage2
    k_last = int(getattr(args, 'stage2_decoder_last_k', 0) or 0)
    _set_decoder_trainable_last_k(raw_model, k_last)
    unfreeze_head = bool(getattr(args, 'stage2_unfreeze_head', True))
    _set_requires_grad(getattr(raw_model, 'out_head_dict', None), unfreeze_head)
    logger.info(
        f'=> train_stage=stage2: freeze audio, unfreeze decoder last_k={k_last} '
        f'({"all" if k_last == 0 else "partial"}), '
        f'out_head_trainable={unfreeze_head}')


def main():
    """
    EATalker主训练函数

    功能概述：
    1. 初始化环境和参数
    2. 创建数据加载器（支持EEG+音频双模态）
    3. 构建EATalker模型（EEGNet + 音频编码器 + 多模态融合）
    4. 设置损失函数和优化器
    5. 执行训练循环（支持分阶段解冻编码器）
    6. 定期验证、测试和保存模型

    关键特点：
    - 支持双模态输入（EEG + 音频）
    - 分阶段训练策略（先冻结编码器，再解冻）
    - 支持传统EEGNet架构
    - 简化的损失函数（直接监督51维blendshape，模仿D6格式）
    """
    seed_everything(42)  # 固定随机种子确保可重复性
    args = get_parser()  # 解析命令行参数
    args.audio_encoder_feature_dim = get_audio_encoder_dim(args.audio_encoder_repo)  # 获取音频编码器输出维度

    # 分布式初始化（torchrun）
    distributed = False
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed = True
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        args.distributed = True
        args.local_rank = local_rank
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    else:
        args.distributed = False
        args.local_rank = 0
        args.world_size = 1
        args.rank = 0
    is_main = args.rank == 0

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(args)
    args.identity_num = train_loader.dataset.get_identity_num()
    identity_num_override = int(getattr(args, 'identity_num_override', 0) or 0)
    if identity_num_override > 0:
        args.identity_num = identity_num_override
    args.annot_type_list = train_loader.dataset.annot_type_list

    if args.random_id_prob > 0.0:
        args.identity_num = args.identity_num + 1

    # 权重/指标仍保存在 args.save_path；日志与 TensorBoard 事件改到 /home/zym/experiments/log
    os.makedirs(args.save_path, exist_ok=True)
    log_root = os.environ.get('EAT_LOG_DIR', '/home/zym/experiments/log')
    os.makedirs(log_root, exist_ok=True)
    log_file = os.path.join(log_root, 'train_eatalker.log')
    logger = get_logger(log_file) if is_main else _DummyLogger()
    writer = SummaryWriter(log_root) if is_main else _DummyWriter()

    logger.info('=> 创建EATalker模型 ...')

    # 创建EATalker模型（而非原始的UniTalker）
    model = EATalker(args)

    # 仅支持 GPU 训练，若无可用 CUDA 直接报错
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，当前脚本仅支持 GPU 训练")
    device = torch.device(f'cuda:{args.local_rank}')
    model = model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,  # 避免未参与前向的参数触发 reducer 错误
        )
    raw_model = model.module if hasattr(model, 'module') else model
    if args.weight_path is not None:
        logger.info(f'=> 加载预训练模型 {args.weight_path} ...')
        style_emb_init_indices = getattr(args, 'style_emb_init_indices', None)
        load_ckpt(raw_model,
                  args.weight_path,
                  args.re_init_decoder_and_head,
                  style_emb_init_indices=style_emb_init_indices)
    _apply_train_stage(raw_model, args, logger)

    if is_main:
        logger.info(args)
        raw_model.summary(logger)

    # 优化器 - 差异化学习率策略（支持 middle-route 微调 wav2vec 最后 N 层）
    optimizer, param_groups = _build_optimizer(raw_model, args)
    
    # 打印学习率配置
    if is_main:
        logger.info('==> 差异化学习率配置:')
        for group in param_groups:
            logger.info(f"    {group.get('name', 'unnamed')}: LR={group['lr']:.1e}, params={len(group['params'])}")
        _log_param_groups(logger, optimizer, prefix='==> ')

    # 使用EATalker专用损失函数
    loss_module = EATalkerLoss(args).to(device)

    # 设置训练和验证指标
    args.train_meter, args.train_metric_name_list = \
        get_average_meter_dict(args, 'train')
    args.val_meter, args.val_metric_name_list = \
        get_average_meter_dict(args, 'val')
    args.logger = logger
    args.writer = writer
    args.is_main = is_main

    # 训练循环
    for epoch in range(args.epochs):
        # 分布式 sampler 设定 epoch，保证各 rank shuffle 不同切片
        if getattr(train_loader, 'sampler', None) and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if args.fix_encoder_first and str(
                getattr(args, 'train_stage', 'joint')).lower() == 'joint':
            if epoch == 0:
                # 第一轮：只冻结音频编码器，训练其他所有模块
                raw_model.audio_encoder._freeze_wav2vec2_parameters(True)
                # 注意：EEGNet编码器保持可训练状态，这是EATalker的关键设计
                # 我们希望EEGNet能在第一阶段就学习到有意义的特征
                logger.info('=> 第一阶段：冻结音频编码器，训练EEGNet+融合模块+解码器')
                optimizer, _ = _build_optimizer(raw_model, args)
                if is_main:
                    _log_param_groups(logger, optimizer, prefix='=> [Stage1] ')
            elif epoch == 1:
                # 第二轮：根据策略解冻音频编码器
                # - middle-route: 仅解冻 Transformer 最后 N 层（wav2vec_train_last_n_layers>0）
                # - 否则：沿用 freeze_wav2vec 逻辑（freeze_wav2vec=False 时训练 Transformer，CNN 前端保持冻结）
                last_n = int(getattr(args, 'wav2vec_train_last_n_layers', 0) or 0)
                if last_n > 0 and hasattr(raw_model.audio_encoder,
                                         '_freeze_wav2vec2_except_last_n_layers'):
                    raw_model.audio_encoder._freeze_wav2vec2_except_last_n_layers(last_n)
                else:
                    raw_model.audio_encoder._freeze_wav2vec2_parameters(
                        args.freeze_wav2vec)
                    if hasattr(raw_model.audio_encoder,
                               'feature_extractor') and hasattr(
                                   raw_model.audio_encoder.feature_extractor,
                                   '_freeze_parameters'):
                        raw_model.audio_encoder.feature_extractor._freeze_parameters()

                logger.info('=> 第二阶段：开始端到端联合训练')
                optimizer, _ = _build_optimizer(raw_model, args)
                if is_main:
                    _log_param_groups(logger, optimizer, prefix='=> [Stage2] ')

        epoch_plus = epoch + 1

        # 训练阶段
        logger.info(f'Epoch {epoch_plus}: loss = vertex + {args.blendshape_weight:.4e} * param')
        train_epoch(train_loader, model, loss_module, optimizer, epoch_plus, args)

        # 验证阶段
        if is_main and val_loader is not None and epoch_plus % args.eval_every == 0:
            logger.info(f'=> 第{epoch_plus}轮验证')
            validate_epoch(val_loader, model, loss_module, epoch_plus, 'val', args)

        # 测试阶段（仅在存在 test_loader 时执行）
        if is_main and (test_loader is not None) and (epoch_plus % args.test_every == 0):
            logger.info(f'=> 第{epoch_plus}轮测试')
            validate_epoch(test_loader, model, loss_module, epoch_plus, 'test', args)

        # 保存模型
        if is_main and epoch_plus % args.save_every == 0:
            model_path = os.path.join(args.save_path, f'eatalker_{epoch_plus:03d}.pt')
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(filter_unitalker_state_dict(state_dict), model_path)
            logger.info(f'=> 模型已保存: {model_path}')

    if is_main:
        logger.info('=> 训练完成!')
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
