import numpy as np
import pickle
import torch
from dataclasses import dataclass
from smplx.lbs import lbs
from smplx.utils import Struct, to_np, to_tensor
from torch import nn

from EATalker.dataset.dataset_config import dataset_config

_NUMPY_COMPAT_ALIASES = (
    ('bool', bool),
    ('int', int),
    ('float', float),
    ('complex', complex),
    ('object', object),
    ('unicode', str),
    ('str', str),
)
for _alias, _target in _NUMPY_COMPAT_ALIASES:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

def _get_inhouse_mouth_bs_idx():
    """Return mouth-related ARKit-51 blendshape indices (D6 order)."""
    d6_order = [
        'EyeBlinkLeft', 'EyeBlinkRight', 'EyeSquintLeft', 'EyeSquintRight',
        'EyeLookDownLeft', 'EyeLookDownRight', 'EyeLookInLeft', 'EyeLookInRight',
        'EyeWideLeft', 'EyeWideRight', 'EyeLookOutLeft', 'EyeLookOutRight',
        'EyeLookUpLeft', 'EyeLookUpRight', 'BrowDownLeft', 'BrowDownRight',
        'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'JawOpen',
        'MouthClose', 'JawLeft', 'JawRight', 'JawForward', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
        'MouthRollUpper', 'MouthRollLower', 'MouthSmileLeft', 'MouthSmileRight',
        'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
        'MouthFrownLeft', 'MouthFrownRight', 'MouthPressLeft', 'MouthPressRight',
        'MouthPucker', 'MouthFunnel', 'MouthLeft', 'MouthRight',
        'MouthShrugLower', 'MouthShrugUpper', 'NoseSneerLeft', 'NoseSneerRight',
        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight'
    ]
    mouth_names = {
        'JawOpen', 'JawLeft', 'JawRight', 'JawForward', 'MouthClose',
        'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight',
        'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
        'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
        'MouthRollUpper', 'MouthRollLower', 'MouthShrugLower', 'MouthShrugUpper',
        'MouthPressLeft', 'MouthPressRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
        'MouthLowerDownLeft', 'MouthLowerDownRight'
    }
    return [i for i, name in enumerate(d6_order) if name in mouth_names]


@dataclass
class FlameParams:
    betas: torch.Tensor
    jaw: torch.Tensor
    eyeballs: torch.Tensor
    neck: torch.Tensor


class FlameLayer(nn.Module):

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.dtype = torch.float32
        with open(model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer('shapedirs',
                             to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(
            to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)
        self.register_buffer(
            'v_template',
            to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype),
        )
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs,
                              [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))
        default_neck_pose = torch.zeros([1, 3],
                                        dtype=self.dtype,
                                        requires_grad=False)
        self.register_parameter(
            'neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        default_eyeball_pose = torch.zeros([1, 6],
                                           dtype=self.dtype,
                                           requires_grad=False)
        self.register_parameter(
            'eyeballs',
            nn.Parameter(default_eyeball_pose, requires_grad=False))
        default_jaw = torch.zeros([1, 3],
                                  dtype=self.dtype,
                                  requires_grad=False)
        self.register_parameter('jaw',
                                nn.Parameter(default_jaw, requires_grad=False))

        self.FLAME_CONSTS = {
            'betas': 400,
            'rotation': 6,
            'jaw': 3,
            'eyeballs': 0,
            'neck': 0,
            'translation': 3,
            'scale': 1,
        }

    def flame_params_from_3dmm(self,
                               tensor_3dmm: torch.Tensor,
                               zero_expr: bool = False):
        constants = self.FLAME_CONSTS

        assert tensor_3dmm.ndim == 2

        cur_index: int = 0
        betas = tensor_3dmm[:, :constants['betas']]
        cur_index += constants['betas']
        jaw = tensor_3dmm[:, cur_index:cur_index + constants['jaw']]
        cur_index += constants['jaw']
        cur_index += constants['rotation']
        eyeballs = tensor_3dmm[:, cur_index:cur_index + constants['eyeballs']]
        cur_index += constants['eyeballs']
        neck = tensor_3dmm[:, cur_index:cur_index + constants['neck']]
        cur_index += constants['neck']
        cur_index += constants['translation']
        cur_index += constants['scale']
        return FlameParams(betas=betas, jaw=jaw, eyeballs=eyeballs, neck=neck)

    def forward(self, tensor_3dmm: torch.Tensor):
        bs = tensor_3dmm.shape[0]
        flame_params = self.flame_params_from_3dmm(tensor_3dmm)
        betas = flame_params.betas
        rotation = torch.zeros([bs, 3], device=tensor_3dmm.device)
        neck_pose = flame_params.neck if not (
            0 in flame_params.neck.shape) else self.neck_pose[[0]].expand(
                bs, -1)
        eyeballs = flame_params.eyeballs if not (
            0 in flame_params.eyeballs.shape) else self.eyeballs[[0]].expand(
                bs, -1)
        jaw = flame_params.jaw if not (
            0 in flame_params.jaw.shape) else self.jaw[[0]].expand(bs, -1)

        full_pose = torch.cat([rotation, neck_pose, jaw, eyeballs], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(bs, 1, 1)
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )
        return vertices


class FlameParamLossForDADHead(nn.Module):

    def __init__(self, mouth_indices_path: str, loss_mask_path: str,
                 flame_model_path: str) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path).astype(np.int64)
        loss_indices = np.load(loss_mask_path).astype(np.int64)
        self.register_buffer('mouth_idx',
                             torch.tensor(mouth_indices, dtype=torch.long))
        self.register_buffer('loss_indices',
                             torch.tensor(loss_indices, dtype=torch.long))
        self.mse = nn.MSELoss()
        self.flame_layer = FlameLayer(flame_model_path)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        x = self.flame_layer(x.reshape(-1, ori_shape[-1]))
        target = self.flame_layer(target.reshape(-1, ori_shape[-1]))
        x = x.reshape((ori_shape[0], ori_shape[1], -1, 3))
        target = target.reshape((ori_shape[0], ori_shape[1], -1, 3))
        return self.mse(x[:, :, self.loss_indices], target[:, :,
                                                           self.loss_indices])

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        x = self.flame_layer(x.reshape(-1, x.shape[-1]))
        return x[:, self.loss_indices]

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        x = self.flame_layer(x.reshape(-1, ori_shape[-1]))
        target = self.flame_layer(target.reshape(-1, ori_shape[-1]))
        metric_L2 = (target[:, self.mouth_idx] - x[:, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class VerticesLoss(nn.Module):

    def __init__(self, mouth_indices_path: str) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path).astype(np.int64)
        self.register_buffer('mouth_idx',
                             torch.tensor(mouth_indices, dtype=torch.long))
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        return x.reshape(-1, x.shape[-1] // 3, 3)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        target = target.reshape(*ori_shape[:-1], -1, 3)
        x = x.reshape(*ori_shape[:-1], -1, 3)
        metric_L2 = (target[:, :, self.mouth_idx] - x[:, :, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class BlendShapeLoss(nn.Module):

    def __init__(self,
                 mouth_indices_path: str,
                 blendshape_path: str,
                 bs_beta: float = 0.0,
                 components_num: int = None) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path).astype(np.int64)
        blendshape = np.load(blendshape_path)
        self.register_buffer('mouth_idx',
                             torch.tensor(mouth_indices, dtype=torch.long))
        mean_shape = blendshape['meanshape']
        mean_shape = mean_shape.reshape(-1).astype(np.float32)
        blend_shape = blendshape['blendshape']
        blend_shape = blend_shape.reshape(len(blend_shape),
                                          -1).astype(np.float32)
        self.register_buffer(
            'mean_shape', torch.tensor(mean_shape, dtype=torch.float32))
        self.register_buffer(
            'blend_shape', torch.tensor(blend_shape, dtype=torch.float32))
        self.bs_beta = bs_beta
        self.mse = nn.MSELoss()
        self.components_num = components_num
        total_vertices = mean_shape.size // 3
        if 'lower_face_idx' in blendshape.files:
            lower_face_idx = blendshape['lower_face_idx'].astype(np.int64)
        else:
            lower_face_idx = np.array([], dtype=np.int64)
        upper_face_idx = np.setdiff1d(
            np.arange(total_vertices, dtype=np.int64), lower_face_idx)
        self.register_buffer(
            'lower_face_idx',
            torch.tensor(lower_face_idx, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            'upper_face_idx',
            torch.tensor(upper_face_idx, dtype=torch.long),
            persistent=False)
        self.l1 = nn.SmoothL1Loss(beta=0.1)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.bs_beta * self.l1(x, target) + self.mse(
            self.bs2vertices(x), self.bs2vertices(target))

    def bs2vertices(self, x: torch.Tensor):
        # 调试：前几次调用打印 bs / verts 范围，便于确认训练/推理时数值正常
        if not hasattr(self, "_dbg_bs2v_cnt"):
            self._dbg_bs2v_cnt = 0
        if self._dbg_bs2v_cnt < 3:
            xm = x.detach()
            try:
                bs_min, bs_max = xm.min().item(), xm.max().item()
            except Exception:
                bs_min = bs_max = None
            verts_tmp = xm[:, :self.components_num] @ self.blend_shape[:self.components_num] + self.mean_shape
            try:
                v_min, v_max = verts_tmp.min().item(), verts_tmp.max().item()
            except Exception:
                v_min = v_max = None
            print(f"[bs2v dbg] call {self._dbg_bs2v_cnt}: bs min/max {bs_min}/{bs_max} -> verts min/max {v_min}/{v_max}")
        self._dbg_bs2v_cnt += 1
        return x[:, :self.components_num] @ \
            self.blend_shape[:self.components_num] + self.mean_shape

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        x = self.bs2vertices(x)
        return x.reshape(-1, x.shape[-1] // 3, 3)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        target = self.bs2vertices(target)
        x = self.bs2vertices(x)
        ori_shape = x.shape
        target = target.reshape(*ori_shape[:-1], -1, 3)
        x = x.reshape(*ori_shape[:-1], -1, 3)
        metric_L2 = (target[:, :, self.mouth_idx] - x[:, :, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class ParamLoss(nn.Module):

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss(beta=0.1)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.weight * self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target), self.L1(x, target)

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        return x.squeeze(0)


class EATalkerLoss(nn.Module):
    """EATalker loss combining blendshape parameters and vertices; PCA branch removed."""

    def __init__(self, args) -> None:
        super().__init__()
        loss_config = {
            'flame_params_from_dadhead': {
                'class': FlameParamLossForDADHead,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/02_flame_mouth_idx.npy',
                    'loss_mask_path':
                    'resources/binary_resources/flame_humanface_index.npy',
                    'flame_model_path': 'resources/binary_resources/flame.pkl',
                }
            },
            '3DETF_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/03_emo_talk_mouth_idx.npy',
                    'blendshape_path':
                    'resources/binary_resources/EmoTalk.npz',
                    'bs_beta': args.blendshape_weight,
                },
            },
            'inhouse_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/05_inhouse_arkit_mouth_idx.npy',
                    'blendshape_path':
                    'resources/binary_resources/inhouse_arkit.npz',
                    'bs_beta': args.blendshape_weight,
                },
            },
            'FLAME_5023_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/02_flame_mouth_idx.npy',
                },
            },
            'BIWI_23370_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/04_BIWI_mouth_idx.npy',
                },
            },
            'FLAME_SUB_2055_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/mouth_idx_FLAME_5023_vertices_wo_eyes_wo_head.npy',
                },
            },
            'meshtalk_6172_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/06_mesh_talk_mouth_idx.npy',
                },
            },
            '24_viseme': {
                'class': ParamLoss,
                'args': {
                    'weight': args.blendshape_weight
                },
            }
        }

        # 仅构建当前配置实际需要的损失模块，避免加载未安装依赖的组件
        required_annot_types = set()
        dataset_name_sources = []
        for attr in ('dataset', 'demo_dataset'):
            values = getattr(args, attr, None)
            if values is None:
                continue
            if isinstance(values, str):
                values = values.split(',')
            dataset_name_sources.extend(values)
        for name in dataset_name_sources:
            if name in dataset_config:
                required_annot_types.add(
                    dataset_config[name]['annot_type'])
        if not required_annot_types:
            required_annot_types = set(loss_config.keys())

        loss_module_dict = {}
        for annot_type, cfg in loss_config.items():
            if annot_type not in required_annot_types:
                continue
            try:
                loss_module = cfg['class'](**cfg['args'])
            except (FileNotFoundError, ImportError, ValueError) as exc:
                raise RuntimeError(
                    f'加载损失模块 {annot_type} 失败，请确认依赖与资源文件存在: {exc}'
                ) from exc
            loss_module_dict[annot_type] = loss_module
        self.loss_module_dict = nn.ModuleDict(loss_module_dict)
        self.mse = nn.MSELoss()
        self.blendshape_weight = args.blendshape_weight
        self.mouth_param_weight = float(
            getattr(args, 'mouth_param_weight', 1.0) or 1.0)
        self.mouth_vertex_weight = float(
            getattr(args, 'mouth_vertex_weight', 0.0) or 0.0)
        mouth_bs_idx = _get_inhouse_mouth_bs_idx()
        self.register_buffer('mouth_bs_idx',
                             torch.tensor(mouth_bs_idx, dtype=torch.long),
                             persistent=False)
        self.latest_loss_components = {}

    def _weighted_param_loss(self, x: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        beta = 0.1
        diff = x - target
        abs_diff = diff.abs()
        loss = torch.where(abs_diff < beta,
                           0.5 * diff * diff / beta,
                           abs_diff - 0.5 * beta)
        weights = torch.ones_like(loss)
        weights[..., self.mouth_bs_idx] = self.mouth_param_weight
        return (loss * weights).mean()

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        annot_type: str,
    ):
        """Return total loss; for inhouse blendshape we supervise parameters and vertices."""
        if annot_type == 'inhouse_blendshape_weight':
            blendshape_loss: BlendShapeLoss = self.loss_module_dict[annot_type]
            if self.mouth_param_weight != 1.0:
                param_loss = self._weighted_param_loss(x, target)
            else:
                param_loss = blendshape_loss.l1(x, target)
            pred_vertices = blendshape_loss.bs2vertices(x)
            target_vertices = blendshape_loss.bs2vertices(target)
            vertex_loss = blendshape_loss.mse(pred_vertices, target_vertices)
            mouth_vertex_loss = None
            if self.mouth_vertex_weight > 0.0 and hasattr(
                    blendshape_loss, 'mouth_idx'):
                try:
                    # pred_vertices: [B,T,V*3]  -> [B,T,V,3]
                    B, T = x.shape[:2]
                    pred_v = pred_vertices.reshape(B, T, -1, 3)
                    tgt_v = target_vertices.reshape(B, T, -1, 3)
                    mouth_idx = blendshape_loss.mouth_idx.to(pred_v.device)
                    pred_m = pred_v.index_select(2, mouth_idx)
                    tgt_m = tgt_v.index_select(2, mouth_idx)
                    mouth_vertex_loss = self.mse(pred_m, tgt_m)
                except Exception:
                    mouth_vertex_loss = None

            total_loss = vertex_loss + self.blendshape_weight * param_loss
            if mouth_vertex_loss is not None:
                total_loss = total_loss + self.mouth_vertex_weight * mouth_vertex_loss
            self.latest_loss_components = {
                'vertex_loss': vertex_loss.detach(),
                'param_loss': param_loss.detach(),
            }
            if mouth_vertex_loss is not None:
                self.latest_loss_components['mouth_vertex_loss'] = mouth_vertex_loss.detach(
                )
            return total_loss

        self.latest_loss_components = {}
        return self.loss_module_dict[annot_type](x, target)

    def pca_loss(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor,
                     annot_type: str):
        return self.loss_module_dict[annot_type].mouth_metric(x, target)

    def get_vertices(self, x: torch.Tensor, annot_type: str):
        return self.loss_module_dict[annot_type].get_vertices(x)

    def get_vertices_sequence(self, x: torch.Tensor, annot_type: str):
        module = self.loss_module_dict[annot_type]
        B, T = x.shape[:2]
        x_flat = x.reshape(-1, x.shape[-1])
        if isinstance(module, (VerticesLoss, BlendShapeLoss,
                               FlameParamLossForDADHead)):
            vert_flat = module.get_vertices(x_flat)
        else:
            raise NotImplementedError(f'Unsupported annot_type: {annot_type}')
        num_vertices = vert_flat.shape[1]
        vertices = vert_flat.view(B, T, num_vertices, 3)
        return vertices

    def get_upper_face_indices(self, annot_type: str):
        module = self.loss_module_dict[annot_type]
        upper_idx = getattr(module, 'upper_face_idx', None)
        if isinstance(upper_idx, torch.Tensor) and upper_idx.numel() > 0:
            return upper_idx
        return None


class UniTalkerLoss(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        loss_config = {
            'flame_params_from_dadhead': {
                'class': FlameParamLossForDADHead,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/02_flame_mouth_idx.npy',
                    'loss_mask_path':
                    'resources/binary_resources/flame_humanface_index.npy',
                    'flame_model_path': 'resources/binary_resources/flame.pkl',
                }
            },
            '3DETF_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/03_emo_talk_mouth_idx.npy',
                    'blendshape_path':
                    'resources/binary_resources/EmoTalk.npz',
                    'bs_beta': args.blendshape_weight,
                },
            },
            'inhouse_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/05_inhouse_arkit_mouth_idx.npy',
                    'blendshape_path':
                    'resources/binary_resources/inhouse_arkit.npz',
                    'bs_beta': args.blendshape_weight,
                },
            },
            'FLAME_5023_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/02_flame_mouth_idx.npy',
                },
            },
            'BIWI_23370_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/04_BIWI_mouth_idx.npy',
                },
            },
            'FLAME_SUB_2055_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/mouth_idx_FLAME_5023_vertices_wo_eyes_wo_head.npy',
                },
            },
            'meshtalk_6172_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'resources/binary_resources/06_mesh_talk_mouth_idx.npy',
                },
            },
            '24_viseme': {
                'class': ParamLoss,
                'args': {
                    'weight': args.blendshape_weight
                },
            }
        }

        loss_module_dict = {}
        for k, v in loss_config.items():
            loss_module = v['class'](**v['args'])
            loss_module_dict[k] = loss_module
        self.loss_module_dict = nn.ModuleDict(loss_module_dict)
        self.mse = nn.MSELoss()
        return

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        annot_type: str,
    ):
        return self.loss_module_dict[annot_type](x, target)

    def pca_loss(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor,
                     annot_type: str):
        return self.loss_module_dict[annot_type].mouth_metric(x, target)

    def get_vertices(self, x: torch.Tensor, annot_type: str):
        return self.loss_module_dict[annot_type].get_vertices(x)
