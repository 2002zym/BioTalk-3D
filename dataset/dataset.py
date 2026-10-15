import json
import numpy as np
import os.path as osp
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from typing import Any, List

from .data_item import DataItem
from .dataset_config import dataset_config

def _resolve_path(root: str, relative_path: str) -> str:
    """Return an OS-specific absolute path for entries that may use Windows slashes."""
    return osp.normpath(osp.join(root, relative_path.replace('\\', '/')))


def _load_cache_manifest(manifest_path: str) -> dict:
    if not manifest_path:
        return {}
    if not osp.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict):
        if isinstance(payload.get('entries'), dict):
            return payload['entries']
        return payload
    return {}


class AudioFaceDataset(data.Dataset):

    def __init__(self, data_label_path: str, args: dict = None) -> None:
        super().__init__()
        data_root = osp.dirname(data_label_path, )
        id_template_path = osp.join(data_root, 'id_template.npy')
        id_template_list = np.load(id_template_path)
        with open(data_label_path) as f:
            labels = json.load(f)
        info = labels['info']
        self.id_list = info['id_list']
        self.sr = 16000
        use_cached = bool(getattr(args, 'use_cached_features', False))
        cache_manifest_path = getattr(args, 'cache_manifest', '')
        cache_entries = _load_cache_manifest(
            str(cache_manifest_path) if use_cached else '')
        data_info_list = labels['data']
        data_list = []
        for data_info in tqdm(data_info_list):
            # 获取EEG路径（如果存在）
            eeg_path = None
            if 'eeg_path' in data_info:
                eeg_path = _resolve_path(data_root, data_info['eeg_path'])

            metadata_fields = (
                'sample_id',
                'subject',
                'session',
                'sheet',
                'language',
                'timestamps',
                'text',
            )
            metadata = {
                k: data_info.get(k, {} if k == 'timestamps' else '')
                for k in metadata_fields
            }
            sample_id = data_info.get('sample_id', '')
            subject = data_info.get('subject', '')
            cache_key = f"{subject}::{sample_id}" if subject else sample_id
            cache_entry = {}
            if use_cached:
                cache_entry = cache_entries.get(cache_key, {})
                if not cache_entry:
                    cache_entry = cache_entries.get(sample_id, {})
            blendshape_path = data_info.get('blendshape_path',
                                            data_info.get('annot_path'))

            data = DataItem(
                annot_path=_resolve_path(data_root, data_info['annot_path']),
                audio_path=_resolve_path(data_root, data_info['audio_path']),
                identity_idx=data_info['id'],
                annot_type=data_info['annot_type'],
                dataset_name=data_info['dataset'],
                id_template=id_template_list[data_info['id']],
                fps=data_info['fps'],
                processor=args.processor,
                eeg_path=eeg_path,
                use_audio=getattr(args, 'use_audio', True),
                use_eeg=getattr(args, 'use_eeg', True),
                mouth_gain=getattr(args, 'mouth_gain', 1.0),
                sample_id=sample_id,
                subject=data_info.get('subject', ''),
                session=data_info.get('session', ''),
                sheet=data_info.get('sheet', ''),
                language=data_info.get('language', ''),
                timestamps=data_info.get('timestamps', {}),
                blendshape_path=_resolve_path(data_root, blendshape_path),
                metadata=metadata,
                cache_entry=cache_entry,
            )
            if data.duration < 0.5:
                continue
            data_list.append(data)
        self.data_list = data_list
        return

    def __len__(self, ):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:
        data = self.data_list[index]
        return data.get_dict()

    def get_identity_num(self, ):
        return len(self.id_list)


class MixAudioFaceDataset(AudioFaceDataset):

    def __init__(self,
                 dataset_list: List[AudioFaceDataset],
                 duplicate_list: list = None) -> None:
        super(AudioFaceDataset).__init__()
        self.id_list = []
        self.sr = 16000
        self.data_list = []
        self.annot_type_list = []
        if duplicate_list is None:
            duplicate_list = [1] * len(dataset_list)
        else:
            assert len(duplicate_list) == len(dataset_list)
        for dup, dataset in zip(duplicate_list, dataset_list):
            id_index_offset = len(self.id_list)
            for d in dataset.data_list:
                d.offset_id(id_index_offset)
                if d.annot_type not in self.annot_type_list:
                    self.annot_type_list.append(d.annot_type)
            self.id_list = self.id_list + dataset.id_list
            self.data_list = self.data_list + dataset.data_list * dup
        return


def get_dataset_list(args):
    dataset_name_list = args.dataset
    dataset_dir_list = [
        dataset_config[name]['dirname'] for name in dataset_name_list
    ]
    train_json_list = [
        osp.join(args.data_root, dataset_name, args.train_json_key)
        for dataset_name in dataset_dir_list
    ]
    args.read_audio = False
    args.processor = None
    train_dataset_list = []
    # 仅在启用音频时构建特征提取器，避免不必要的开销
    if getattr(args, 'use_audio', True):
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.audio_encoder_repo)
        args.processor = processor
    else:
        args.processor = None
    for train_json in train_json_list:
        dataset = AudioFaceDataset(
            train_json,
            args,
        )
        train_dataset_list.append(dataset)
    return train_dataset_list


def get_single_dataset(args, json_path: str = ''):
    args.read_audio = True
    if getattr(args, 'use_audio', True):
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.audio_encoder_repo)
        args.processor = processor
    else:
        args.processor = None

    dataset = AudioFaceDataset(
        json_path,
        args,
    )
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers)
    return test_loader


def get_dataloaders(args):
    dataset_name_list = args.dataset
    print(dataset_name_list)
    dataset_dir_list = [
        dataset_config[name]['dirname'] for name in dataset_name_list
    ]
    train_json_list = [
        osp.join(args.data_root, dataset_dir, args.train_json_key)
        for dataset_dir in dataset_dir_list
    ]
    val_json_list = [
        osp.join(args.data_root, dataset_dir, 'val.json')
        for dataset_dir in dataset_dir_list
    ]
    test_json_list = [
        osp.join(args.data_root, dataset_dir, 'test.json')
        for dataset_dir in dataset_dir_list
    ]
    if getattr(args, 'use_audio', True):
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.audio_encoder_repo)
        args.processor = processor
    else:
        args.processor = None
    if getattr(args, 'do_train', True) is True:
        train_dataset_list = []
        for train_json in train_json_list:
            dataset = AudioFaceDataset(train_json, args)
            train_dataset_list.append(dataset)
        mix_train_dataset = MixAudioFaceDataset(train_dataset_list,
                                                args.duplicate_list)
        train_sampler = DistributedSampler(
            mix_train_dataset) if getattr(args, 'distributed', False) else None
        train_loader = data.DataLoader(
            dataset=mix_train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            num_workers=args.workers)
    else:
        train_loader = None
    if getattr(args, 'do_validate', True) is True:
        val_dataset_list = []
        for val_json in val_json_list:
            dataset = AudioFaceDataset(val_json, args)
            val_dataset_list.append(dataset)
        mix_val_dataset = MixAudioFaceDataset(val_dataset_list)
        val_loader = data.DataLoader(
            dataset=mix_val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers)
    else:
        val_loader = None
    if getattr(args, 'do_test', True) is True:
        test_dataset_list = []
        for test_json in test_json_list:
            dataset = AudioFaceDataset(test_json, args)
            test_dataset_list.append(dataset)
        mix_test_dataset = MixAudioFaceDataset(test_dataset_list)
        test_loader = data.DataLoader(
            dataset=mix_test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader
