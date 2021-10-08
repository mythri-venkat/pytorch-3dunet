import collections
import os

import imageio
import numpy as np
import torch
import nibabel as nib
import json
import math

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats, sample_instances
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('Dataset')


def dsb_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DSB2018Dataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, mirror_padding=(0, 32, 32), expand_dims=True,
                 instance_ratio=None, random_seed=0):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        # use mirror padding only during the 'test' phase
        if phase in ['train', 'val']:
            mirror_padding = None
        if mirror_padding is not None:
            assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        self.mirror_padding = mirror_padding

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, 'images')
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims)
        self.file_path = images_dir
        self.instance_ratio = instance_ratio

        min_value, max_value, mean, std = calculate_stats(self.images)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')

        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != 'test':
            # load labeled images
            masks_dir = os.path.join(root_dir, 'masks')
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(masks_dir, expand_dims)
            # prepare for training with sparse object supervision (allow sparse objects only in training phase)
            if self.instance_ratio is not None and phase == 'train':
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(random_seed)
                self.masks = [sample_instances(m, self.instance_ratio, rs) for m in self.masks]
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_imgs = []
                for img in self.images:
                    padded_img = np.pad(img, pad_width=pad_width, mode='reflect')
                    padded_imgs.append(padded_img)

                self.images = padded_imgs

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != 'test':
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)
        expand_dims = dataset_config.get('expand_dims', True)
        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)
        return [cls(file_paths[0], phase, transformer_config, mirror_padding, expand_dims, instance_ratio, random_seed)]

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths
    
class NiiDataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, mirror_padding=(0, 32, 32), expand_dims=True,
                 instance_ratio=None, random_seed=0,patch_shape=(80,80,80),atlas_path=None,suffix_raw='_ana_strip_1mm_center_cropped.nii.gz',suffix_label='_seg_ana_1mm_center_cropped.nii.gz',dirpath='',suffix_aux='_seg_tissue_1mm_center_cropped.nii.gz'):
        # assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        # use mirror padding only during the 'test' phase
        if phase in ['train', 'val']:
            mirror_padding = None
        if mirror_padding is not None:
            assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        self.mirror_padding = mirror_padding

        self.phase = phase

        # load raw images
        # images_dir = os.path.join(root_dir, 'images')
        # assert os.path.isdir(images_dir)
        self.paths = self._load_files(root_dir,phase, suffix_raw,dirpath)
        self.images = self._load_images(self.paths,expand_dims=expand_dims)
        self.atlas = self._load_nii(atlas_path,True) if atlas_path else None
        self.file_path = root_dir
        self.instance_ratio = instance_ratio
        self.expand_dims=expand_dims
        min_value, max_value, mean, std = calculate_stats(self.images)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')

        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()
        
        self.raws = [np.array(self.images[0].shape)]
        self.patch_shape = patch_shape
        # print(self.raws,self.raws[0])
        # if phase != 'test':
            # load labeled images
            # masks_dir = os.path.join(root_dir, 'masks')
        # assert os.path.isdir(masks_dir)
        self.mask_paths = self._load_files(root_dir,phase, suffix_label,dirpath)
        self.masks = self._load_masks(self.mask_paths)

        self.tissues=None
        if suffix_aux is not None:
            self.tissue_paths = self._load_files(root_dir,phase, suffix_aux,dirpath)
            self.tissues = self._load_masks(self.tissue_paths)
        # load label images transformer
        self.masks_transform = transformer.label_transform()
    # else:
        # self.masks = None
        # self.masks_transform = None
        self.subjects = self._load_subjects(root_dir,phase)



    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img =self.images[idx]
        if self.phase != 'test':
            mask = self.masks[idx]
            # icls = np.random.randint(0,15)
            
            if self.tissues is not None:
                tissue = self.tissues[idx]
            if self.expand_dims:
                img = np.expand_dims(img, axis=0)
            if self.atlas is not None:
                if self.tissues is not None:
                    return self.raw_transform(img), self.masks_transform(mask),self.atlas,self.masks_transform(tissue)
                else:
                    return self.raw_transform(img), self.masks_transform(mask),self.atlas
            else:
                return self.raw_transform(img), self.masks_transform(mask)
        else:
            mask = self.masks[idx]
            if self.atlas is not None:
                return self.raw_transform(img), self.masks_transform(mask),self.subjects[idx],self.atlas
            else:
                return self.raw_transform(img), self.masks_transform(mask),self.subjects[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return None

    @classmethod
    def create_datasets(cls, dataset_config, phase):

        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        dirpath =phase_config.get('dirpath','')
        atlas_path = phase_config.get('atlas_path',None)
        patch_shape = phase_config['slice_builder']['patch_shape']
        suffix_raw = dataset_config.get('suffix_raw','_ana_strip_1mm_center_cropped.nii.gz')
        suffix_label = dataset_config.get('suffix_label','_seg_ana_1mm_center_cropped.nii.gz')
        suffix_aux = dataset_config.get('suffix_aux',None)

        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)
        expand_dims = dataset_config.get('expand_dims', True)
        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)
        return [cls(file_paths[0], phase, transformer_config, mirror_padding, expand_dims, instance_ratio, random_seed,patch_shape,atlas_path,suffix_raw,suffix_label,dirpath,suffix_aux)]

    @staticmethod
    def _load_nii(path,expand_dims):
        img = nib.load(path).get_fdata()
        return img

    @staticmethod
    def _load_images(paths,expand_dims=False):
        images = []
        
        for path in paths:
            img = nib.load(path).get_fdata()
            
            # if expand_dims:
            #     dims = img.ndim
            #     img = np.expand_dims(img, axis=0)
                # if dims == 3:
                #     img = np.transpose(img, (3, 0, 1, 2))
            images.append(img)
        return images

    @staticmethod
    def _load_masks(paths):
        images = []
        
        for path in paths:
            img = nib.load(path).get_fdata()
            images.append(img.astype(np.int64))
        return images

    @staticmethod
    def read_pkl(path,phase):
        with open(path,'r') as f:
            dct = json.load(f)
        return dct[phase]

    @staticmethod
    def _load_files(path,phase, suffix,rootdir=''):
        with open(path,'r') as f:
            dct = json.load(f)
        paths = [rootdir+f+suffix for f in dct[phase]]
        return paths

    @staticmethod
    def _load_subjects(path,phase):
        with open(path,'r') as f:
            dct = json.load(f)
        
        paths = [f.split('/')[-1] for f in dct[phase]]
        return paths