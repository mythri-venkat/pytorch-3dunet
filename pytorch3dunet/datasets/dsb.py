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
                 instance_ratio=None, random_seed=0,patch_shape=(80,80,80),atlas_path=None):
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
        self.paths = self._load_files(root_dir,phase, '_ana_strip_1mm_center_cropped.nii.gz')
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
        self.mask_paths = self._load_files(root_dir,phase, '_seg_ana_1mm_center_cropped.nii.gz')
        self.masks = self._load_masks(self.mask_paths)
        self.boxes=[self.get_cropped_structure(mask) for mask in self.masks]
        # print([(box[1]-box[0],box[3]-box[2],box[5]-box[4]) for box in self.boxes[0]])
        assert len(self.paths) == len(self.masks)
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
            boxes=self.boxes[idx]
            
            if self.expand_dims:
                img = np.expand_dims(img, axis=0)
            return self.raw_transform(img), self.masks_transform(mask),boxes,self.atlas
        else:
            mask = self.masks[idx]
            boxes=self.boxes[idx]
            return self.raw_transform(img), self.masks_transform(mask),boxes, self.subjects[idx],self.atlas

    def bbox2_3D(self,img,icls=0):

        r = np.any(img == icls, axis=(1, 2))
        c = np.any(img == icls, axis=(0, 2))
        z = np.any(img == icls, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        

        return [rmin,rmax,cmin,cmax,zmin,zmax]

    def getpwr(self,n):
        pos=math.ceil(math.log(n,2))
        pwr = math.pow(2,pos)
        if pwr>self.patch_shape[0]:
            return n
        if pwr <= 4:
            pwr =8
    
        if n<48 and (48-n)<(pwr-n):
            pwr = 48
        if n<40 and (40-n)<(pwr-n):
            pwr = 40
        if n<24 and (40-n)<(pwr-n):
            pwr = 24
        if n<20 and (20-n)<(pwr-n):
            pwr = 20
        
        return pwr
        

    def get_cropped_structure(self,lbl,ncls=15):
        boxes=[]
        for icls in range(ncls):
            box = self.bbox2_3D(lbl,icls)
            if icls == 0:
                box[0],box[1],box[2],box[3],box[4],box[5] = 0,self.patch_shape[0],0,self.patch_shape[1],0,self.patch_shape[2]
            center = [int((box[1] + box[0]) / 2), int((box[3] + box[2]) / 2), int((box[5] + box[4]) / 2)]
                
            b1=self.getpwr(box[1]-box[0])
            b2=self.getpwr(box[3]-box[2])
            b3=self.getpwr(box[5]-box[4])
            box[0]=center[0]-int(math.floor(b1/2))
            box[1]=center[0]+int(math.floor(b1/2))
            if box[0]<0:
                box[0]=0
                box[1]+=1
            if box[1]>self.patch_shape[0]:
                box[1]=self.patch_shape[0]
                box[0]-=1
            box[2]=center[1]-int(math.floor(b2/2))
            box[3]=center[1]+int(math.floor(b2/2))
            if box[2]<0:
                box[2]=0
                box[3]+=1
            if box[3]>self.patch_shape[0]:
                box[3]=self.patch_shape[0]
                box[2]-=1
            box[4]=center[2]-int(math.floor(b3/2))
            box[5]=center[2]+int(math.floor(b3/2))
            if box[4]<0:
                box[4]=0
                box[5]+=1
            if box[5]>self.patch_shape[0]:
                box[5]=self.patch_shape[0]
                box[4]-=1
            
            boxes.append(box)
        # img_cropped = img[int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]
        # lbl_cropped = lbl[int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]
        
        # lbl_cropped = np.expand_dims(lbl_cropped, axis=0)
        return boxes

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
        atlas_path = phase_config.get('atlas_path',None)
        patch_shape = phase_config['slice_builder']['patch_shape']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)
        expand_dims = dataset_config.get('expand_dims', True)
        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)
        return [cls(file_paths[0], phase, transformer_config, mirror_padding, expand_dims, instance_ratio, random_seed,patch_shape,atlas_path)]

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
    def _load_files(path,phase, suffix):
        with open(path,'r') as f:
            dct = json.load(f)
        paths = [f+suffix for f in dct[phase]]
        return paths

    @staticmethod
    def _load_subjects(path,phase):
        with open(path,'r') as f:
            dct = json.load(f)
        
        paths = [f.split('/')[-1] for f in dct[phase]]
        return paths