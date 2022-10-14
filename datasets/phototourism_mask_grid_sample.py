import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from math import sqrt, exp
import random

from . import global_val

class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1, use_cache=False, batch_size=1024, scale_anneal=-1, min_scale=0.25):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale

        if ('hagia_sophia_interior' in self.root_dir) or ('taj_mahal' in self.root_dir):
            self.img_downscale_appearance = 4
        else:
            self.img_downscale_appearance = 8

        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num) # at least 1
        self.use_cache = use_cache
        self.define_transforms()

        self.read_meta()
        self.white_back = False

        # no effect if scale_anneal<0, else the minimum scale decreases exponentially until converge to min_scale
        self.scale_anneal = scale_anneal
        self.min_scale = min_scale

        self.batch_size = batch_size

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                if filename in img_path_to_id:
                    id_ = img_path_to_id[filename]
                    self.image_paths[id_] = filename
                    self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.img_names_test = [self.files.loc[i, 'filename'] for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
                # with open(os.path.join(self.root_dir, f'cache/all_imgs{self.img_downscale}.pkl'), 'rb') as f:
                #     self.all_imgs = pickle.load(f)
                with open(os.path.join(self.root_dir, f'cache/all_imgs{8}.pkl'), 'rb') as f:
                    self.all_imgs = pickle.load(f)
                all_imgs_wh = np.load(os.path.join(self.root_dir,
                                                f'cache/all_imgs_wh{self.img_downscale}.npy'))
                self.all_imgs_wh = torch.from_numpy(all_imgs_wh)
            else:
                self.all_rays = []
                self.all_rgbs = []
                self.all_imgs = []
                self.all_imgs_wh = []
                for id_ in self.img_ids_train:
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img_rs = img.resize((img_w, img_h), Image.LANCZOS)
                    img_rs = self.transform(img_rs) # (3, h, w)

                    img_8 = img.resize((img_w//self.img_downscale_appearance, img_h//self.img_downscale_appearance), Image.LANCZOS)
                    img_8 = self.normalize(self.transform(img_8)) # (3, h, w)
                    self.all_imgs += [img_8]
                    self.all_imgs_wh += [torch.Tensor([img_w, img_h]).unsqueeze(0)]
                    img_rs = img_rs.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img_rs]
                    
                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 8)
                    
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
                self.all_imgs_wh = torch.cat(self.all_imgs_wh, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        if self.split == 'train':
            self.iterations = len(self.all_rays)//self.batch_size
            return self.iterations
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_test':
            return self.N_images_test
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            np.random.seed(global_val.current_epoch * self.iterations + idx)
            sample_ts = np.random.randint(0, len(self.all_imgs))
            img_w, img_h = self.all_imgs_wh[sample_ts]
            img = self.all_imgs[sample_ts]
            # grid
            w_samples, h_samples = torch.meshgrid([torch.linspace(0, 1-1/img_w, int(sqrt(self.batch_size))), \
                                                    torch.linspace(0 , 1-1/img_h, int(sqrt(self.batch_size)))])
            if self.scale_anneal > 0:
                min_scale_cur = min(max(self.min_scale, 1. * exp(-(global_val.current_epoch * self.iterations + idx)* self.scale_anneal)), 0.9)
            else:
                min_scale_cur = self.min_scale
            scale = torch.Tensor(1).uniform_(min_scale_cur, 1.)
            h_offset = torch.Tensor(1).uniform_(0, (1-scale.item())*(1-1/img_h))
            w_offset = torch.Tensor(1).uniform_(0, (1-scale.item())*(1-1/img_w))
            h_sb = h_samples * scale + h_offset
            w_sb = w_samples * scale + w_offset
            h = (h_sb * img_h).floor()
            w = (w_sb * img_w).floor()
            img_sample_points = (w + h * img_w).permute(1, 0).contiguous().view(-1).long()
            uv_sample = torch.cat((h_sb.permute(1, 0).contiguous().view(-1,1), w_sb.permute(1, 0).contiguous().view(-1,1)), -1)

            rgb_sample_points = (img_sample_points + (self.all_imgs_wh[:sample_ts, 0]*self.all_imgs_wh[:sample_ts, 1]).sum()).long()

            sample = {'rays': self.all_rays[rgb_sample_points, :8],
                      'ts': self.all_rays[rgb_sample_points, 8].long(),
                      'rgbs': self.all_rgbs[rgb_sample_points],
                      'whole_img': img,
                      'rgb_idx': img_sample_points,
                      'min_scale_cur': min_scale_cur,
                      'img_wh': self.all_imgs_wh[sample_ts],
                      'uv_sample': uv_sample}

        elif self.split in ['val', 'test_train', 'test_test']:
            sample = {}
            if self.split == 'val':
                id_ = self.val_id
            elif self.split == 'test_test':
                id_ = self.img_ids_test[idx]
            elif self.split == 'test_train':
                id_ = self.img_ids_train[idx]

            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img_s = img.resize((img_w, img_h), Image.LANCZOS)
            img_s = self.transform(img_s) # (3, h, w)

            img_s = img_s.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img_s

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])

            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])
            sample['rgb_idx'] = torch.LongTensor([i for i in range (0, (img_w*img_h))])
            
            w_samples, h_samples = torch.meshgrid([torch.linspace(0, 1-1/img_w, int(img_w)), \
                                                    torch.linspace(0, 1-1/img_h, int(img_h))])
            uv_sample = torch.cat((h_samples.permute(1, 0).contiguous().view(-1,1), w_samples.permute(1, 0).contiguous().view(-1,1)), -1)
            sample['uv_sample'] = uv_sample

            img_w, img_h = img.size
            img_8 = img.resize((img_w//self.img_downscale_appearance, img_h//self.img_downscale_appearance), Image.LANCZOS)
            img_8 = self.normalize(self.transform(img_8)) # (3, h, w)
            sample['whole_img'] = img_8

        else:
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

        return sample
