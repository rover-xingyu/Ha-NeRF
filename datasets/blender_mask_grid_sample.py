import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
import random

from math import sqrt, exp

from .ray_utils import *

from . import global_val

def add_perturbation(img, perturbation, seed):
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)

    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))

    return img


class BlenderDataset(Dataset):
    def __init__(self, root_dir, batch_size=1024, split='train', img_wh=(800, 800),
                 perturbation=[], NeuralRenderer_downsampleto=None, scale_anneal=-1, min_scale=0.25):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.NeuralRenderer_downsampleto = NeuralRenderer_downsampleto
        self.define_transforms()

        assert set(perturbation).issubset({"color", "occ"}), \
            'Only "color" and "occ" perturbations are supported!'
        self.perturbation = perturbation
        if self.split == 'train':
            print(f'add {self.perturbation} perturbation!')
        self.read_meta()
        self.white_back = True
        self.w_samples, self.h_samples = torch.meshgrid([torch.linspace(0, 1-1/img_wh[0], int(sqrt(batch_size))), \
          torch.linspace(0 , 1-1/img_wh[0], int(sqrt(batch_size)))])
        self.scale_anneal = scale_anneal
        # no effect if scale_anneal<0, else the minimum scale decreases exponentially until converge to min_scale
        self.min_scale = min_scale
        # self.max_scale = 1.

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split.split('_')[-1]}.json"), 'r') as f:
            self.meta = json.load(f)
        if self.NeuralRenderer_downsampleto == None:
            w, h = self.img_wh
        else:
            w, h = self.NeuralRenderer_downsampleto

        # original focal length
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x'])
                                                                     # when W=800

        self.focal *= w/800  # modify focal length to match size self.img_wh
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w/2
        self.K[1, 2] = h/2

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.K)  # (h, w, 3)

        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_imgs = []
            for t, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)
                if t != 0: # perturb everything except the first image.
                           # cf. Section D in the supplementary material
                    img = add_perturbation(img, self.perturbation, t)

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.toTensor(img) # (4, h, w)
                img = img[:3, :, :]*img[-1:, :, :] + (1-img[-1:, :, :]) # blend A to RGB (3, h, w)
                self.all_imgs += [self.normalize(img).unsqueeze(0)]
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_imgs = torch.cat(self.all_imgs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.resize = T.Resize(self.img_wh, Image.BICUBIC)
        self.toTensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        if self.split == 'train':
            self.iterations = len(self.all_rays)//self.batch_size
            return self.iterations
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample_ts = np.random.randint(0, len(self.meta['frames']))
            if self.scale_anneal > 0:
                min_scale_cur = min(max(self.min_scale, 1. * exp(-(global_val.current_epoch * self.iterations + idx)* self.scale_anneal)), 0.9)
            else:
                min_scale_cur = self.min_scale
            scale = torch.Tensor(1).uniform_(min_scale_cur, 1.)
            h_offset = torch.Tensor(1).uniform_(0. ,(1. - scale.item())*(1. - 1./self.img_wh[1]))
            w_offset = torch.Tensor(1).uniform_(0. ,(1. - scale.item())*(1. - 1./self.img_wh[0]))
            h_sb = self.h_samples * scale + h_offset
            w_sb = self.w_samples * scale + w_offset
            h = (h_sb * self.img_wh[1]).round()
            w = (w_sb * self.img_wh[0]).round()
            img_sample_points = (w + h * self.img_wh[0]).permute(1, 0).contiguous().view(-1).long()
            rgb_sample_points = (img_sample_points.view(-1) + (self.img_wh[0]*self.img_wh[1]) * sample_ts).long()
            uv_sample = torch.cat((h_sb.permute(1, 0).contiguous().view(-1,1), w_sb.permute(1, 0).contiguous().view(-1,1)), -1)

            sample = {'rays': self.all_rays[rgb_sample_points, :8],
                      'ts': self.all_rays[rgb_sample_points, 8].long(),
                      'rgbs': self.all_rgbs[rgb_sample_points],
                      'whole_img': self.all_imgs[sample_ts],
                      'rgb_idx': img_sample_points,
                      'min_scale_cur': min_scale_cur,
                      'uv_sample': uv_sample}
        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            t = 0 # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.split == 'test_train' and idx != 0:
                t = idx
                img = add_perturbation(img, self.perturbation, idx)

            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.toTensor(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img[:3, :, :]*img[-1:, :, :] + (1-img[-1:, :, :]) # blend A to RGB (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask,
                      'rgb_idx': torch.LongTensor([i for i in range (0, (self.img_wh[0]*self.img_wh[1]))])}

            if self.split == 'test_train' and self.perturbation:
                  # append the original (unperturbed) image
                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.toTensor(img) # (4, H, W)
                valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
                img = img[:3, :, :]*img[-1:, :, :] + (1-img[-1:, :, :]) # blend A to RGB (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB

                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask

        return sample
