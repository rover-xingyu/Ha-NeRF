import torch
from torch.utils.data import Dataset
from .ray_utils import *

class PhototourismDataset(Dataset):
    def __init__(self,):
        self.white_back = False

    def __len__(self):
        return len(self.poses_test)

    def __getitem__(self, idx):
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
        sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

        return sample
