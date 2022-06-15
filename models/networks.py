import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

# E_attr, implicit_mask
class E_attr(nn.Module):
  def __init__(self, input_dim_a, output_nc=8):
    super(E_attr, self).__init__()
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))  ## 1*1
    return

  def forward(self, x):
    x = self.model(x)
    output = x.view(x.size(0), -1)
    return output

class implicit_mask(nn.Module):
    def __init__(self, latent=128, W=256, in_channels_dir=42):
        super().__init__()
        self.mask_mapping = nn.Sequential(
                            nn.Linear(latent + in_channels_dir, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, 1), nn.Sigmoid())

    def forward(self, x):
        mask = self.mask_mapping(x)
        return mask