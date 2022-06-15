import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

root_dir = "/apdcephfs/private_faneggchen/PNW/datasets/nerf_synthetic/lego/"
file = "test" # train or val or test
perturbation_type = "color_occ" # occ or color

def perturbation_visualization():
    with open(os.path.join(root_dir, f"transforms_{file}.json"), 'r') as f:
            meta = json.load(f)
            
    for t, frame in enumerate(meta['frames']):
        image_path = os.path.join(root_dir, f"{frame['file_path']}.png")
        img = Image.open(image_path)
        img = add_perturbation(img, perturbation_type, t)
        ### save the perturbation
        new_dir = ('/').join(root_dir.split('/')[0:-2] + [root_dir.split('/')[-2] + '_'+ perturbation_type])
        new_image_path = os.path.join(new_dir, f"{frame['file_path']}.png")
        save_dir = ('/').join(new_image_path.split('/')[0:-1])
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        img.save(new_image_path, format="png")

def define_transforms(self):
    self.transform = T.ToTensor()

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
        

if __name__ == '__main__':
  perturbation_visualization()