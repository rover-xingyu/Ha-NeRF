
from .blender_mask_grid_sample import BlenderDataset
from .phototourism_mask_grid_sample import PhototourismDataset

dataset_dict = {'blender': BlenderDataset,
                'phototourism': PhototourismDataset}