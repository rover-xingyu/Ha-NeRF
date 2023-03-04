# Ha-NeRF:laughing:: Hallucinated Neural Radiance Fields in the Wild 
**[Project Page](https://rover-xingyu.github.io/Ha-NeRF/) |
[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Hallucinated_Neural_Radiance_Fields_in_the_Wild_CVPR_2022_paper.pdf) |
[Latest arXiv](https://arxiv.org/pdf/2111.15246.pdf) |
[Supplementary](https://rover-xingyu.github.io/Ha-NeRF/files/Ha_NeRF_CVPR_2022_supp.pdf)**

[Xingyu Chen¹](https://rover-xingyu.github.io/), 
[Qi Zhang²](https://qzhang-cv.github.io/), 
[Xiaoyu Li²](https://xiaoyu258.github.io/), 
[Yue Chen¹](https://fanegg.github.io/), 
[Ying Feng²](https://github.com/rover-xingyu/Ha-NeRF/),
[Xuan Wang²](https://scholar.google.com/citations?user=h-3xd3EAAAAJ&hl=en/),
[Jue Wang²](https://juewang725.github.io/). 

[¹Xi'an Jiaotong University)](http://en.xjtu.edu.cn/),
[²Tencent AI Lab](https://ai.tencent.com/ailab/en/index/).


This repository is an official implementation of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/) (Hallucinated Neural Radiance Fields in the Wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). 

<!-- I try to reproduce (some of) the results on the lego dataset (Section D). Training on [Phototourism real images](https://github.com/ubc-vision/image-matching-benchmark) (as the main content of the paper) has also passed. Please read the following sections for the results.

The code is largely based on NeRF implementation (see master or dev branch), the main difference is the model structure and the rendering process, which can be found in the two files under `models/`. -->

# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2**
* We optimize all implementations for 600 000 iterations with a batch size of 1024 on 4 A100s, where Ha-NeRF and NeRF-W take 20 and 18 hours, respectively. Both methods require around 20 GB of memory for training and 5 GB for inference.

## Software

* Clone this repo by `git clone https://github.com/rover-xingyu/Ha-NeRF`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n HaNeRF python=3.6` to create a conda environment and activate it by `conda activate HaNeRF`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    
# :key: Training

## Data download

Download the scenes you want from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) 

Download the train/test split from [here](https://nerf-w.github.io/) and put under each scene's folder (the **same level** as the "dense" folder)

(Optional but **highly** recommended) Run `python3.6 prepare_phototourism.py --root_dir $ROOT_DIR --img_downscale {an integer, e.g. 2 means half the image sizes}` to prepare the training data and save to disk first, if you want to run multiple experiments or run on multiple gpus. This will **largely** reduce the data preparation step before training.

Run (example)

```
python3.6 prepare_phototourism.py --root_dir /path/to/the/datasets/brandenburg_gate/ --img_downscale 2
```

## Training model
Run (example)
```
python3.6 train_mask_grid_sample.py \
  --root_dir /path/to/the/datasets/brandenburg_gate/ --dataset_name phototourism \
  --save_dir save \
  --img_downscale 2 --use_cache \
  --N_importance 64 --N_samples 64 \
  --num_epochs 20 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name exp_HaNeRF_Brandenburg_Gate \
  --N_emb_xyz 15 --N_vocab 1500 \
  --use_mask --maskrs_max 5e-2 --maskrs_min 6e-3 --maskrs_k 1e-3 --maskrd 0 \
  --encode_a --N_a 48 --weightKL 1e-5 --encode_random --weightRecA 1e-3 --weightMS 1e-6 \
  --num_gpus 4
```

Add `--encode_a` for using appearance hallucination module, `--use_mask` for using  anti-occlusion module. `--N_vocab` should be set to an integer larger than the number of images (dependent on different scenes). For example, "brandenburg_gate" has in total 1363 images (under `dense/images/`), so any number larger than 1363 works (no need to set to exactly the same number). **Attention!** If you forget to set this number, or it is set smaller than the number of images, the program will yield `RuntimeError: CUDA error: device-side assert triggered` (which comes from `torch.nn.Embedding`).

See [opt.py](opt.py) for all configurations.

The checkpoints and logs will be saved to `{save_dir}/ckpts/{scene_name} ` and `{save_dir}/logs/{scene_name}`, respectively.

You can monitor the training process by `tensorboard --logdir save/logs/exp_HaNeRF_Brandenburg_Gate --port=8600` and go to `localhost:8600` in your browser.

# :mag_right: Evaluation

Use [eval.py](eval.py) to inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered
images.

Run (example)
```
python3.6 eval.py \
  --root_dir /path/to/the/datasets/brandenburg_gate/ \
  --save_dir save \
  --dataset_name phototourism --scene_name HaNeRF_Trevi_Fountain \
  --split test_test --img_downscale 2 \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500 --encode_a \
  --ckpt_path save/ckpts/HaNeRF_Brandenburg_Gate/epoch\=19.ckpt \
  --chunk 16384 --img_wh 320 240
```

Then you can use [eval_metric.py](eval_metric.py) to get the quantitative report of different metrics based on the rendered images from [eval.py](eval.py). It will create a file `result.txt` in the folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the metrics.

Run (example)
```
python3.6 eval_metric.py \
  --root_dir /path/to/the/datasets/brandenburg_gate/ \
  --save_dir save \
  --dataset_name phototourism --scene_name HaNeRF_Trevi_Fountain \
  --split test_test --img_downscale 2 \
  --img_wh 320 240
```

# :laughing: Hallucination

Use [hallucinate.py](hallucinate.py) to play with Ha-NeRF by hallucinating appearance from different scenes `{example_image}` in different views! It will create folder `{save_dir}/hallucination/{scene_name}` and render the hallucinations, finally create a gif out of them.

Run (example)
```
python3.6 hallucinate.py \
    --save_dir save \
    --ckpt_path save/ckpts/HaNeRF_Trevi_Fountain/epoch\=19.ckpt \
    --chunk 16384 \
    --example_image artworks \
    --scene_name artworks_2_fountain
```

# Cite
If you find our work useful, please consider citing:
```bibtex
@inproceedings{chen2022hallucinated,
  title={Hallucinated neural radiance fields in the wild},
  author={Chen, Xingyu and Zhang, Qi and Li, Xiaoyu and Chen, Yue and Feng, Ying and Wang, Xuan and Wang, Jue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12943--12952},
  year={2022}
}
```

# Acknowledge
Our code is based on the awesome pytorch implementation of NeRF in the Wild ([NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw/)). We appreciate all the contributors.
