import os

from numpy.lib.utils import who
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

from math import sqrt

# models
from models.nerf import *
from models.rendering import *
from models.networks import E_attr, implicit_mask

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger

from datasets import global_val

import random

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.loss = loss_dict['hanerf'](hparams, coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embedding_uv = PosEmbedding(10-1, 10)

        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.enc_a = E_attr(3, hparams.N_a)
            self.models_to_train += [self.enc_a]
            self.embedding_a_list = [None] * hparams.N_vocab

        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_random=hparams.encode_random)

            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]

        if hparams.use_mask:
            self.implicit_mask = implicit_mask()
            self.models_to_train += [self.implicit_mask]
            self.embedding_view = torch.nn.Embedding(hparams.N_vocab, 128)
            self.models_to_train += [self.embedding_view]
        
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender):
        results = defaultdict(list)
        kwargs ={}
        if self.hparams.encode_a:
            if test_blender:
                kwargs['a_embedded_from_img'] = self.embedding_a_list[0] if self.embedding_a_list[0] != None else self.enc_a(whole_img)
            else:
                kwargs['a_embedded_from_img'] = self.enc_a(whole_img)

            if self.hparams.encode_random:
                idexlist = [k for k,v in enumerate(self.embedding_a_list) if v != None]
                if len(idexlist) == 0:
                    kwargs['a_embedded_random'] = kwargs['a_embedded_from_img']
                else:
                    kwargs['a_embedded_random'] = self.embedding_a_list[random.choice(idexlist)]

        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        if self.hparams.use_mask:
            if test_blender:
                results['out_mask'] = torch.zeros(results['rgb_fine'].shape[0], 1).to(results['rgb_fine'])
            else:
                uv_embedded = self.embedding_uv(uv_sample)
                results['out_mask'] = self.implicit_mask(torch.cat((self.embedding_view(ts), uv_embedded), dim=-1))

        if self.hparams.encode_a:
            results['a_embedded'] = kwargs['a_embedded_from_img']
            if self.hparams.encode_random:
                results['a_embedded_random'] = kwargs['a_embedded_random']
                rec_img_random = results['rgb_fine_random'].view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1
                results['a_embedded_random_rec'] = self.enc_a(rec_img_random)
                self.embedding_a_list[ts[0]] = kwargs['a_embedded_from_img'].clone().detach()

        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['batch_size'] = self.hparams.batch_size
            kwargs['scale_anneal'] = self.hparams.scale_anneal
            kwargs['min_scale'] = self.hparams.min_scale
        elif self.hparams.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
            kwargs['batch_size'] = self.hparams.batch_size
            kwargs['scale_anneal'] = self.hparams.scale_anneal
            kwargs['min_scale'] = self.hparams.min_scale
            if self.hparams.useNeuralRenderer:
                kwargs['NeuralRenderer_downsampleto'] = (self.hparams.NRDS, self.hparams.NRDS)
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # self.hparams.batch_size a time
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()
        rgbs = batch['rgbs'].squeeze()
        uv_sample = batch['uv_sample'].squeeze()
        if self.hparams.encode_a or self.hparams.use_mask:
            whole_img = batch['whole_img']
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None
        H = int(sqrt(rgbs.size(0)))
        W = int(sqrt(rgbs.size(0)))

        test_blender = False
        results = self(rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams, self.global_step)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/AnnealingWeight', AnnealingWeight)
        self.log('train/min_scale_cur', batch['min_scale_cur'])
        for k, v in loss_d.items():
            self.log(f'train/{k}', v)
        self.log('train/psnr', psnr_)

        if (self.global_step + 1) % 5000 == 0:
            img = results[f'rgb_{typ}'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].detach().view(H, W)) # (3, H, W)
            if self.hparams.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    stack = torch.stack([img_gt, img, depth, img_random, mask]) # (4, 3, H, W)
                    self.logger.experiment.add_images('train/GT_pred_depth_random_mask',
                                                      stack, self.global_step)
                else:
                    stack = torch.stack([img_gt, img, depth, mask]) # (3, 3, H, W)
                    self.logger.experiment.add_images('train/GT_pred_depth_mask',
                                                      stack, self.global_step)
            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = torch.stack([img_gt, img, depth, img_random]) # (4, 3, H, W)
                self.logger.experiment.add_images('train/GT_pred_depth_random',
                                                  stack, self.global_step)
            else:
                stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
                self.logger.experiment.add_images('train/GT_pred_depth',
                                                  stack, self.global_step)
        
        return loss

    def validation_step(self, batch, batch_nb):
        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()
        rgbs =  batch['rgbs'].squeeze()
        if self.hparams.dataset_name == 'phototourism':
            uv_sample = batch['uv_sample'].squeeze()
            WH = batch['img_wh']
            W, H = WH[0, 0].item(), WH[0, 1].item()
        else:
            W, H = self.hparams.img_wh
            uv_sample = None

        if self.hparams.encode_a or self.hparams.use_mask or self.hparams.deocclusion:
            if self.hparams.dataset_name == 'phototourism':
                whole_img = batch['whole_img']
            else:
                whole_img = rgbs.view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None

        test_blender = (self.hparams.dataset_name == 'blender')
        results = self(rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams, self.global_step)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[k] = v
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        if batch_nb == 0:
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            if self.hparams.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    stack = torch.stack([img_gt, img, depth, img_random, mask]) # (5, 3, H, W)
                    self.logger.experiment.add_images('val/GT_pred_depth_random_mask',
                                                      stack, self.global_step)
                else:
                    stack = torch.stack([img_gt, img, depth, mask]) # (4, 3, H, W)
                    self.logger.experiment.add_images('val/GT_pred_depth_mask',
                                                      stack, self.global_step)
            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = torch.stack([img_gt, img, depth, img_random]) # (4, 3, H, W)
                self.logger.experiment.add_images('val/GT_pred_depth_random',
                                                  stack, self.global_step)
            else:
                stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
                self.logger.experiment.add_images('val/GT_pred_depth',
                                                  stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        ssim_ = ssim(img[None,...], img_gt[None,...])
        log['val_psnr'] = psnr_
        log['val_ssim'] = ssim_

        return log

    def validation_epoch_end(self, outputs):
        if len(outputs) == 1:
            global_val.current_epoch = self.current_epoch
        else:
            global_val.current_epoch = self.current_epoch + 1
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim, prog_bar=True)

        if self.hparams.use_mask:
            self.log('val/c_l', torch.stack([x['c_l'] for x in outputs]).mean())
            self.log('val/f_l', torch.stack([x['f_l'] for x in outputs]).mean())
            self.log('val/r_ms', torch.stack([x['r_ms'] for x in outputs]).mean())
            self.log('val/r_md', torch.stack([x['r_md'] for x in outputs]).mean())

def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(hparams.save_dir,
                                              f'ckpts/{hparams.exp_name}', '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=-1)

    logger = TestTubeLogger(save_dir=os.path.join(hparams.save_dir,"logs"),
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus= hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)

if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)