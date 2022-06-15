import torch
from torch import nn
import math

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss

class CosineAnnealingWeight():
    def __init__(self, max, min, Tmax):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.Tmax = Tmax

    def getWeight(self, Tcur):
        return self.min + (self.max - self.min) * (1 + math.cos(math.pi * Tcur / self.Tmax)) / 2

class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))

class HaNeRFLoss(nn.Module):
    def __init__(self, hparams, coef=1, lambda_u=0.01):
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        # self.Annealing = CosineAnnealingWeight(max = hparams.maskrs_max, min = hparams.maskrs_min, Tmax = hparams.num_epochs-1)
        self.Annealing = ExponentialAnnealingWeight(max = hparams.maskrs_max, min = hparams.maskrs_min, k = hparams.maskrs_k)
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets, hparams, global_step):
        ret = {}

        if 'a_embedded' in inputs:
            ret['kl_a'] = self._l2_regularize(inputs['a_embedded']) * hparams.weightKL
            if 'a_embedded_random_rec' in inputs:
                ret['rec_a_random'] = torch.mean(torch.abs(inputs['a_embedded_random'].detach() - inputs['a_embedded_random_rec'])) * hparams.weightRecA
                ret['mode_seeking'] = hparams.weightMS * 1 / \
                  ((torch.mean(torch.abs(inputs['rgb_fine'].detach() - inputs['rgb_fine_random'])) / \
                  torch.mean(torch.abs(inputs['a_embedded'].detach() - inputs['a_embedded_random'].detach()))) + 1 * 1e-5)

        if 'out_mask' in inputs:
            mask = inputs['out_mask']
            ret['c_l'] = 0.5 * ((1 - mask.detach()) * (inputs['rgb_coarse'] - targets)**2).mean()
        else:
            ret['c_l'] = 0.5 * ((inputs['rgb_coarse'] - targets)**2).mean()

        if 'rgb_fine' in inputs:
            if 'out_mask' in inputs:
                ret['r_ms'], ret['r_md'] = self.mask_regularize(inputs['out_mask'],  self.Annealing.getWeight(global_step), hparams.maskrd)
                ret['f_l'] = 0.5 * ((1 - mask) * (inputs['rgb_fine'] - targets)**2).mean()
            else:
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret, self.Annealing.getWeight(global_step)

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02

        # # l2 regularize
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_size, loss_focus_digit
      
    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

loss_dict = {'color': ColorLoss,
             'hanerf': HaNeRFLoss}