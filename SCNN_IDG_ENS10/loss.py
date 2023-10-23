# TODO:  WCRPS metric (Weighted CRPS with EFI)

import numpy as np
from torch import nn
import torch.nn.functional as F 
import torch
from typing import Union 
from mmseg.registry import MODELS
# 导入pytorch中的l1损失
# 
_normal_dist = torch.distributions.Normal(0., 1.)
_frac_sqrt_pi = 1 / np.sqrt(np.pi)


#  losss设计的不是很好 想用pytorch的要重新继承一下 因为forward的参数不一样 decode_head的loss_by_feat 里 loss的forward需要接受其他参数 而
@MODELS.register_module()
class L1Loss(torch.nn.L1Loss):
    def forward(self, input , target ,**kwargs)  :
        return F.l1_loss(input, target, reduction=self.reduction)
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return 'loss_l1'


@MODELS.register_module()
class MSELoss(torch.nn.MSELoss):
    def forward(self, input , target ,**kwargs)  :
        return F.mse_loss(input, target, reduction=self.reduction)
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return 'loss_mse'

@MODELS.register_module()
class SmoothL1Loss(torch.nn.SmoothL1Loss):
    def __init__(self,
                 lam_w=1):
        """_summary_

        Args:
            lam_w (int, optional) 添加的权重参数 我认为应该两个loss（CRPS 和 l1loss）一起跑结果可能会更好
        """        
        super(SmoothL1Loss, self).__init__()
        self.lam_w = lam_w
    def forward(self, input , target  ,**kwargs):
        return self.lam_w * F.smooth_l1_loss(input, target, reduction=self.reduction)
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return 'loss_SmoothL1'

@MODELS.register_module()
class MSE_VAR(nn.Module):
    # 公式 8  What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision
    def __init__(self, var_weight):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, var = results['mean'], results['var']
        var = self.var_weight * var

        loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
        loss2 = var
        loss = .5 * (loss1 + loss2)
        return loss.mean()
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return 'loss_mesvar'

@MODELS.register_module()
class CrpsGaussianLoss(nn.Module):
    """
      This is a CRPS loss function assuming a gaussian distribution. We use
      the following link:
      https://github.com/tobifinn/ensemble_transformer/blob/9b31f193048a31efd6aacb759e8a8b4a28734e6c/ens_transformer/measures.py

      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15,
                 lam_w=1):
        super(CrpsGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps
        self.lam_w = lam_w

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor,
 
                
                 **kwargs):

        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
 
        if self.mode == 'mean':
            crps = torch.mean(crps)
            
        return self.lam_w*crps
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return 'loss_crps'

@MODELS.register_module()
class EECRPSGaussianLoss(nn.Module):
    """
      This is a EECRPS loss function assuming a gaussian distribution with EFI indeces.
      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15):
        super(EECRPSGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor,
                efi_tensor: torch.Tensor):

        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = torch.abs(efi_tensor) * pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)

        if self.mode == 'mean':
            return torch.mean(crps)
        return crps
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return 'loss_eecrps'

# @MODELS.register_module()  
# class mixLoss(nn.Module):
#     # 混合 两个loss,输入的参数用字典形式包含loss_dic1,loss_dic2,losw1 loss1的权重 loss2的权重为1-lossw1
#     def __init__(self,loss_dic1,loss_dic2,lossw1):
#         super(mixLoss, self).__init__()
#         self.loss1= MODELS.build(loss_dic1)
#         self.loss2= MODELS.build(loss_dic2)
#         self.lossw1 = lossw1
#     def forward(self,results,label):
        
        
    