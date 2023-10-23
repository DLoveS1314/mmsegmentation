# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn as nn

from mmseg.utils import SampleList
from .icounet import IcoConvModule,icopad_conv2d
from mmseg.registry import MODELS
from mmseg.models  import FCNHead 
from .icoEncoderDecoder  import idg2erp_decoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import warnings
from abc import ABCMeta, abstractmethod
from mmengine.model import BaseModule
from typing import List, Tuple
from mmseg.models.builder import build_loss

@MODELS.register_module()
class IcoFCNHead(BaseModule):
    # 对 BaseDecodeHead 的改写 里面有很多回归不需要的内容 因此不能集成
    def __init__(self,
                 in_channels,
                 channels=None,
                 num_convs=1,
                 kernel_size=3,
                 concat_input=False,
                 dilation=1, 
                 num_classes=None,
                 separate =False,#均值和方差是否分开卷积
                 inter_times =1 ,
                 #插值次数  只在separate =True 时有效 1代表均值和方差分开卷积之后 进行相同模块的插值 2代表均值和方差分开卷积之后 进行不同模块的插值
                 erp2igd_dict =dict(k_size=9, delta=0.5,rootpath='./index_table',dggrid_level=7,p=0,dggird_type='FULLER4D'),
                 conv_cfg  = dict(type = 'Conv2d'),
                 norm_cfg  =dict(type = 'BN2d'),
                 act_cfg=dict(type='Swish'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.channels = channels
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        
        self.conv_cfg = conv_cfg
        self.norm_cfg =norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.input_transform = input_transform
        
        
        self.out_channels = 2 # 2个通道，一个是均值一个是方差
        super().__init__( )
        
        # 借鉴 BaseDecodeHead 的方法
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        
        
        
        
        self.seprate = separate
        self.inter_times = inter_times
        if num_convs == 0 :#num_convs = 0 时，不进行卷积，直接回归 也就不存在均值和方差分开卷积
            assert channels == None and concat_input == False and separate == False
        
        if separate:
            self.conv_mean = self.create_convs(
                out_channels=self.out_channels//2,
                num_convs=num_convs,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.conv_std = self.create_convs(
                out_channels=self.out_channels//2,
                num_convs=num_convs,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            if inter_times == 1 :
                erp2igd_dict['in_channels'] =self.out_channels//2
                self.idg2erp_mean =  idg2erp_decoder(**erp2igd_dict)
                self.idg2erp_std =  idg2erp_decoder(**erp2igd_dict)

        else:
            self.convs = self.create_convs(
                out_channels=self.out_channels,
                num_convs=num_convs,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        # 如果不插值两次 则通道数均为2 如果不分开卷积 通道数也为2
        erp2igd_dict['in_channels'] =self.out_channels
        self.idg2erp =  idg2erp_decoder(**erp2igd_dict)
    def create_convs(self, out_channels, num_convs, kernel_size,
                     dilation):
        """Create conv layers used in FCNHead.创建过程中包含了最后一层的回归"""
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            IcoConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                IcoConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        # if self.concat_input: #不使用这个模块
        #     self.conv_cat = IcoConvModule(
        #         self.in_channels + self.channels,
        #         self.channels,
        #         kernel_size=kernel_size,
        #         padding=kernel_size // 2,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        #     convs.append(self.conv_cat)
        reg_seg = icopad_conv2d(
            self.channels, out_channels, kernel_size=3, padding=0) #最后一层对结果进行回归
        convs.append(reg_seg)
        convs = nn.Sequential(*convs)
        
        return convs
    
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        # print('IcoFCNHead x.shape',x.shape)
        if self.seprate:#均值和方差是否分开卷积
            feats_mean = self.conv_mean(x) #回归也各自进行
            feats_std = self.conv_std(x)
            if self.inter_times >1: # 方差和均值分开进行插值 先插值 后合并 B，2，H，W
                feats_mean = self.idg2erp_mean(feats_mean) #插值到经纬度格网上 B，1，H，W
                feats_std = self.idg2erp_std(feats_std) # 插值到经纬度格网上 B，1，H，W
                feats = torch.cat([feats_mean,feats_std],dim=1)
            else:# 方差和均值统一进行插值 先合并 后插值 B，2，H，W  由于使用了group卷积 好像给分开卷积没有区别？？
                feats = torch.cat([feats_mean,feats_std],dim=1) # B，2，H，W  
                feats = self.idg2erp(feats) #插值到经纬度格网上 B，2，H，W
            # feats = torch.cat([feats_mean,feats_std],dim=1)#分开后拼接 进行插值
        else:
            feats = self.convs(x)
        # if self.concat_input: #不使用这个模块
        #     feats = self.conv_cat(torch.cat([x, feats], dim=1))

        return feats

    def forward(self, inputs):
        """Forward function.
        inputs (list[Tensor]): List of multi-level img features. 本文一般只使用一个输入 
        inputs[0] 为unet最后一层的输入 B*10 ,C,H,W
        """
        output = self._forward_feature(inputs) #是否再卷积一次，因为H W已经回复到原始大小了 所以这次卷积的计算量应该会很大 是否添加慎重考虑
        # output = self.reg_seg(output) # 通过卷积对Unet的结果进行回归 不添加bn和relu 放在了create_convs中
        # output = self.idg2erp(output) # 插值到经纬度格网上  放在了 _forward_feature中
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # seg_logits B,C,H,W 第一个通道是均值 第二个通道是方差
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # seg_logits = resize(
        #     input=seg_logits,
        #     size=seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # if self.sampler is not None:
        #     seg_weight = self.sampler.sample(seg_logits, seg_label)
        # else:
        #     seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:#本文一共有两个loss 一个是mse类别的loss 一个是crps组成的loss
            if 'crps' in  loss_decode.loss_name :
                loss[loss_decode.loss_name] = loss_decode(
                    pred_mean=seg_logits[:,0], pred_stddev=seg_logits[:,1],target=seg_label)
            else:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits[:,0], seg_label)
        # loss['acc_seg'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        # seg_logits = resize(
        #     input=seg_logits,
        #     size=batch_img_metas[0]['img_shape'],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return seg_logits
