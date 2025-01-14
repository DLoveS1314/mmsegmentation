# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence
from typing import  Union

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
# from mmseg.utils import stack_batch
import numpy as np
import torch.nn.functional as F

# 修改了stack_batch 使其不再padding
def ico_stack_batch(inputs: List[torch.Tensor],
                data_samples =  None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'#只能是chw三维数据
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid #不明白为什么要padding  暂且放弃这个功能
    # assert (size is not None) ^ (size_divisor is not None), \
    #     'only one of size and size_divisor should be valid'

    # padded_inputs = [] #不明白为什么要padding  暂且放弃这个功能
    # padded_samples = []
    # inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    # max_size = np.stack(inputs_sizes).max(0)
    # if size_divisor is not None and size_divisor > 1:
    #     # the last two dims are H,W, both subject to divisibility requirement
    #     max_size = (max_size +
    #                 (size_divisor - 1)) // size_divisor * size_divisor
    # 相当于使用dataloader里的collate_fn
    return torch.stack(inputs, dim=0), data_samples

    for i in range(len(inputs)):
        tensor = inputs[i]
        # 不明白为什么要padding  暂且放弃这个功能
        # if size is not None:
        #     width = max(size[-1] - tensor.shape[-1], 0)
        #     height = max(size[-2] - tensor.shape[-2], 0)
        #     # (padding_left, padding_right, padding_top, padding_bottom)
        #     padding_size = (0, width, 0, height)
        # elif size_divisor is not None:
        #     width = max(max_size[-1] - tensor.shape[-1], 0)
        #     height = max(max_size[-2] - tensor.shape[-2], 0)
        #     padding_size = (0, width, 0, height)
        # else:
        #     padding_size = [0, 0, 0, 0]

        # pad img
        # pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(inputs[i])
 
        
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            # gt_sem_seg = data_sample.gt_sem_seg.data
            # del data_sample.gt_sem_seg.data
            # data_sample.gt_sem_seg.data = F.pad(
            #     gt_sem_seg, padding_size, value=seg_pad_val)
            # if 'gt_edge_map' in data_sample:
            #     gt_edge_map = data_sample.gt_edge_map.data
            #     del data_sample.gt_edge_map.data
            #     data_sample.gt_edge_map.data = F.pad(
            #         gt_edge_map, padding_size, value=seg_pad_val)
            # data_sample.set_metainfo({
            #     'img_shape': tensor.shape[-2:],
            #     'pad_shape': data_sample.gt_sem_seg.shape,
            #     'padding_size': padding_size
            # })
            padded_samples.append(data_sample)
        # else:
        #     padded_samples.append(
        #         dict(
        #             img_padding_size=padding_size,
        #             pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples

@MODELS.register_module()
class IcoDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.
    """

    def __init__(
        self,
        mean   = None,
        std    = None,
        std_mean =None,
        std_std = None,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        # self.size = size
        # self.size_divisor = size_divisor
        # self.pad_val = pad_val
        # self.seg_pad_val = seg_pad_val
        self.eps = 1e-8
        if mean is not None:
            assert std is not None and std_mean is not None and std_std is not None , 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            # mean = torch.as_tensor(np.load(mean))
            # std = torch.as_tensor(np.load(std))+self.eps
            # std_mean = torch.as_tensor(np.load(std_mean))
            # std_std = torch.as_tensor(np.load(std_std))+self.eps
            self.register_buffer('mean',
                                 torch.as_tensor(np.load(mean)), False)
            self.register_buffer('std',
                                 torch.as_tensor(np.load(std))+self.eps, False)
            self.register_buffer('std_mean',
                                 torch.as_tensor(np.load(std_mean)), False) 
            self.register_buffer('std_std',
                                    torch.as_tensor(np.load(std_std))+self.eps, False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore 这一步就能跟随model的device一样 很奇怪 先不追究了
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        inputs = [_input.float() for _input in inputs]
        # assert self._enable_normalize  ,'To enable the normalization in '    
        if self._enable_normalize:
            inputstmp = []
            # _input 维度为 C,H,W 对于t2m C为22 其中 偶数维使用 mean 和std初始化 奇数维使用std_mean 和std_std初始化
            for _input in inputs:
                _input[::2] = (_input[::2] - self.mean) / (self.std + self.eps) 
                _input[1::2] = (_input[1::2] - self.std_mean) / (self.std_std+ self.eps)
                inputstmp.append(_input)
            inputs = torch.stack(inputstmp, dim=0)
        return dict(inputs=inputs, data_samples=data_samples)
        # if training:
        #     assert data_samples is not None, ('During training, ',
        #                                       '`data_samples` must be define.')
        #     inputs, data_samples = ico_stack_batch(
        #         inputs=inputs,
        #         data_samples=data_samples,
        #         size=self.size,
        #         size_divisor=self.size_divisor,
        #         pad_val=self.pad_val,
        #         seg_pad_val=self.seg_pad_val)

        #     if self.batch_augments is not None:
        #         inputs, data_samples = self.batch_augments(
        #             inputs, data_samples)
        # else:
        #     img_size = inputs[0].shape[1:]
        #     assert all(input_.shape[1:] == img_size for input_ in inputs),  \
        #         'The image size in a batch should be the same.'
        #     # pad images when testing
        #     if self.test_cfg:
        #         inputs, padded_samples = ico_stack_batch(
        #             inputs=inputs,
        #             size=self.test_cfg.get('size', None),
        #             size_divisor=self.test_cfg.get('size_divisor', None),
        #             pad_val=self.pad_val,
        #             seg_pad_val=self.seg_pad_val)
        #         for data_sample, pad_info in zip(data_samples, padded_samples):
        #             data_sample.set_metainfo({**pad_info})
        #     else:
                # inputs = torch.stack(inputs, dim=0)
        # inputs = torch.stack(inputs, dim=0)
        # return dict(inputs=inputs, data_samples=data_samples)
