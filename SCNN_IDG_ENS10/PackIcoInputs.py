# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

# 把读取的数据处理成 results 和 data_samples 两部分 这样能在后续模型forward的时候保持相同的输入
@TRANSFORMS.register_module()
class PackIcoInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.
 

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """
    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            # 在dataset中已经处理好了 不做判断了
            # if len(img.shape) < 3:
            #     img = np.expand_dims(img, -1)
            # if not img.flags.c_contiguous:
            #     img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            # else:
            #     img = img.transpose(2, 0, 1)
            img = to_tensor(img).contiguous()
            # print('PackIcoInputs img.shape',img.shape)
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:#处理标签值
            if len(results['gt_seg_map'].shape) != 3:
                # data = to_tensor(results['gt_seg_map'][None,
                #                                        ...].astype(np.int64))
            # else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                # data = to_tensor(results['gt_seg_map'].astype(np.int64))
            data = to_tensor(results['gt_seg_map'])
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
            

        # if 'gt_edge_map' in results:
        #     # gt_edge_data = dict(
        #     #     data=to_tensor(results['gt_edge_map'][None,
        #     #                                           ...].astype(np.int64)))
        #     data_sample.set_data(dict(gt_edge_map=PixelData(**results['gt_edge_map'])))

        img_meta = {}
        if 'time' in results:
            img_meta['time'] = results['time']
        # img_meta['img_path'] = results['img_path']
        img_meta['ori_shape'] =  img.shape #ico_encoder_decoder inference 中 有一步判断用到了 所以要添加上
        data_sample.set_metainfo(img_meta)
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
