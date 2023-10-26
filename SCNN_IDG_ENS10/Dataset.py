
import torch
from torch.utils.data import Dataset 
import numpy as np
import xarray as xr
from mmseg.registry import DATASETS
from mmseg.registry import TRANSFORMS
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from .PackIcoInputs import PackIcoInputs
# 改写自ENS10给的demo 加载太慢了 转为npy模式进行加载 数据放在了data文件夹下
from mmengine.dataset import BaseDataset, Compose

import os
@DATASETS.register_module()
class ENS10Dataset(BaseDataset):
    """
        主要修改的地方 是BaseDataset 的 full_init 以及 getitem里面对应的函数
        ann_file (str, optional): Annotation file path. Defaults to ''. 存储的是mat文件的路径 保存为json
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None. 元数据 可以在文件中保存好
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''. 文件的根目录 与ann_file、data_prefix的路径拼接  
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
            
        _join_prefix 中会把data_root和data_prefix拼接起来 以及data_root和 ann_file 拼接起来
        
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """

    def __init__(self,
                 target_var, 
                 #输入目标输出的变量名 确定 target_mean 和 target_std 为了与ENS10代码中最后的计算一致            
                 # mu = output[:, 0] * scale_std + scale_mean
                 #sigma = torch.exp(output[:, 1]) * scale_std
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline= [dict(type= 'PackIcoInputs')],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)
        # # 创建一个PackIcoInputs的实例 在基类中自动创建
        assert target_var in ['t2m','t850','z500'] , 'target_var must be in t2m,t850,z500 but got {}'.format(target_var)
        if target_var == "t850":#确定target_mean在输入中的位置 见exp.ipynb 中的测试
            # self.variables = ["Z", "T", "Q", "W", "D", "U", "V"]
            #self.variables = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']
            self.target_mean_index = 2
            self.target_std_index = 3
        elif target_var == "t2m":
            self.target_mean_index = 16
            self.target_std_index = 17
        else :#'z500'
            self.target_mean_index = 0
            self.target_std_index = 1
        # self.pipeline = PackIcoInputs()
    # def __len__(self): 基类自动实现
    #     return self.length
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    def loadImageFromFile(self,data_info):
        # 从 data_info里读取数据和标签
        filename = data_info['img_path']
        # print('loadImageFromFile',filename)
        data_info['img'] = np.load(filename)
        # print('loadImageFromFile gt_seg_map',data_info['gt_seg_map'])
        
        data_info['gt_seg_map'] = np.load(data_info['gt_seg_map'] )
        data_info['out_mean'] = data_info['img'][self.target_mean_index].copy() #防止'img'在后续被改变后，'out_mean'也跟着改变
        data_info['out_std'] = data_info['img'][self.target_std_index].copy()
        # rgb数据[0~255]  这样在数据搬运过程中很快 DataPreprocessor 搬运完成在归一化 相当于搬运的是uint 而不是float
        #  https://mmengine.readthedocs.io/zh_CN/latest/tutorials/model.html
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        # 里面主要是反序列化的过程 真正的过程在 fullinit完成 主要是 self.data_list = self.load_data_list()
        data_info = self.get_data_info(idx)
        self.loadImageFromFile(data_info)
        return self.pipeline(data_info)
        # 不接受其它步骤的处理，只接受PackIcoInputs的处理
 
    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):#最大尝试次数 如果训练集找不到数据
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data
        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')


