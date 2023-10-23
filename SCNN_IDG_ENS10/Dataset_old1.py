
import torch
from torch.utils.data import Dataset 
import numpy as np
import xarray as xr
from mmseg.registry import DATASETS
from mmseg.registry import TRANSFORMS
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from .PackIcoInputs import PackIcoInputs

@DATASETS.register_module()
class ENS10GridDataset(Dataset):
    """
    data_path: the path of folder storing the preprocessed ENS10 & ERA5 data
    target_var: one of "t850", "z500", "t2m" indicating target variable to predict
    dataset_type: one of "train", "test" indicating generating train (1998-2015) or test (2016-2017) dataset
    The dataset will return the mean/std of the target_var in ERA5 +
    the mean/std of all variables in ENS10 on the same pressure level
    """

    def __init__(self, data_path, target_var,

                 dataset_type="train", normalized=True, return_time=False,
                #  参考的mmengine BaseDataset里的参数
                #  pipeline: List[Union[dict, Callable]] = [],
                 max_refetch = 10,):
        self.return_time = return_time
        suffix = ""
        self.test_mode = False
        self.max_refetch = max_refetch
        # 提前读取数据
        if normalized:
            suffix = "_normalized"
        if dataset_type == "train":
            time_range = slice("1998-01-01", "2014-12-31")
            self.test_mode = False
        elif dataset_type == "val":
            time_range = slice("2015-01-01", "2015-12-31")
            self.test_mode = True
        elif dataset_type == "test":
            time_range = slice("2016-01-01", "2017-12-31")
            self.test_mode = True
        if target_var in ["t850", "z500"]:#预测的是t850或者z500 使用pl 变量
            # ds_mean = xr.open_dataset(f"{data_path}/ENS10_pl_mean{suffix}.nc", chunks={"time": 10}).sel(time=time_range)
            # ds_std = xr.open_dataset(f"{data_path}/ENS10_pl_std{suffix}.nc", chunks={"time": 10}).sel(time=time_range)
            # self.ds_scale_mean = xr.open_dataset(f"{data_path}/ENS10_pl_mean.nc", chunks={"time": 1},
            #                                      engine="h5netcdf").sel(time=time_range)
            # self.ds_scale_std = xr.open_dataset(f"{data_path}/ENS10_pl_std.nc", chunks={"time": 1},
                                                # engine="h5netcdf").sel(time=time_range)
            self.variables = ["Z", "T", "Q", "W", "D", "U", "V"]
            if target_var == "t850":
                # self.ds_mean = ds_mean.sel(plev=85000)
                # self.ds_std = ds_std.sel(plev=85000)
                self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_t850.nc", chunks={"time": 10}).sel(
                    time=time_range).isel(plev=0).T
                # self.ds_scale_mean = self.ds_scale_mean.sel(plev=85000).T
                # self.ds_scale_std = self.ds_scale_std.sel(plev=85000).T
            elif target_var == "z500":
                # self.ds_mean = ds_mean.sel(plev=50000)  
                # self.ds_std = ds_std.sel(plev=50000)
                self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_z500.nc", chunks={"time": 10}).sel(
                    time=time_range).isel(plev=0).Z
                # self.ds_scale_mean = self.ds_scale_mean.sel(plev=50000).Z
                # self.ds_scale_std = self.ds_scale_std.sel(plev=50000).Z
        elif target_var in ["t2m"]: #预测的是t2m 使用sfc 变量
            # self.ds_mean = xr.open_dataset(f"{data_path}/ENS10_sfc_mean{suffix}.nc", chunks={"time": 10}).sel(
            #     time=time_range)
            # self.ds_std = xr.open_dataset(f"{data_path}/ENS10_sfc_std{suffix}.nc", chunks={"time": 10}).sel(
            #     time=time_range)
            self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_sfc_t2m.nc", chunks={"time": 10}).sel(
                time=time_range).T2M
            # self.ds_scale_mean = xr.open_dataset(f"{data_path}/ENS10_sfc_mean.nc", chunks={"time": 1},
            #                                      engine="h5netcdf").sel(time=time_range).T2M
            # self.ds_scale_mean = xr.open_dataset(f"{data_path}/ENS10_sfc_mean.nc", chunks={"time": 1}).sel(time=time_range).T2M
            # # self.ds_scale_std = xr.open_dataset(f"{data_path}/ENS10_sfc_std.nc", chunks={"time": 1},
            # #                                     engine="h5netcdf").sel(time=time_range).T2M
            # self.ds_scale_std = xr.open_dataset(f"{data_path}/ENS10_sfc_std.nc", chunks={"time": 1}).sel(time=time_range).T2M
            self.variables = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']

        self.length = len(self.ds_era5)
        # 创建一个PackIcoInputs的实例
        self.pipeline = PackIcoInputs()
    def __len__(self):
        return self.length
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    def perpare_target_only(self,idx):
        targets = torch.as_tensor(self.ds_era5.compute().to_numpy()[idx])
        targets = targets.unsqueeze(0) ##为了后面组成batch
        
        results =dict()
        results['gt_seg_map'] = targets
        # results['scale_mean'] = scale_mean
        # results['scale_std'] = scale_std
        if self.return_time:
            results['time'] = self.ds_era5.time[idx].dt.strftime("%Y-%m-%d").item()
        # 不接受其它步骤的处理，只接受PackIcoInputs的处理
        return self.pipeline(results)
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        inputs = torch.zeros((len(self.variables) * 2, 361, 720))
        # print('self.ds_mean.shape',self.ds_mean )
        means = self.ds_mean.isel(time=idx).compute()
        stds = self.ds_std.isel(time=idx).compute()

        for k in range(len(self.variables)):
            variable = self.variables[k]
            inputs[2 * k, :] = torch.as_tensor(means[variable].to_numpy())
            inputs[2 * k + 1, :] = torch.as_tensor(stds[variable].to_numpy())

        targets = torch.as_tensor(self.ds_era5.compute().to_numpy()[idx])
        targets = targets.unsqueeze(0) ##为了后面组成batch
        # # 最后能不能都处理成numpy 一个一个的 这样加载可能会太慢
        # scale_mean = torch.as_tensor(
        #     self.ds_scale_mean.isel(time=idx).compute().to_numpy())
        # scale_std = torch.as_tensor(
        #     self.ds_scale_std.isel(time=idx).compute().to_numpy())

        results =dict()
        results['img'] = inputs 
        results['gt_seg_map'] = targets
        # results['scale_mean'] = scale_mean
        # results['scale_std'] = scale_std
        if self.return_time:
            results['time'] = self.ds_mean.time[idx].dt.strftime("%Y-%m-%d").item()
        # 不接受其它步骤的处理，只接受PackIcoInputs的处理
        return self.pipeline(results)
        

            # return self.ds_mean.time[idx].dt.strftime("%Y-%m-%d").item(), inputs, targets, scale_mean, scale_std
        # return inputs, targets, scale_mean, scale_std
         
    def __getitem__(self, idx):
        # if self.test_mode:
            # data = self.prepare_data(idx)
            data = self.perpare_target_only(idx)
            # if data is None:
                # raise Exception('Test time pipline should not get `None` '
                #                 'data_sample')
            return data

        # for _ in range(self.max_refetch + 1):#最大尝试次数 如果训练集找不到数据
            # data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            # if data is None:
            #     idx = self._rand_another()
            #     continue
            # return data
        # raise Exception(f'Cannot find valid image after {self.max_refetch}! '
        #                 'Please check your image path and pipeline')


