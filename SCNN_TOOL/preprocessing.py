
# 在时间维度计算均值和方差 作为归一化的均值和方差 而不是直接使用单一的均值和方差

import xarray as xr 
import numpy as np
import os
def numpy_save(path,data):
    print('======saving====== ',path)
    np.save(path,data)
    print('======saved======',path)

def get_mean_std_pl(data_root,data_name,variables=None,outDir = None):
    """
    data: xarray.DataArray
    """
    # nc中保存的顺序
    if outDir == None:
        outDir = data_root
    if variables == None:
        variables = ["Z", "T", "Q", "W", "D", "U", "V"] # 顺序不能变

    data_path = os.path.join(data_root,data_name)
    data = xr.open_dataset(data_path, chunks={"time": 400})[variables]
    
    data_5k =data.sel(plev=50000)

    mean5k = data_5k.mean(dim='time').to_array().values
    numpy_save(os.path.join(outDir,f'{data_name.split(".")[0]}_mean_50k.npy'),mean5k)
    # np.save(os.path.join(outDir,f'{data_name.split(".")[0]}_mean_50k.npy'),mean5k)s
 
    std5k = data_5k.std(dim='time').to_array().values
    numpy_save(os.path.join(outDir,f'{data_name.split(".")[0]}_std_50k.npy'),std5k)
    # np.save(os.path.join(outDir,f'{data_name.split(".")[0]}_std_50k.npy'),std5k)
 
    data_8k = data.sel(plev=85000) 
    mean8k = data_8k.mean(dim='time').to_array().values
    numpy_save(os.path.join(outDir,f'{data_name.split(".")[0]}_mean_80k.npy'),mean8k)
    # np.save(os.path.join(outDir,f'{data_name.split(".")[0]}_mean_80k.npy'),mean8k)
 
 
    std8k = data_8k.std(dim='time').to_array().values
    numpy_save(os.path.join(outDir,f'{data_name.split(".")[0]}_std_80k.npy'),std8k)
    # np.save(os.path.join(outDir,f'{data_name.split(".")[0]}_std_80k.npy'),std8k)
# 计算sfc的均值和方差

def get_mean_std_sfc(data_root,data_name,variables=None,outDir = None):

    if outDir == None:
        outDir = data_root
    if variables == None:
        variables =  ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']

    data_path = os.path.join(data_root,data_name)
    data_sfc = xr.open_dataset(data_path, chunks={"time": 400})[variables]

    mean_sfc = data_sfc.mean(dim='time').to_array().values
    outname = os.path.join(outDir,f'{data_name.split(".")[0]}_mean.npy') 
    numpy_save(outname,mean_sfc)
    # np.save(outname,mean_sfc)
 
    std_sfc = data_sfc.std(dim='time').to_array().values
    outname = os.path.join(outDir,f'{data_name.split(".")[0]}_std.npy') 
    numpy_save(outname,std_sfc)

    # np.save(outname,std_sfc)


if __name__ == "__main__":

    outDir = '/media/dls/WeatherData/ENS10/normalized/'





    variables_pl =None
    data_root = '/media/dls/WeatherData/ENS10/meanstd_pl'
    data_name_pl = ['ENS10_pl_mean.nc','ENS10_pl_std.nc']

    for name in data_name_pl:
        print('processing ',name)
        get_mean_std_pl(data_root,name,outDir=outDir,variables=variables_pl)


    variables_sfc = None
    data_root = '/media/dls/WeatherData/ENS10/meanstd'
    data_name = ['ENS10_sfc_mean.nc','ENS10_sfc_std.nc'] 

    for name in data_name:
        print('processing ',name)
        get_mean_std_sfc(data_root,name,outDir=outDir,variables=variables_sfc)