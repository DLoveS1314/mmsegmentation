# 新版本 更改了meshgrid 重新组织了代码 本文件主要是生成相应的索引文件 供icotransform使用  最终应该调用的函数为 cal_knn，生成的索引文件应该包含经纬度和二十面体格网的索引
# 生成的索引文件应该包含经纬度和二十面体格网的索引 
# pycharm测试

# 如何球面插值？
# weatherbench  使用xesmf 示例包括  regrid.py 中的方法 python_practice/interpolater/main.py python_practice/testcf/
# 或者使用 weyn中的 Tempest-Remap library

# 插值新方法 借鉴Graphcast 把插值融入深度学习过程中 建立Encoder 把临近的几个经纬度的值作为输入，通过卷积模块（conv+Bn+Relu） 输出一个值 作为离散格网的点属性值
# 建立 Decoder 把临近的格网点数据作为输入，通过卷积模块（conv+Bn+Relu） 输出一个值 作为一个经纬度的点属性值


# 借鉴Graphcast的思想 提供一些基本的索引创建函数 核心函数 包括
# calknnei 根据erp投影的索引数据 和 dggrid 的投影数据 计算每个经纬度点的最近的k个格网点和每个格网点的最近的k个经纬度点
# get_dggrid_index_table 生成dggrid的索引数据
# create_Index_table 根据经度和纬度格网数 生成ERP投影的索引数据
from tqdm import tqdm
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
 
from .maketablefromfile import creatcellandcode


# 为了创建经纬度编码使用的辅助函数
def getequij(codes,MaxJ ):
    """_summary_ 根据erp投影的编码获取行列号

    Args:
        codes (numpy or pytorch ): array 经纬度的编码
        MaxJ (_type_): 最大列数 不是最大列的序号数

    Returns:
        (numpy or pytorch ): 返回行号和列号
    """    
    # codes 从0开始计数 行从上到下 列从左到右
    i =  codes // MaxJ
    j =  codes % MaxJ
    return  i,j 
def getequcode( i,j,MaxJ):
    # 先行后列进行一维展开 
    # print(f'q:{q},i:{i},j:{j}')
    codes =  i*MaxJ + j  
    return codes
def getlatlon_dgree(i,j,deltai,deltaj,MaxI,israd=False ):
    # 给i j 求经纬度 NOTE 没有用 也没有验证 这个函数在新版本应该错误了
    # i,j 从0开始计数 结果的范围是 [-180~180] [-90~90]MaxI最大行数 不是最大行的序号数
    # 行从上到下 列从左到右
    if israd:#如果是弧度制
        deltai = deltai * 180 / torch.pi
        deltaj = deltaj * 180 / torch.pi
    lon = j*deltaj+0.5*deltaj-180
    lat = (MaxI-i)*deltai  - 0.5*deltai-90 # 行从上到下 维度从下到上是增大
    return (lon,lat)
def getlatlonbycodes(codes,tablepath):
    # NOTE 没有用 也没有验证
    equ_table = pd.read_csv(tablepath)
    result = equ_table.query('codes in @codes')
    latlon =result[['lon','lat']]
    return latlon
def linespace (delta):#
    # 生成气象版本的linspace,生成的是经纬度格网的顶点，经度从-180到180-deltaLon 维度从-90到90 所以对于0.25°的格网来说，经度是从-180到179.75 维度是从-90到89.75
    # 纬度格网数为721 经度格网数为1440
    # /home/dls/data/openmmlab/ens10/mydatatest.ipynb 数据展示中可以发现 lon结果0到359.75 lat结果是90到90 要和气象数据对应上 这个决定了谁是第一列
    # 如果从0开始 则 0是第一列 如果从-180开始 则-180是第一列
    # lon = np.arange(-180,180,delta)
    lon = np.arange(0,360,delta)
    # 按照地理坐标系的规则 维度从下到上是增大
    lat = np.arange(90,-90-delta,-delta)
    # print('lon',lon.shape)
    # print('lat',lat.shape)
    return lon,lat
def meshgrid (delta):
    return np.meshgrid(*linespace(delta) )    
#  经纬度 转为 x y z
def ll2xyz(lon,lat, israd ):
    # 输入是弧度制
    if not israd:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
    x_ = np.cos(lat) * np.cos(lon)
    y_ = np.cos(lat) * np.sin(lon)
    z_ = np.sin(lat)
    return x_,y_,z_
def xyz2ll(X,Y,Z):
    lonR = np.arctan2(Y,X) #  经度[-180,180]
    xy_r = np.sqrt(X^2+Y^2) 
    latR = np.arctan(Z/xy_r) # 维度[-90,90]
    return lonR,latR

# 创建 离散格网或者经纬度格网编码与经纬度的对应关系 为生成对应的索引表（calknnei）做准备
def create_dggrid_index_table(dggtype,res,dggs_vert0_lon= "0.0",dggs_vert0_lat= "90.0",path=os.path.join(os.getcwd(),'index_table')):
    name = f'{ dggtype}_{ res}_{ dggs_vert0_lon}_{ dggs_vert0_lat}.csv'
    outpath =os.path.join(path,name)
    if os.path.isfile(outpath):
        print(f'get_dggrid_index_table: {outpath} has exist')
        dggrid_data = pd.read_csv(outpath)
    else:
        gdf=creatcellandcode( dggtype, res,dggs_vert0_lon= dggs_vert0_lon,dggs_vert0_lat = dggs_vert0_lat)
        gdf =gdf.sort_values(by='seqnum',ascending=True)
        dggrid_data = pd.DataFrame()
        dggrid_data['seqnum'] = gdf['seqnum']
        dggrid_data['lon'] = gdf['point'].x
        dggrid_data['lat'] = gdf['point'].y
        print(f'create_dggrid_index_table is saving {outpath}')
        dggrid_data.to_csv(path_or_buf=outpath,index=False)
    print(f'create_dggrid_index_table:{dggrid_data.shape}')
    
    return dggrid_data
def create_erp_index_table(delta,path=os.path.join(os.getcwd(),'index_table')):
    
    """根据经纬度间隔 生成气象专用的经纬度索引表 包含编码 经纬度 行列号 纬度包含90 和 -90 经度包含180-delta 和 -180
    Args:
        delta (_type_): 经纬度间隔 单位需要是度数
        path (_type_, optional): 输出的文件夹路径. Defaults to os.path.join(os.getcwd(),'index_table').
    """    
    # 创建经纬度的索引表 包含编码 经纬度 行列号
    os.makedirs(path,exist_ok=True)
    outpath = os.path.join(path,f'equ_{delta}.csv')
    if os.path.isfile(outpath):
        print(f'create_index_table: {path} has exist')
        index_table = pd.read_csv(outpath)
    else :
        index_table = pd.DataFrame()
        lon, lat =meshgrid(delta) 
        index_mapping = np.array(list(np.ndindex(lon.shape)))
        # print('index_mapping',index_mapping)
        i = index_mapping[:,0]
        j = index_mapping[:,1]
        codes =getequcode(i,j,lon.shape[1])
        index_table['codes'] = codes
        index_table['lon'] = lon.flatten()
        index_table['lat'] = lat.flatten()
        index_table.to_csv(path_or_buf=os.path.join(path,f'equ_{delta}.csv'),index=False)
    print(f'create_index_table:{index_table.shape}')
    return index_table
def cal_euclidean_distance(point, matrix):
    """ 用于apply的自定义函数 计算一个点 point 到矩阵 matrix 中每个点的欧氏距离
    Args:
        point (_type_): (x,y,z), 一个点的xyz坐标
        matrix (_type_): (n,3) 一个矩阵的xyz坐标

    Returns:
        _type_:  (n,) 一个向量 每个元素是一个点到矩阵中每个点的欧氏距离
    """    
    # 创建一个与矩阵相同维度的向量
    repeated_vector = np.tile(point, (matrix.shape[0], 1))
    # 计算差值矩阵
    diff_matrix = repeated_vector - matrix
    # 平方差值矩阵的每个元素
    squared_diff_matrix = np.square(diff_matrix)
    # 沿着轴求和差值矩阵的元素
    summed_diff_matrix = np.sum(squared_diff_matrix, axis=1)
    # 开方和，得到欧氏距离矩阵 不需要开平方
    return summed_diff_matrix
def get_smallest_indices(vector, k):
    """ 用于apply的自定义函数 计算一个向量中最小的k个值的索引

    Args:
        vector (_type_):  cal_euclidean_distance 返回的欧式距离向量
        k (_type_):  最小的k个值

    Returns:
        _type_:  最小的k个值的索引
    """    
    sorted_indices = np.argsort(vector)  # 对向量进行排序，并返回排序后的索引
    smallest_indices = sorted_indices[:k]  # 取前k个最小值的索引
    return smallest_indices
def equapply(row,dggrid_xyz,dggridindex,k=8,dggrid_level=None):
    """ 用于pandas 的apply的自定义函数 
        计算 equ dataframe 中的每一个经纬度点的最近的k个二十面体dggrid_level层级的格网点
    Args:
        row (pd.serise):  equ dataframe的一行, 每个格网点 包含列名 codes,lon,lat,i,j
        dggrid_xyz (np.ndarray): dggrid_level 层二十面体格网的xyz坐标,顺序很重要,要按照seqnum从小到大的方式排列
        dggrid_level (_type_): int 格网的层级 用于计算q i j 
        k (int, optional): _description_. Defaults to 8.
    Returns:
        _type_: _description_
    """        
    lonlat = row[['lon','lat']].values
    equxyz= np.array(ll2xyz(lonlat[0],lonlat[1],israd=False)).T
    disarray = cal_euclidean_distance(equxyz,dggrid_xyz)
    smallest_indices = get_smallest_indices(disarray,k) #得到的是最大值索引 不是seqnum编码
    seqnumcodes = smallest_indices+1 # seqnum 从1开始计数
    # qijarray = np.array(getqij(seqnumcodes,dggrid_level)) 只需要输出编码即可
    # lon_d = dggridindex.iloc[smallest_indices][ 'lon'  ].values.squeeze()
    # lat_d = dggridindex.iloc[smallest_indices][ 'lat'  ].values.squeeze()
    return seqnumcodes   

def dggridapply(row,equxyz, equindex, k=8):
    """  用于pandas 的apply的自定义函数

    Args:
        row (_type_): dggrid dataframe的一行, 每个格网点 包含列名 seqnum,lon,lat
        equxyz (_type_):   经纬度格网的 xyz坐标矩阵,顺序很重要,要按照codes从小到大的方式排列
        equindex (_type_): 经纬度格网的 索引数据
        k (int, optional):  _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """    
    lonlat_d = row[['lon','lat']].values #提取二十面体格网每一行的经度和纬度
    dggridxyz= np.array(ll2xyz(lonlat_d[0],lonlat_d[1],israd=False)).T #把经纬度转化为 矩阵
    disarray = cal_euclidean_distance(dggridxyz,equxyz).copy()
    smallest_indices = get_smallest_indices(disarray,k).copy()#得到的是前k个最小值索引  经纬度格网的编码是从0开始的 所以可以直接用 
    # NOTE 不加copy 内存会激增 但是equapply 没有这个问题？？ https://zhuanlan.zhihu.com/p/80689571
    # disarray
    # ijarray = np.array(getequij(smallest_indices,MaxJ=MaxJ))只需要输出编码和经纬度即可
    # lon_equ = equindex.iloc[smallest_indices]['lon'].values.squeeze()
    # lat_equ = equindex.iloc[smallest_indices]['lat'].values.squeeze()
    return smallest_indices 

def calknnei(equpath,dggridpath,dggrid_level,knei =8,outpath =os.path.join(os.getcwd(),'index_table')):
    """
    总函数 根据erp投影的索引数据 和 dggrid 的投影数据 计算每个经纬度点的最近的k个格网点和每个格网点的最近的k个经纬度点
    Args:
        equpath (_type_): ERP投影的索引数据路径 csv格式 pd调用
        dggridpath (_type_):  dggrid 的索引数据路径 csv格式 pd调用
        dggrid_level (_type_): 格网的层级 额外需要的信息
        MaxJ (_type_): erp投影的最大列数 是ij转编码需要的额外信息
        knei (int, optional): 求多少邻近数 Defaults to 8.
    Returns:没有return 顶层函数
    """    
    # 如果路径不存在 不能继续
    if not os.path.isfile(equpath):
        print(f'equpath:{equpath} is not exist')
        return
    if not os.path.isfile(dggridpath):
        print(f'dggridpath:{dggridpath} is not exist')
        return
    # 读取数据并按照各自的编码排序 从小到大升序排序 这个顺序很重要 必须要排 因为后面的计算按照这个顺序来的
    equindex = pd.read_csv(equpath).sort_values(by='codes',ascending=True)
    dggridindex = pd.read_csv(dggridpath).sort_values(by='seqnum',ascending=True)
    # 根据格网点经纬度生成xyz
    dggrid_xyz = ll2xyz(dggridindex['lon'].values,dggridindex['lat'].values,israd=False)
    dggrid_xyz = np.array(dggrid_xyz).T
    #  根据经纬度生成xyz
    equ_xyz = ll2xyz(equindex['lon'].values,equindex['lat'].values,israd=False)
    equ_xyz = np.array(equ_xyz).T


    dggrid_basename  =os.path.splitext(os.path.basename(dggridpath))[0]
    equ_basename = os.path.splitext(os.path.basename(equpath))[0]
    dggrid_out = pd.DataFrame()
    dggrid_out['seqnum'] = dggridindex['seqnum']
    # 给每个格网点找到最近的8个经纬度点
    # dggridindex['level'] = dggrid_level.
    tqdm.pandas(desc="get feature in dggridindex")
    # dggridindex[['codeskmin','lon_equ','lat_equ']]= dggridindex.progress_apply(dggridapply,args=(equ_xyz, equindex,knei),axis=1, result_type="expand")
    dggrid_out['codeskmin' ]= dggridindex.progress_apply(dggridapply,args=(equ_xyz, equindex,knei),axis=1 )
    dggrid_out.to_pickle(path=os.path.join(outpath,f'{dggrid_basename}_k{knei}l{dggrid_level}_{equ_basename}.pkl') )
    # dggrid_out.to_csv(path_or_buf=os.path.join(outpath,f'{dggrid_basename}_k{knei}l{dggrid_level}_{equ_basename}.csv'),index=False)
    # 保存数据 csv 和 pkl 真正的数据存储再pkl # 在csv时 矩阵无法存储 会被存储为字符串 包括read_csv读取时也会读取为字符串 所以要存储为pkl
    # dggridindex[['codeskmin','lon_equ','lat_equ']]= dggridindex.apply(dggridapply,args=(equ_xyz, equindex,knei),axis=1, result_type="expand")


    equ_out = pd.DataFrame()
    equ_out['codes'] = equindex['codes']
     # 给每个经纬度点找到最近的8个格网点
    tqdm.pandas(desc="get feature in equindex")
    # equindex[['seqnumkmin','lon_d','lat_d']]= equindex.progress_apply(equapply,args=(dggrid_xyz, dggridindex,knei),axis=1, result_type="expand")
    equ_out[ 'seqnumkmin' ]= equindex.progress_apply(equapply,args=(dggrid_xyz, dggridindex,knei),axis=1  ) #用不到那么多参数
    equ_out.to_pickle(path=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.pkl') )
    # equ_out.to_csv(path_or_buf=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.csv'),index=False)
    #
    # equindex.to_csv(path_or_buf=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.csv'),index=False)
    # equindex.to_pickle(path=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.pkl') )
 

def get_ico2erp_table(delta,rootpath,dggs_type, dggrid_level,knei):
    # 只计算二十面体转化为erp时 erp上的每个经纬度离得最近的格网点
    save_path = os.path.join(rootpath,f'equ_{delta}_k{knei}l{dggrid_level}_{dggs_type}.pkl') 
    if not os.path.isfile(save_path):
        # 创建经纬度格网
        erp_index=create_erp_index_table(delta,rootpath).sort_values(by='codes',ascending=True)
        #创建二十面体格网
        dggrid_index= create_dggrid_index_table( dggs_type, res = dggrid_level , path= rootpath ).sort_values(by='seqnum',ascending=True)

        #  根据经纬度生成xyz
        equ_xyz = ll2xyz(erp_index['lon'].values,erp_index['lat'].values,israd=False)
        equ_xyz = np.array(equ_xyz).T
        
        # 根据格网点经纬度生成xyz
        dggrid_xyz = ll2xyz(dggrid_index['lon'].values,dggrid_index['lat'].values,israd=False)
        dggrid_xyz = np.array(dggrid_xyz).T
    
        equ_out = pd.DataFrame()
        equ_out['codes'] = erp_index['codes']
        # 给每个经纬度点找到最近的8个格网点
        tqdm.pandas(desc="get feature in equindex")
        # equindex[['seqnumkmin','lon_d','lat_d']]= equindex.progress_apply(equapply,args=(dggrid_xyz, dggridindex,knei),axis=1, result_type="expand")
        equ_out[ 'seqnumkmin' ]= erp_index.progress_apply(equapply,args=(dggrid_xyz, dggrid_index,knei),axis=1  ) #用不到那么多参数
        equ_out.to_pickle(path=save_path)
        return equ_out
    else:
        return pd.read_pickle(save_path)
    # equ_out.to_csv(path_or_buf=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.csv'),index=False)
    #
    # equindex.to_csv(path_or_buf=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.csv'),index=False)
    # equindex.to_pickle(path=os.path.join(outpath,f'{equ_basename}_k{knei}l{dggrid_level}_{dggrid_basename}.pkl') )


 
def test_getequcode():
    halflonnum  = 20
    halflatnum  = 30 
    lon, lat =meshgrid(halflonnum,halflatnum) 
    index_mapping = np.array(list(np.ndindex(lon.shape)))
    print('index_mapping',index_mapping)
    i = index_mapping[:,0]
    j = index_mapping[:,1]
    codes =getequcode(i,j,halflonnum*2)
    print('codes',codes)
    print('lon',lon.flatten())
    print('lat',lat.flatten())
def test_apply() :
    # 测试apply 求的过程是否正确 修改自 calknnei 输出的disarray应该升序排列
    def inner_func(equpath,dggridpath,dggrid_level, knei =8):
        if not os.path.isfile(equpath):
            print(f'equpath:{equpath} is not exist')
            return
        if not os.path.isfile(dggridpath):
            print(f'dggridpath:{dggridpath} is not exist')
            return
        # 读取数据并按照各自的编码排序 从小到大升序排序 这个顺序很重要 必须要排 因为后面的计算按照这个顺序来的
        equindex = pd.read_csv(equpath).sort_values(by='codes',ascending=True)
        dggridindex = pd.read_csv(dggridpath).sort_values(by='seqnum',ascending=True)
        # 根据格网点经纬度生成xyz
        dggrid_xyz = ll2xyz(dggridindex['lon'].values,dggridindex['lat'].values,israd=False)
        dggrid_xyz = np.array(dggrid_xyz)
        dggrid_xyz_T =  dggrid_xyz .T
        #  根据经纬度生成xyz
        equ_xyz = ll2xyz(equindex['lon'].values,equindex['lat'].values,israd=False)
        equ_xyz = np.array(equ_xyz).T
        # equapply(equindex.iloc[0],dggrid_xyz,dggrid_level,dggridindex,knei)
        smallest_indices,_,_  = dggridapply(dggridindex.iloc[0],equ_xyz ,equindex,knei)
        print('smallest_indices',smallest_indices)
        
        lonlat_d = dggridindex.iloc[0][['lon','lat']].values
        dggridxyz= np.array(ll2xyz(lonlat_d[0],lonlat_d[1],israd=False)).T
        disarray = cal_euclidean_distance(dggridxyz,equ_xyz)
        print('disarray',disarray[smallest_indices])
        # # 给每个经纬度点找到最近的8个格网点
        # equindex[['seqnumkmin','q','i_d','j_d','lon_d','lat_d']]= equindex.apply(equapply,args=(dggrid_xyz,dggrid_level,dggridindex,knei),axis=1, result_type="expand")
        # equindex['level'] = dggrid_level

 
    dggrid_level = 4  #二十面体格网的层级
    delta = 0.25
    dgg_type = 'ISEA4D'
    outpath=os.path.join(os.getcwd(),'index_table')
    create_index_table(delta,path = outpath)
    create_dggrid_index_table( dgg_type, res = dggrid_level , path= outpath )
    knei =8
    equpath = os.path.join(outpath,f'equ_{delta}.csv')
    dggridpath = os.path.join(outpath,f'{dgg_type}_{dggrid_level}_0.0_90.0.csv')
    inner_func(equpath,dggridpath,dggrid_level ,knei =knei)
    
# 最终的测试函数 生成索引文件 和计算最近的k个格网点和经纬度点
def test_calknnei():
    dggrid_level = 6#二十面体格网的层级
    delta = 0.5
    dgg_type = 'ISEA4D'
    outpath=os.path.join(os.getcwd(),'index_table')
    create_index_table(delta,path = outpath)
    create_dggrid_index_table( dgg_type, res = dggrid_level , path= outpath )
    knei =8
    equpath = os.path.join(outpath,f'equ_{delta}.csv')
    dggridpath = os.path.join(outpath,f'{dgg_type}_{dggrid_level}_0.0_90.0.csv')
    calknnei(equpath,dggridpath,dggrid_level ,knei =knei)
def test_read_file():
    # 保存csv时 float向量保存为了字符串 测试读取时会出现什么情况
    path= '/home/dls/data/openmmlab/letter2/equ_lon40_lat60_k8_FULLER4D_4_0.0_90.0.pkl'
    data = pd.read_pickle(path)
    print(type(data[ 'lon_d' ][0]) )
    print( data[ 'lon_d' ][0]) 
def test_cal_euclidean_distance():
    # 测试计算欧式距离的函数是否正确
    point = np.array([1,1,1])
    matrix = np.array([[1,1,1],[2,2,2],[3,3,3]])
    euclidean_distance_array = cal_euclidean_distance(point,matrix)
    print('disarray',euclidean_distance_array)

 
if __name__ == '__main__':
    test_calknnei()
    # test_apply()
    # test_cal_euclidean_distance()
 