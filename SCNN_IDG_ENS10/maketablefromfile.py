'''
The module is used to construct the adjcency table of the subdivision icosahedron
'''
import numpy as np
from .dggrid_runner import DGGRIDv7
from einops import rearrange
import os

dggridPath = '/home/dls/data/openmmlab/DGGRID/build/src/apps/dggrid/dggrid'
path = '/home/dls/data/openmmlab/mmclassification/data'
working_dir = os.path.join(path, 'sphere/temp')  ##存放临时文件


# save_dir_adj = os.path.join(path,'sphere/adjacency_table')
# save_dir_chi = os.path.join(path,'sphere/pool_table') ##存放临时文件
# save_dir_code =os.path.join(path,'sphere/code_table' )
# working_dir = '/home/dls/data/openmmlab/SpherePHD_py/DGGRID_train/temp'##存放临时文件 生成后自动删除
# save_dir_adj =  '/home/dls/data/openmmlab/SpherePHD_py/DGGRID_train/data/adjacency_table'
# save_dir_chi =  '/home/dls/data/openmmlab/SpherePHD_py/DGGRID_train/data/pool_table'

def make_res_code(dggs_type, resolution, save_dir):
    # 生成当前层的编码
    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    # print(gdf_cell_point['seqnum'])
    df_code = dggrid.getcode(dggs_type=dggs_type, resolution=resolution, save_dir=save_dir)
    # print(df_code.to_numpy().shape)
    return df_code.to_numpy()
    # 把编码转化为编码所对应的索引


def code2index(neiorchi, code):
    l, _ = neiorchi.shape
    # print('make_adjacency_table',df_nei,df_code)
    df_nei_r = rearrange(neiorchi, 'L  K ->(L K)')
    ##利用了广播机制 获取编码所在索引  https://blog.csdn.net/u014426939/article/details/109737841
    chunk_size = 100000
    current = 0
    size = df_nei_r.shape[0]
    # size = df_nei.shape[0]
    adj_list = []
    # 高层次的时候 df_code==df_nei[idxs,None] 会因为数量过多 导致崩溃
    while current < size:
        idxs = np.arange(current, min(size, current + chunk_size))
        adj_indice = np.where(code == df_nei_r[idxs, None])
        adj_list.append(adj_indice[-1])
        if current > 0:
            print(f"make_adjacency_table loop : {current}/{size}")
        current += chunk_size
    adjs = np.concatenate(adj_list)
    # print('make_adjacency_table adj_indice',df_code==df_nei)
    ##增加一个维度
    # df_code =df_code.reshape(l,1)
    adjs = rearrange(adjs, '(L  K )-> L K', L=l)
    return adjs

def creatadjtable(dggs_type, resolution, save_dir):
    """
        按照编码从大到小的顺序返回 编码的临近编码 是DGGRIDv7 getadjtable封装 结果直接保存在了save_dir里
    """
    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    dggrid.getadjtable(dggs_type=dggs_type, resolution=resolution, save_dir=save_dir)
def make_adjacency_table(dggs_type, resolution, save_dir):
    """_summary_

    Args:
        dggs_type (str, optional): 格网类型. Defaults to 'ISEA4D'.
        resolution (int, optional): 需要剖分的等级. Defaults to 6.
        返回的是编码所在的行索引值
    Returns:
        _type_: 返回resolution对应的邻接矩阵 按照格网编码的大小进行拍学
    """
    # 只有菱形 其余的还没有写
    # if "4D" in dggs_type:
    #     neinum =9
    # elif "4T" in dggs_type:
    #     neinum =10
    # elif "H" in dggs_type :
    #     neinum =7
    # else:
    #     raise ValueError('make_adjacency_table 只实现了菱形的')
    # adj_name = [ 'code' if x == 0 else f'nei_{x-1}' for x in range(neinum)]
    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    df_nei = dggrid.getadjtable(dggs_type=dggs_type, resolution=resolution, save_dir=save_dir)
    df_code = df_nei['code'].to_numpy()  ##用于比较产生编码所在的行索引值
    df_nei = df_nei.to_numpy()  # code 编码也要算进去
    adjs = code2index(df_nei, df_code)
    # l,_ = df_nei.shape
    # # print('make_adjacency_table',df_nei,df_code)
    # df_nei_r =rearrange(df_nei,'L  K ->(L K)'  )
    # ##利用了广播机制 获取编码所在索引  https://blog.csdn.net/u014426939/article/details/109737841
    # chunk_size =100000
    # current =0
    # size = df_nei_r.shape[0]
    # # size = df_nei.shape[0]
    # adj_list = []
    # #高层次的时候 df_code==df_nei[idxs,None] 会因为数量过多 导致崩溃
    # while current < size:
    #     idxs = np.arange(current, min(size, current +chunk_size))
    #     adj_indice = np.where(df_code==df_nei_r[idxs,None])
    #     adj_list.append(adj_indice[-1])
    #     if current>0:
    #         print(f"make_adjacency_table loop : {current}/{size}")
    #     current +=  chunk_size
    # adjs = np.concatenate(adj_list)
    # # print('make_adjacency_table adj_indice',df_code==df_nei)
    # ##增加一个维度
    # # df_code =df_code.reshape(l,1)
    # adjs =rearrange(adjs,'(L  K )-> L K',L=l)
    # # conv_indice =np.concatenate([df_code,adj_indice],axis=1)
    # # gdf_cell = dggrid.gen_cell( dggs_type=dggs_type,resolution = resolution)
    # # gdf = gdf_cell_point.join(df_nei_chi,on='seqnum',how ='inner')
    # # gdf = gdf_point
    # # return adj_indice,df_nei_chi

    return adjs


def make_pooling_table(dggs_type, resolution, save_dir, useAperture=0):
    """_summary_

    Args:
        dggs_type (str, optional): 格网类型.. Defaults to 'ISEA4D'.
        resolution (int, optional): 需要剖分的等级. Defaults to 6.
    Returns:
        _type_: _description_
    """
    '''
        池化就是从细变粗 所以 resolution 层的索引列表 其实是 resolution-1层的子格网编码(子格网编码对应resolution) ，对于seqmnum编码,编码是连续的 1 ~ N . 
        可以省略seqnnum编码，将所在行号+1 当做seqnum 但是对于二进制编码（把Q也编进去）他的编码值存储顺序可能不是连续的(也有可能是连续的如 001 代表第一个面  (前三位表示面编号))这样是不是一样了？后续在看
    '''
    assert resolution > 0, '0级为最底级无法再池化'

    # chi_name = ['code','0','1','2','3']# 注意 code列的编码是resolution-1列的十进制编码 0,1,2,3 对应的是resolution层的编码

    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    # print(gdf_cell_point['seqnum'])

    pool_res = resolution - 1
    df_chi = dggrid.getpartable(dggs_type=dggs_type, resolution=pool_res, save_dir=save_dir)
    # 排除掉'code'列 这样写的目的 是有可能孔径不是4 这样 就不能写[0,1,2,3] ，求 pool 不需要code编码
    if useAperture:  ## 使用对应孔径的编码
        chi_name = df_chi.columns[1:useAperture + 1]
    else:  ##使用所有邻域的数据
        chi_name = [x for x in df_chi.columns if x != 'code']
    df_chi = df_chi[chi_name].to_numpy()
    # 对于不连续值得二进制编码还需要再进行一次转换 求出值所在索引
    df_code = make_res_code(dggs_type=dggs_type, resolution=resolution, save_dir=save_dir)
    chi_indice = code2index(df_chi, df_code)
    # l,_ = df_chi.shape
    # df_chi =rearrange(df_chi,'L  K ->(L K)'  )
    # ##利用了广播机制 https://blog.csdn.net/u014426939/article/details/109737841
    # chi_indice = np.where(df_code==df_chi[:,None])[-1]
    # chi_indice =rearrange(chi_indice,'(L  K )-> L K',L=l)
    return chi_indice


def make_hex_pooling_table(dggs_type, resolution, save_dir, useAperture=4):
    """_summary_

    Args:
        dggs_type (str, optional): 格网类型.. Defaults to 'ISEA4D'.
        resolution (int, optional): 需要剖分的等级. Defaults to 6.
        useAperture 0代表不使用孔径 使用全部的邻近 大于0代表孔径 六边形有 3 4 7
    Returns:
        _type_: _description_
    """
    '''
        池化就是从细变粗 所以 resolution 层的索引列表 其实是 resolution-1层的子格网编码(子格网编码对应resolution) ，对于seqmnum编码,编码是连续的 1 ~ N . 
        可以省略seqnnum编码，将所在行号+1 当做seqnum 但是对于二进制编码（把Q也编进去）他的编码值存储顺序可能不是连续的(也有可能是连续的如 001 代表第一个面  (前三位表示面编号))这样是不是一样了？后续在看
    '''
    assert resolution > 0, '0级为最底级无法再池化'

    # chi_name = ['code','0','1','2','3']# 注意 code列的编码是resolution-1列的十进制编码 0,1,2,3 对应的是resolution层的编码

    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    # print(gdf_cell_point['seqnum'])

    pool_res = resolution - 1
    df_chi = dggrid.getpartable(dggs_type=dggs_type, resolution=pool_res, save_dir=save_dir)
    # 排除掉'code'列 这样写的目的 是有可能孔径不是4 这样 就不能写[0,1,2,3] ，求 pool 不需要code编码
    # chi_name = [ x  for x in df_chi.columns if x!='code']
    if useAperture:  ## 使用对应孔径的编码
        chi_name = df_chi.columns[1:useAperture + 1]
    else:
        chi_name = [x for x in df_chi.columns if x != 'code']
    df_chi = df_chi[chi_name].to_numpy()
    # 对于不连续值得二进制编码还需要再进行一次转换 求出值所在索引
    df_code = make_res_code(dggs_type=dggs_type, resolution=resolution, save_dir=save_dir)
    chi_indice = code2index(df_chi, df_code)
    # l,_ = df_chi.shape
    # df_chi =rearrange(df_chi,'L  K ->(L K)'  )
    # ##利用了广播机制 https://blog.csdn.net/u014426939/article/details/109737841
    # chi_indice = np.where(df_code==df_chi[:,None])[-1]
    # chi_indice =rearrange(chi_indice,'(L  K )-> L K',L=l)
    return chi_indice


def make_deconv_table(dggs_type, resolution, save_dir):
    # 由于转置卷积生成的编码就是当前层的子编码 但是排序不是从小到大 而是按照生成子编码的顺序生成的，如1	=> 1	2	65	66 ，
    # 详细见demo.ipynb 中 df_chi的输出 N,C0,L0, =>N,C1,L0*4 因此需要对生成的L0*4 按照编码顺序从大到小进行排序
    # 因此要处理一下重新排序 让生成的值根据 编码值一起排序，所以输出一下这个编码值 排序方法详解见 利用python进行数据分析第二版 附录A.6
    # 以前排序放在了sde_conv这个模型里，x = x[:,:,np.argsort(sort_array)] 为了合理性 现在直接输出相应的编码索引
    deconv_res = resolution  # 反卷积生成的就是下一层的编码
    # chi_name = ['code','0','1','2','3']# 注意 code列的编码是resolution-1列的十进制编码 0,1,2,3 对应的是resolution层的编码
    # chi_name = ['0','1','2','3']
    # 排除掉'code'列 这样写的目的 是有可能孔径不是4 这样 就不能写[0,1,2,3] ，求 pool 不需要code编码
    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    # print(gdf_cell_point['seqnum'])
    df_chi = dggrid.getchitable(dggs_type=dggs_type, resolution=deconv_res, save_dir=save_dir)
    chi_name = [x for x in df_chi.columns if x != 'code']
    # df_chi=process_chi(df_chi)
    chi_Array = df_chi[chi_name].to_numpy()
    chi_Array = rearrange(chi_Array, 'L  K -> (L K)')
    ##输出 编码排序过程中原始索引值
    chi_Array_indice = np.argsort(chi_Array)
    return chi_Array_indice


# def creatcellandcode(dggs_type,res,dggs_vert0_lon="0.0",dggs_vert0_lat ="90.0"):
#     dggrid = DGGRIDv7(executable= dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
#     gdf = dggrid.gen_cell_point(dggs_type,res,if_drop=True,dggs_vert0_lat=dggs_vert0_lat,dggs_vert0_lon=dggs_vert0_lon)
#     ##NOTE 如果使用填充曲线编码 则应该重新排序一下 这里使用了seqnum编码默认是排序的 就先不改了
#     return gdf

def creatcellandcode(dggs_type, res, dggs_vert0_lon="0.0", dggs_vert0_lat="90.0", dens="0"):
    dggrid = DGGRIDv7(executable=dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    gdf = dggrid.gen_cell_point(dggs_type, res, if_drop=True, dggs_vert0_lat=dggs_vert0_lat,
                                dggs_vert0_lon=dggs_vert0_lon, dens=dens)
    ##NOTE 如果使用填充曲线编码 则应该重新排序一下 这里使用了seqnum编码默认是排序的 就先不改了
    return gdf

# def main():
#     '''
#     Construct necessary tables
#     '''
#     dggs_type='ISEA4D'
#     resolution = 4
#     #if not os.path.exist('./tables'):
#         #os.mkdir('./tables')
#     #for i in range(1, 9):
#         #conv_table = make_conv_table(i)
#         #np.save('./tables/conv_table_'+str(i)+'.npy', conv_table)
#     pooling_table = make_pooling_table(dggs_type=dggs_type,resolution=resolution)
#     # print(pooling_table)
#     # np.save('./tables/pooling_table_'+str(i)+'.npy', pooling_table)
#     adj_table = make_adjacency_table(dggs_type=dggs_type,resolution=resolution)
#     print(f'pooling_table.shape {pooling_table }')
#     print(f'adj_table.shape {adj_table.shape}')

# #


# if __name__ == '__main__':
#     main()
