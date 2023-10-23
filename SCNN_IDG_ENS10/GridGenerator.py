import numpy as np
import os
from  .maketablefromfile import creatcellandcode
import geopandas as gpd
from einops import rearrange

class GridGenerator_icopoint:
  # 利用ico点的经纬度生成采样点
  def __init__(self,  dggs_type,res, imgHeight,imgWidth, kernel_size=(3,3) ):
    # 根据菱形格网level 生成经纬度点 最好提前存储好
    self.kernel_size = kernel_size  # (Kh, Kw)
    self.dggs_type =dggs_type
    self.res =res
    self.imgHeight = imgHeight
    self.imgWidth = imgWidth
    self.deltaLon = np.pi*2/imgWidth
    self.deltaLat = np.pi/imgHeight
    # print(f'imgHeight:{imgHeight},imgWidth:{imgWidth},deltaLon:{np.rad2deg(self.deltaLon)},deltaLat:{np.rad2deg(self.deltaLat)}')
  def create_ico_points(self,dggs_type,res,root = '/home/dls/data/openmmlab/letter2/sphereConv-pytorch'):
    """_summary_

    Args:
        dggs_type (_type_): _description_
        res (_type_): _description_
        root (str, optional): _description_. Defaults to '/home/dls/data/openmmlab/letter2/sphereConv-pytorch'.

    Returns:
        _type_: lonrange,latrange ico格网的经度和纬度 对于菱形格网 数量为 10*4^level 输出需要是弧度制 不然没法进行numpy的计算
    """ 
    # 测试maketablefromfile的creatcellandcode结果 为了生成GridGenerator_icopoint
    outpath = os.path.join(root,'ico_points_data' )
    os.makedirs(outpath,exist_ok=True)
    outname  = f'{dggs_type}{res}_p.geojson'
    # 判断文件是否存在
    if os.path.exists(os.path.join(outpath,outname)):
        gdf = gpd.read_file(os.path.join(outpath,outname))

        lonrange = list(np.deg2rad(np.array(gdf['geometry'].x)) )
        latrange = list(np.deg2rad(np.array(gdf['geometry'].y)))
    # gdf:geodatafarme 保存为geojson 文件
    else :
        gdf= creatcellandcode(dggs_type=dggs_type,res=res)
        gdf =gdf[['seqnum','point']]#不要cell列
        gdf.to_file(os.path.join(outpath,outname),driver='GeoJSON')
        lonrange = list(np.deg2rad(np.array(gdf['point'].x)) )
        latrange = list(np.deg2rad(np.array(gdf['point'].y)))
    return lonrange,latrange
  def createSamplingPattern(self):
    """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """
    kerX, kerY = self.createKernel()  # (Kh, Kw) 生成在0 0 点的时候 卷积核的中心点经纬度 公式7-10

    # create some values using in generating lat/lon sampling pattern 公式11
    rho = np.sqrt(kerX ** 2 + kerY ** 2)
    Kh, Kw = self.kernel_size
    # when the value of rho at center is zero, some lat values explode to `nan`.
    if Kh % 2 and Kw % 2:
      rho[Kh // 2][Kw // 2] = 1e-8
    nu = np.arctan(rho)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    # stride_h, stride_w = self.stride
    # h_range = np.arange(0, self.height, stride_h)
    # w_range = np.arange(0, self.width, stride_w)
    # 替换成 icopoints
    # lat_range = ((h_range / self.height) - 0.5) * np.pi
    # lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi) #经度范围是 -pi pi 这个程序的经度范围是 0 2*pi 应该是个bug
    lon_range,lat_range = self.create_ico_points(dggs_type=self.dggs_type,res=self.res) #按照文章的公式 生成的经纬度点范围应该是 -pi/2 pi/2 -pi pi
    # print(f'createSamplingPattern lat_range:{np.rad2deg(lat_range)}')
    # print(f'createSamplingPattern lon_range:{np.rad2deg(lon_range)}')
   
    # generate latitude sampling pattern 公式11
    lat = np.array([
      np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
    ])  # (L, Kh, Kw)

    # lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw) #不需要重复 ico 不是用meshgrid生成的 直接输出的事 经纬度点对
    # lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # generate longitude sampling pattern
    lon = np.array([
      lon_range[index]+np.arctan(kerX * sin_nu / (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for index ,_lat in enumerate(lat_range) 
    ])  # (L, Kh, Kw)
    # lon 可能会超出[-pi,pi]的范围  小于-pi的部分需要加上2*pi  大于pi的部分需要减去2*pi
    lon = np.where(lon>np.pi,lon-2*np.pi,lon)
    lon = np.where(lon<-np.pi,lon+2*np.pi,lon)
    
    # lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw) 只加对应位置的经度就行了 不用所有的经度都加 因为不是meshgrid生成的
    # lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # (radian) -> (index of pixel) 把经纬度转为行列号 https://blog.csdn.net/OrdinaryMatthew/article/details/127376503
    # lat = (lat / np.pi + 0.5) * self.height
    # lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width #前期处理好 把lon放在-pi 到pi 内
    # print(f'createSamplingPattern lat:{np.rad2deg(lat)}')
    # print(f'createSamplingPattern lon:{np.rad2deg(lon)}')
    # LatLon = np.stack((lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
    LatLon = self.getijfromlatlon(lon,lat ) # (2, L, Kh, Kw) = ((lat, lon), L, Kh, Kw) 纬度在前是因为先行后列
    # LatLon = LatLon.transpose((1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
    LatLon= rearrange(LatLon,'d L Kh Kw -> (L Kh ) Kw d' ) # (2, L*Kh, Kw) = ((lat, lon), L*Kh, Kw) 纬度在前是因为先行后列 =》 (L*Kh, Kw, 2) 为了spgereConv2d的输入
    LKh, Kw, d = LatLon.shape
    LatLon = LatLon.reshape((1, LKh,Kw, d))  # (1,L*Kh, Kw, 2)
    # H, Kh, W, Kw, d = LatLon.shape
    # LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)
    return LatLon

  def getijfromlatlon( self,lon, lat  ):
    """  根据 lat lon deltalat 和 deltalon 获取 lat lon 所在行列号

    Args: 
        lon (_type_): _description_ numpy or pytorch 输入是弧度制 不是角度制
        lat (_type_): _description_  numpy or pytorch    当前经纬度 
        deltalon (_type_): _description_  float 输入是弧度制 不是角度制 
        deltalat (_type_): _description_  float 经纬度格网每一个格网的经纬度间隔
    """
    # 要求输入的范围是 [0,pi] [0 2*pi] #生成的数据是从[-pi,pi] [-pi,2*pi]的 所以在此转换一下
    lon = lon + np.pi
    lat = -lat + np.pi/2 #纬度越大 行越小 
    # 判断经度等于2*pi的情况,维度等于0的情况 在这个情况下计算行列需要特殊处理 纬度越大 行越小 经度越大 列越大
    # diffvalue = 1e-6
    # maxlat  = np.pi
    # equal_maxlat = np.isclose(lat,maxlat ) 
    # lat_new = np.where(equal_maxlat, lat - diffvalue,lat) #把最小值加上一个值 防止超限
    # 不需要特殊处理 ，grid_Sample中默认左上角顶点为-1，-1所以相当于按照栅格的边长数作为行数 而不是中心点数作为行数
    # 也就对应格网的行数为self.imgHeight+1(如栅格数4*4 则行号为  0 1 2 3 4 五个) 而不是self.imgHeight 列同理
    # i = self.imgHeight - np.floor(lat_new/self.deltaLat) #纬度越大 行越小
    new_r = (lat)*(self.imgHeight)/np.pi 
    # maxlon = 2*np.pi
    # equal_maxlon = np.isclose(lon,maxlon )
    # lon_new = np.where(equal_maxlon, lon - diffvalue,lon)
    new_c = (lon)*(self.imgWidth)/(2*np.pi) #计算未缩放前的行列号(是个小数)
    
    # 缩放为-1 到 1 缩放的方式为 先缩放到0~1 再-1 缩放到-1~1
    # https://github.com/ChiWeiHsiao/SphereNet-pytorch/blob/master/spherenet/sphere_cnn.py
    # 本程序的缩放放在了spherecov2d的genSamplingPattern_ico中
    # coordinates[0] = (coordinates[0] * 2 / h) - 1
    # coordinates[1] = (coordinates[1] * 2 / w) - 1
    # j = np.floor(lon_new/self.deltaLon)
    # 利用stack进行返回 先行后列
    return np.stack((new_r,new_c))
  def createKernel(self):
    """
    :return: (Ky, Kx) kernel pattern
    #  spherenet的公式 7 -10
    """
    Kh, Kw = self.kernel_size

    delta_lat = np.pi / self.imgHeight
    delta_lon = 2 * np.pi / self.imgWidth

    range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      range_x = np.delete(range_x, Kw // 2)

    range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      range_y = np.delete(range_y, Kh // 2)

    kerX = np.tan(range_x * delta_lon)
    kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

    return np.meshgrid(kerX, kerY)  # (Kh, Kw)


