"""
对 testexp 中的icotransform 进行封装 包含 Graphcast中的encoder 和 decoder
"""
# 基于卷积的球面插值，借鉴Graphcast 把插值融入深度学习过程中 建立Encoder 把临近的几个经纬度的值作为输入，通过卷积模块（conv+Bn+Relu） 输出一个值 作为离散格网的点属性值
# 建立 Decoder 把临近的格网点数据作为输入，通过卷积模块（conv+Bn+Relu） 输出一个值 作为一个经纬度的点属性值
# 这里建立模型需要的的卷积块 包括idg2erp 和 erp2idg
# 模型中需要用到的pd表数据 都在icotransformfuc.py中的testcalknei()生成
# 生成IDG_pdading idg2erp 和 erp2idg
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from .SphereConv2d import SphereConv2d_icopoint
from .icotransfomrfunc import get_ico2erp_table,get_ico2erp_knn ,get_erp2ico_knn
eps = 1.0e-5
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer,build_upsample_layer,build_padding_layer,build_conv_layer
class SpatialAttentionModule_v1(nn.Module):
    '''
        # sphereconv2d中有一个 conv2d实现的空间卷积 这里 实现一个新的注意力 给kernel_size个点赋予不同的权重 
        对于erp2igd x => b,c,L,9  对于igd2erp => (b,q) C,L1,9
        所以 b ,c ,l,ks => b,ks,l =>  b,ks,l =>sigmod进行归一化 
    '''
    def __init__(self,kernel_size=9  ):
        super(SpatialAttentionModule_v1, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=kernel_size, kernel_size=(1,kernel_size), stride=1, padding=0) #逐行卷积 保持输入的大小不变
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True) #通道维度上求平均值 
        maxout, _ = torch.max(x, dim=1, keepdim=True) # 通道维度上求最大值
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv(out).squeeze(-1)
        out = self.sigmoid(out)
        out = rearrange(out, 'b c l -> b 1 l c')
        out = out * x + x #是否应该加shortcut
        return out
class erp2igd_encoder(nn.Module):
    def __init__(self, in_channels, k_size, dgg_kn_erp, dggrid_level, p=0):
        """ 初始化函数 把经纬度格网的数据转化为二十面体格网数据 借鉴Graphcast的思想 命名为encoder041
          Args:
              in_channels (_type_): 输入的通道数
                k_size (_type_): 卷积核的大小
                dgg_kn_erp (_type_): _description_ 最邻近的八个经纬度点的索引 路径
                dggrid_level (_type_): _description_ 二十面体的层数
                p (_type_): _description_ dropout的概率
        """
        super(erp2igd_encoder, self).__init__()
        # 只需要使用组卷积 不使用可分离卷积 也不用增加通道数 只当作插值来用
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k_size), padding=0, groups=in_channels)
        # self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.drop_out = nn.Dropout(p=p) if p > eps else nn.Identity()
        self.act = nn.ReLU()
        # if not DEBUG:

        self.table = pd.read_pickle(dgg_kn_erp)
        # print('self.table', self.table.head(10))
        self.table = self.table.sort_values(by=['seqnum'], ascending=True)  # 再次确认是升序列
        codeskmin_ndarray = np.stack(self.table['codeskmin'].values, axis=0)  # 维度为[20面体格网数量，8]
        codeskmin_tensor = torch.as_tensor(codeskmin_ndarray)
        # print('torch.max(codeskmin_tensor )',torch.max(codeskmin_tensor ))
        self.register_buffer('codeskmin', codeskmin_tensor)
        self.maxij_dggrid = 2 ** dggrid_level
        
        # else:
        #     self.maxij_dggrid = 2 ** dggrid_level
        #     codeskmin_tensor = torch.as_tensor(np.random.randint(0, self.maxij_dggrid * self.maxij_dggrid,
        #                                                          [10 * self.maxij_dggrid * self.maxij_dggrid, k_size]),
        #                                        dtype=torch.int64)
        #     self.register_buffer('codeskmin', codeskmin_tensor)

    def forward(self, x):
        # x: [batch_size, channel, H, W] 输入为erp投影
        # print('x0',x.shape)
        x = rearrange(x, 'b c h w -> b c (h w)')
        # print('x1',x.shape)
        # x_expand : [batch_size, channel, 10*4**L, 8]  10*4**L => B，C，二十面体格网 8 =>每个格网点最近的8个经纬度点
        x_expand = x[:, :, self.codeskmin]
        # print('x_expand',x_expand.shape)
        out = self.depth_conv(x_expand).squeeze()
        # out =self.point_conv(out).squeeze()
        out = self.act(out)
        # 转化为二十面体十个面的数据 [batch_size, channel, 10*4**L] => [batch_size, channel, 10, 2**L, 2**L]
        # 球面格网使用的数据维度是 (b q ) c  h w
        out = rearrange(out, 'b c (q h w ) -> (b q ) c  h w', q=10, h=self.maxij_dggrid, w=self.maxij_dggrid)
        return out

class encoder_v2(nn.Module):
    def __init__(self, in_channels, kernel_size, delta,dggird_type,rootpath,dggrid_level,p=0,useAttention=True,
                 norm_cfg = dict(type='BN'),act_cfg=dict(type='Swish')):
        """ 初始化函数 把经纬度格网的数据转化为二十面体格网数据 借鉴Graphcast的思想 命名为encoder041
          Args:
              in_channels (_type_): 输入的通道数
                k_size (_type_): 卷积核的大小
                dgg_kn_erp (_type_): _description_ 最邻近的八个经纬度点的索引 路径
                dggrid_level (_type_): _description_ 二十面体的层数
                p (_type_): _description_ dropout的概率
        """
        super(encoder_v2, self).__init__()
        # 只需要使用组卷积 不使用可分离卷积 也不用增加通道数 只当作插值来用 padding设置为0 逐渐行卷积的时候不会改变输入的大小
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=0, groups=in_channels)
        # self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.drop_out = nn.Dropout(p=p) if p > eps else nn.Identity()
        # self.act = nn.ReLU()
        self.indices = get_erp2ico_knn(delta=delta,rootpath=rootpath,dggs_type=dggird_type,dggrid_level=dggrid_level,knei=kernel_size)
        self.codeskmin = torch.tensor(self.indices)
        self.att = SpatialAttentionModule_v1(kernel_size=kernel_size) if useAttention else nn.Identity()
        self.maxij_dggrid = 2 ** dggrid_level
        
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
                # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', True)
            self.act = build_activation_layer(act_cfg_)
        if self.with_norm:
            norm_channels = in_channels
            _, self.norm = build_norm_layer(
                norm_cfg, norm_channels)
    def forward(self, x):
        # x: [batch_size, channel, H, W] 输入为erp投影
        # print('x0',x.shape)
        x = rearrange(x, 'b c h w -> b c (h w)') #按行展开
        # print('x1',x.shape)
        # x_expand : [batch_size, channel, 10*4**L, 8]  10*4**L => B，C，二十面体格网 8 =>每个格网点最近的8个经纬度点
        x_expand = x[:, :, self.codeskmin] #编码也是按行扫描，得到的是按照seqnum的顺序的数据 seqnum 是按照先 q 后 h 最后 w 的顺序排列的
        # print('x_expand',x_expand.shape)
        x_expand = self.att(x_expand) #把每个格网点的9个经纬度点的数据进行注意力加权
        out = self.depth_conv(x_expand).squeeze(-1)
        out = rearrange(out, 'b c (q h w ) -> (b q ) c  h w', q=10, h=self.maxij_dggrid, w=self.maxij_dggrid)
        out = self.norm(out) if self.with_norm else out
        out = self.drop_out(out)
        # out =self.point_conv(out).squeeze()
        out = self.act(out) if self.with_activation else out
        # 转化为二十面体十个面的数据 [batch_size, channel, 10*4**L] => [batch_size, channel, 10, 2**L, 2**L]
        # 球面格网使用的数据维度是 (b q ) c  h w
        return out
class encoder_v3(nn.Module):
    """使用spherenet的方法进行 

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """    
    def __init__(self, in_channels: int, out_channels: int, 
                kernel_size=(3, 3),
            #    stride=(3,1)  ,
                bias: bool = True,  
                dggs_type = "FULLER4D",
                dggrid_level :int = 6 ,
                grid_mode = "bilinear" ,
                p=0) -> None:
        # 一般输出的outchannel 是1 因此可以再二十面体上将inchannel 转为 outchannel个通道 然后再使用分组卷积转为了
        super(encoder_v3, self).__init__()
        stride = (kernel_size[0],1)
        self.model = SphereConv2d_icopoint(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias,
                                           dggs_type=dggs_type, res=dggrid_level, grid_mode=grid_mode)
        # print(f'dggrid_level:{dggrid_level},dggs_type:{dggs_type},grid_mode:{grid_mode}')
        self.maxij_dggrid = 2 ** dggrid_level
        self.drop_out = nn.Dropout(p=p) if p > eps else nn.Identity()
        
    def forward(self, x):
        out = self.drop_out(x)
        out = self.model(out).squeeze(-1)
        # print('encoder_v2 out',out.shape)
        out = rearrange(out, 'b c (q h w ) -> (b q ) c  h w', q=10, h=self.maxij_dggrid, w=self.maxij_dggrid)
        return out
                                           
class idg2erp_decoder(nn.Module):
    """
        将最终的二十面体格网数据转化为经纬度格网数据作为输出进行精度验证
    """
    def __init__(self, in_channels, k_size, delta,dggird_type,rootpath,dggrid_level,p=0) -> None:
        """_summary_

        Args:
            in_channel (_type_): 输入通道 本文的该输入通道一般是1或者2 如果把均值和方差统一放进来就是2 不然就是1
            k_size (_type_): _description_: 卷积核的大小 3*3的卷积核 表示为 9
            delta (_type_): _description_ :要插值到的经纬度的间隔 ENS10中是0.5
            rootpath (_type_): _description_: 如果经纬度点最近的k个各网点索引没有生成，生成后需要保存的路径，如果已经生成那么直接在这个文件夹下读取
            dggrid_level (_type_): _description_：二十面体的层数
            p (int, optional): _description_. Defaults to 0.：是否使用dropout
        """        
        # 一般输出的outchannel 是1 因此可以再二十面体上将inchannel 转为 outchannel个通道 然后再使用分组卷积转为了
        super(idg2erp_decoder, self).__init__()
        # # 首先使用球卷积转化为output个通道 ,与普通卷积唯一的区别就是padding的方式不同，这里没有体现到
        # self.sphere_conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=1)
        self.trans_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k_size),groups=in_channels)  # #再转化为平面图像
        self.drop_out = nn.Dropout(p=p) if p > eps else nn.Identity()
        # self.act = nn.ReLU()
        self.table  =get_ico2erp_table(delta=delta,rootpath=rootpath,dggs_type=dggird_type,dggrid_level=dggrid_level,knei=k_size)
        # print('self.table',self.table.head(10))
        seqnumkmin_ndarray = np.stack(self.table['seqnumkmin'].values, axis=0)
        seqnumkmin_tensor = torch.tensor(seqnumkmin_ndarray - 1)
        self.register_buffer('seqnumkmin', seqnumkmin_tensor)
        self.MaxJ = int(360/delta) 

    def forward(self, x):
        # x: [batch_size*10, channel,  H, W]  球面卷积的输入通道 应该是 (b q) c h w
        # 首先使用sphere_conv 把通道数降低
        # x = self.sphere_conv(x)
        x = rearrange(x, '(b q ) c  h w -> b c (q h w)', q=10)  #  转换是否正确 有待确认=》 test_rearrange 确认可以恢复原状
        # x_expand : [batch_size, channel, 10*4**L, 8]  10*4**L => 二十面体格网 8 =>每个格网点最近的8个经纬度点
        x_expand = x[:, :, self.seqnumkmin]
        # print('self.seqnumkmin',self.seqnumkmin.shape)
        # print('x_expand',x_expand.shape) 
        x_expand = self.drop_out(x_expand)
        out = self.trans_conv(x_expand).squeeze(-1)
        # 转化为二十面体十个面的数据 [batch_size, channel, 10*4**L] => [batch_size, channel, 10, 2**L, 2**L]
        out = rearrange(out, 'b c (h w ) -> b c  h w', w=self.MaxJ)  # 转化为了 经纬度格网的输出结果
        return out

class decoder_v2(nn.Module):
    """
        将最终的二十面体格网数据转化为经纬度格网数据作为输出进行精度验证
    """
    def __init__(self, in_channels, kernel_size, delta,dggird_type,rootpath,dggrid_level,p=0,useAttention=True) -> None:
        # 一般输出的outchannel 是1 因此可以再二十面体上将inchannel 转为 outchannel个通道 然后再使用分组卷积转为了
        super(decoder_v2, self).__init__()
        # # 首先使用球卷积转化为output个通道 ,与普通卷积唯一的区别就是padding的方式不同，这里没有体现到
        # self.sphere_conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=1)
        self.trans_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size),groups=in_channels)  # #再转化为平面图像
        self.drop_out = nn.Dropout(p=p) if p > eps else nn.Identity()
        self.indices = get_ico2erp_knn(delta=delta,rootpath=rootpath,dggs_type=dggird_type,dggrid_level=dggrid_level,knei=kernel_size)
        self.seqnumkmin = torch.tensor(self.indices - 1)
        # self.register_buffer('seqnumkmin', seqnumkmin_tensor)
        self.MaxJ = int(360/delta) 
        self.att = SpatialAttentionModule_v1(kernel_size=kernel_size) if useAttention else nn.Identity()
        
    def forward(self, x):
        # x: [batch_size*10, channel,  H, W]  球面卷积的输入通道 应该是 (b q) c h w
        # 首先使用sphere_conv 把通道数降低
        # x = self.sphere_conv(x)
        x = rearrange(x, '(b q ) c  h w -> b c (q h w)', q=10)  #  转换是否正确 有待确认=》 test_rearrange 确认可以恢复原状
        # x_expand : [batch_size, channel, 10*4**L, 8]  10*4**L => 二十面体格网 8 =>每个格网点最近的8个经纬度点
        x_expand = x[:, :, self.seqnumkmin]
        # print('self.seqnumkmin',self.seqnumkmin.shape)
        # print('x_expand',x_expand.shape) 
        x_expand = self.drop_out(x_expand)
        x_expand = self.att(x_expand) # 把每个经纬度点转化为9个格网点的数据进行注意力加权
        out = self.trans_conv(x_expand).squeeze(-1)
        # 转化为二十面体十个面的数据 [batch_size, channel, 10*4**L] => [batch_size, channel, 10, 2**L, 2**L]
        out = rearrange(out, 'b c (h w ) -> b c  h w', w=self.MaxJ)  # 转化为了 经纬度格网的输出结果
        return out
        
#  测试rearrange '(b  q ) c  h w -> b c (q h w)', q=10是否符合预期
def test_rearrange():
    # 按照特定的顺序 可以回复原状 比如都是 b 在前 q在后 如果反了结果就不一样了
    h=w=2
    q=2
    c =2
    b =2
    array = np.array(range(h*w*q*c*b))
    out = rearrange(array, ' (b c q h w ) ->  b c q h w',b=b,c=c ,q=q, h=h, w=w)
    print(out)

    out_1 = rearrange(out, ' b c q h w -> (b q) c h w', q=q, h=h, w=w,c=c)
    print(out_1)

    out_2 = rearrange(out_1, '(b q ) c h w->  b c q h w', q=q, h=h, w=w,c=c)
    print(out_2)
    out_3 = rearrange(out_1, '(q b ) c h w->  b c q h w', q=q, h=h, w=w,c=c)
    print(out_3)


def test_erp2igd_encoder():
    from tqdm import tqdm
    # dd dasdf
    k_size = 8
    in_channel = 128
    out_channel = 256
    dggrid_level = 6
    # dgg_kn_erp = '/home/dls/data/openmmlab/letter2/ico_baye_cnn/index_table/equ_2.0_k8l4_ISEA4D_4_0.0_90.0.csv'
    dgg_kn_erp = '/home/dls/data/openmmlab/letter2/ico_baye_cnn/index_table/ISEA4D_6_0.0_90.0_k8l6_equ_0.5.pkl'
    # GraphCast: Learning skillful medium-range global weather forecasting 中 使用的就是第6层
    p = 0.5
    encoder = erp2igd_encoder(in_channel=in_channel, k_size=k_size, dgg_kn_erp=dgg_kn_erp, dggrid_level=dggrid_level,
                              p=p).cuda()
    batch = 10
    MaxJ = int(360 / 0.5)
    MaxI = int(180 / 0.5) + 1
    x = torch.randn(batch, in_channel, MaxI, MaxJ).cuda()
    for _ in tqdm(range(1000)):
        out = encoder(x)
    print('out', out.shape)


def test_idg2erp_decoder():
    # dgg_kn_erp =  '/home/dls/data/openmmlab/letter2/FULLER4D_4_0.0_90.0_k8_equ_lon40_lat60.pkl'
    erp_kn_dgg = '/home/dls/data/openmmlab/letter2/ico_baye_cnn/index_table/equ_0.5_k8l6_ISEA4D_6_0.0_90.0.pkl'
    k_size = 8
    dggrid_level = 6
    h = w = 2 ** dggrid_level
    MaxJ = int(360 / 0.5)
    MaxI = int(180 / 0.5) + 1
    in_channel = 128
    out_channel = 1
    decoder = idg2erp_decoder(in_channel, out_channel, k_size, erp_kn_dgg, MaxJ).cuda()
    batch = 10
    x = torch.randn(batch, in_channel, 10, h, w).cuda()
    # 把第三个纬度和第一个纬度合并
    x_sphere = rearrange(x, 'b c q h w -> (b q) c h w')
    for _ in tqdm(range(10000)):
        out = decoder(x_sphere)
    print('out', out.shape)


def test_seqnum2qij():
    # 测试 seqnum2qij 时是否符合预期 erp2igd_encoder forward 最后一步时
    array = np.array(range(2560))
    out = rearrange(array, ' (q h w ) ->  q h w', q=10, h=16, w=16)
    print(out)


if __name__ == "__main__":
    # test_linear()
    # test_erp2igd_encoder()
    # test_erp2igd_encoder()
    # test_idg2erp_decoder()
    # test_rearrange()
    pass
    # test_seqnum2qij()