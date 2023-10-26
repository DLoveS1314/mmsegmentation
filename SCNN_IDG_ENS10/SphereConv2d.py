import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .GridGenerator import  GridGenerator_icopoint

  
  
class SpatialAttentionModule(nn.Module):
    def __init__(self,kernel_size=3,stride=1,padding=1):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out * x + x #是否应该加shortcut
        return out
class SphereConv2d_icopoint(nn.Module):
  """
  kernel_size: (H, W)
  forward with icopoint 而不是erp投影点
  给这个加个attention 机制
  对于6层的菱形格网
  1、p => b,c,L,1 
  2、grid_sample =>b,c,L*Kh,Kw
  3、conv2d   (out_c, in_c, Kh, Kw)  stirde =1 padding=0 => b,out_c,L,1
  4、reshape L =>10,2^6,2^6
  5、class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
  """
# 使用深度可分离卷积
  def __init__(self, in_channels: int, out_channels: int, 
               kernel_size=(3, 3),
               stride=(3,1) ,padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
               dggs_type = "FULLER4D",
               res = 6,
               grid_mode = "bilinear" ,
               usesa = True
               ):
    super(SphereConv2d_icopoint, self).__init__()
    
    self.grid_shape = None
    self.grid = None
    
    self.groups = in_channels
    self.kernel_size = kernel_size
    self.depthConv = nn.Conv2d(in_channels, in_channels, kernel_size ,groups=groups, bias=bias,stride=stride) ##相当于插值了
    self.pointConv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias, padding_mode) #相当于提高二十面体的维度
    
    self.sa = SpatialAttentionModule() if usesa else nn.Identity()
    self.grid_mode = grid_mode
    # 新添加的球面格网的数据参数
    self.dggs_type = dggs_type
    self.res = res
  def genSamplingPattern_ico(self, h, w):
    gridGenerator = GridGenerator_icopoint(dggs_type=self.dggs_type,res=self.res, imgHeight = h,imgWidth=w,kernel_size=self.kernel_size )
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()
    # print('LonLatSamplingPattern.shape',LonLatSamplingPattern[:, :, :, 0].shape)
    
    # print('LonLatSamplingPattern.shape',LonLatSamplingPattern[:, :, :, 0])
    # generate grid to use `F.grid_sample` 用于归一化
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern 不管x 输入是什么 都可以直接扩展 另外B这个参数 在训练/测试的时候 最后一一个batch可能会变换 只能在这里动态求取 所以干脆 H W也在这里求取 
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern_ico(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2) => (B, L*Kh, Kw, 2)
      grid.requires_grad = False
    # print('grid.shape',grid.shape)
    
    out = F.grid_sample(x, grid, align_corners=False, mode=self.grid_mode)  # (B, in_c, H*Kh, W*Kw) =>(B, L*Kh, Kw, 2)
    # print('x.shape',x.shape)
    # self.weight -> (out_c, in_c, Kh, Kw)
    # x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size,padding=0,groups=self.groups) # padding=0
    out = self.sa(out)
    # print('sa x.shape',x.shape)
    
    out=self.depthConv(out)
    # print('depthConv x.shape',x.shape)
    
    out=self.pointConv(out)
    # print('pointConv x.shape',x.shape)

    return out  # (B, out_c, H/stride_h, W/stride_w)
