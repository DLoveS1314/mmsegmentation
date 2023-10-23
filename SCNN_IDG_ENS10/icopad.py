# 生成IDG_pdading 非矩阵版本
import numpy as np
import pandas as pd
import torch.nn.functional as F 
import torch
import torch.nn as nn
import math 
import os
from einops import rearrange 
from mmseg.registry import MODELS
# from .maketablefromfile import creatadjtable
# NEI_TABLE_ROOT = '/home/dls/data/openmmlab/letter2/ico_baye_cnn/neitable'
# 注释基本删除 详情注释请看 icopaddingfunc.py  IcoPadding 属于 get_idg_padding 改编而来
@MODELS.register_module()
class IcoPad(nn.Module):
    def __init__(self, padding,nei_table_root=os.path.join(os.getcwd(),'neitable'),dggtype='ISEA4D'):
        # 更改参数pad_s为 padding 名字与torch官方的名字相同 方便build_padding_layer调用
        super(IcoPad, self).__init__()
        self.pad_s = padding
        self.nei_table_root = nei_table_root
        self.nei_directions={##padding的值不能超过当前等级的最大行
                    'miniup':['left','down'],'minidown':['left','left'],'maxiup':['right','right'],'maxidown':['right','up'],
                    'minjup':['down','down'],'minjdown':['down','left'],'maxjup':['up','right'],'maxjdown':['up','up']
                    }
        
        self.relationall = ['mini','maxi','minj','maxj']
        self.directions = ('up','left','down','right')
        self.quad =10
        self.dtype = torch.int64
        self.dggtype = dggtype

        self.current_neitable = None # 在使用get_padding_data的时候才能赋值 因为level是不确定的
        tablepath_7 = os.path.join(self.nei_table_root,f'{self.dggtype}_7_nei.csv')
        # 判断临近文件是否存在，如果不存在则使用creatad:jtable创建文件 1-7级的临近文件 这样方式生成的是9临近 因为第一篇论文的缘故无法生成四临近
        # 我在calnei.h中 写了一个 outnei函数 专门生成4临近 就不直接自动化生成了
        # if not os.path.exists(tablepath_7):
        #     os.makedirs(self.nei_table_root,exist_ok=True)
        #     for level in range(1,8):
        #         creatadjtable(dggs_type=dggtype, resolution=level, save_dir=nei_table_root)

        assert os.path.exists(tablepath_7) ,f'{tablepath_7} 不存在,请使用calnei.h中的outnei函数进行生成'
    def get_nei_direction(self,seqnums,direction ):
        """_summary_ 根据编码串和确定的邻域方向和邻域表求邻域
        Args:
            seqnums(torch.tensor): 一系列编码串,一般这些编码串是需要padding 出于同一行或者同一列的编码  
            nei_direction (_type_): _description_ :需要选择的邻域方向  directions中的一种 每个编码对应一个direction
            neitable (_type_): _description_:某一层级的邻域表 datafarme
        Return:
            返回的是 numpy array矩阵 其实返回的是self.neitable的视图
        """    
        rowindex =  seqnums - 1 #索引号取巧了，按理说应该是np.where() 才是，见code2index(maketablefile文件里) 但是seqnum代码与行号差1 所以这里就直接用了 
        # 计算 rowindex的最大值
        # rowindex_max = torch.max(rowindex)
        # pandas 的index class 的功能
        columnsindex = self.current_neitable.columns.get_indexer(direction)
        value = self.current_neitable.values[rowindex,columnsindex]
        return value
    def get_nei(self,codes,relation,level ,padsize):
        """_summary_ get_padding_data负责调用 输出对应编码 需要输出的padding索引
            每个边分别处理的函数 不能四个边处理 因为在角点处的顶点 既是最大行又是最大列 
        Args:
            codes (_type_):  tensor数据 codes[:,0]存储的面片编码 codes[:,1]存储的是行编码 codes[:,2]存储的是列编码
            relation (_type_):  mini maxi minj maxj中的一种
            level (_type_):  需要padding的level等级 从forward可以获得
            padsize (_type_):   需要padding的次数
        Return:
            返回的是列表 列表中的元素是一维的numpy矩阵。元素按照从小到大的序号排列存储的是codes的一阶邻近、二阶临近
        """
        # 临近表是算好的 否则每次都去循环求不但要改代码 还需要循环计算的时间很耗时

        # 根据 quad 编码，确定是up还是down 在全局变量nei_directions中 面片1~5是up 6~10是down
        quadlist = np.where(codes[:,0] > 5, 'down', 'up')
        relation_key =np.char.add(relation, quadlist) #每个编码对应一个relation的值
        seqnums_list =[]
        seqnums = self.getseqnum(codes[:,0],codes[:,1],codes[:,2],level)#初始层的编码串 tensor
        # 根据relation_key 从nei_directions中获取对应的方向 由于relation_key是一个nd.array 因此需要循环取值
        directions =  np.array([self.nei_directions.get(key) for key in relation_key])
        # print('directions',directions)
        for i in range(padsize):
            flag = 0 if i == 0 else 1
            direction = directions[:,flag]
            seqnums = self.get_nei_direction(seqnums,direction)
            seqnums_list.append(seqnums)
        return seqnums_list
    def get_padding_data(self,seqnum_codes ,relation, level,data,pad_size):
        """_summary_  
        负责调用self.get_nei 并把self.get_nei输出的seqnum code索引数据转化为实际的需要填充的真是数据
        此函数只能处理相同relation的 padding_data 
        Args:
            seqnum_codes (_type_): torch.tensor
            relation (_type_): self.relationall = ['mini','maxi','minj','maxj'] 中的一种  每一种分开进行处理
            level (_type_): 当前需要padding的二十面体格网等级
            data (_type_): forward中的x
            pad_size (_type_):  需要padding的大小
        Returns:
            _type_: padding_data tensor 返回需要填充的真实数据
        """
        tablepath =  os.path.join(self.nei_table_root,f'{self.dggtype}_{level}_nei.csv')
        self.current_neitable = self.getneitable(tablepath)
        nei_seqnums_list = self.get_nei(seqnum_codes,relation,level,pad_size)
        # qij_array = self.getqij(np.array(nei_seqnums_list),level)
        qij_array = self.getqij(torch.as_tensor(np.array(nei_seqnums_list),dtype=torch.int64),level)

        padding_data = data[:,:,self.getq_dim(qij_array[0].flatten()),qij_array[1].flatten(),qij_array[2].flatten()].clone()
        # return padding_data 
        return padding_data ,nei_seqnums_list ## 用于测试   
    

    def merge_padding_data(self,x):
        """
        Args: 用于draw_exp.py的测试
        :param x:
        :return:
        """
        # x 的输入纬度为 batch_quad ,channel , height, width 第一纬度是batch和quad的乘积 
        # 返回也需要这个纬度 
        batch_quad ,channel , quda,height, width = x.shape
        # level 可以根据输入的高度计算出来 不需要再输入参数 但是要确保池化都是2的倍数
        current_level = self.get_level(height)
        # x =  rearrange(x, '(b q) c h w -> b c q h w',q=self.quad)
        assert height == width, "height and width must be equal"
        print('x',x.shape)
        MaxI = MaxJ = height
        assert self.pad_s < MaxI, f'padding的值{self.pad_s}不能超过当前等级的最大行号{MaxI}'
        new_data = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s), 'constant', 0) 
        current_level = self.get_level(height)
   
        # NOTE torch 不支持uint32 
        # 按照原始图像的行列号计算需要pading的 q i j（对应最大最小行、列）
        seqnum_codes_mini = torch.as_tensor([(q_nei, 0, j) for q_nei in range(1, 11) for j in range(MaxJ)],dtype=self.dtype )
        seqnum_codes_maxi = torch.as_tensor([(q_nei, MaxI-1, j) for q_nei in range(1, 11) for j in range(MaxJ)],dtype=self.dtype )
        seqnum_codes_minj = torch.as_tensor([(q_nei, i, 0) for q_nei in range(1, 11) for i in range(MaxI)],dtype= self.dtype )
        seqnum_codes_maxj = torch.as_tensor([(q_nei, i, MaxJ-1) for q_nei in range(1, 11) for i in range(MaxI)],dtype=self.dtype )

        # 计算padding的数据 indexi0new生成的索引顺序要与padding_data_mini的顺序相同 先 i/j 后q
        # padding_data 的长度为  2**level *10* pad_size  
        # 存储顺序为 （q,i,j） =>(q1,i0,j0),(q1,i0,j1),...,(q1,i0,jn),(q1,i1,j0),...,(q0,in,jn),(q1,i0,j0),...,(q1,in,jn),...,(q10,in,jn)
        #  n = 2**level -1 
        # 以padding_data_mini为例 ，i恒为0 因此存储顺序为 (q1,0,0),(q1,0,1),...,(q1,0,n), (q2,0,0),(q2,0,1),...,(q2,0,n),...,(q10,0,0),(q10,0,1),...,(q10,0,n)
        # 一次padiing存储完成之后 再接着存储下一个padding的 因此padding_data 的长度为  2**level *10* pad_size 
        
        # nei_seqnums_list 长度为 [l0,l1, ... ln] 其中 n = pad_size-1 l的长度为 2**level *10
        # 按照顺序l0 代表第一次padding的seqnum编码 l1代表第二次padding的seqnum编码 以此类推
        
        padding_data_mini, nei_seqnums_list_mini = self.get_padding_data(seqnum_codes_mini, 'mini', current_level, x, pad_size=self.pad_s)
        padding_data_maxi, nei_seqnums_list_mmxi = self.get_padding_data(seqnum_codes_maxi, 'maxi', current_level, x, pad_size=self.pad_s)
        padding_data_minj, nei_seqnums_list_minj  = self.get_padding_data(seqnum_codes_minj, 'minj', current_level, x, pad_size=self.pad_s)
        padding_data_maxj, nei_seqnums_list_maxj = self.get_padding_data(seqnum_codes_maxj, 'maxj', current_level, x, pad_size=self.pad_s)
        # print('padding_data_mini',padding_data_mini.shape)
        # print('nei_seqnums_list_mini',nei_seqnums_list_mini)
     
        
        # print('seqnum_codes_mini',seqnum_codes_mini)
        
        padding_data = torch.cat([padding_data_mini.flatten(),padding_data_maxi.flatten(),padding_data_minj.flatten(),padding_data_maxj.flatten()],dim=0)
        # print('padding_data',padding_data.shape)
        # 维度为 4**pad_size , (10*2**level)
        nei_seqnums = torch.as_tensor(np.array([*nei_seqnums_list_mini,*nei_seqnums_list_mmxi,*nei_seqnums_list_minj,*nei_seqnums_list_maxj]))
        # print('nei_seqnums',nei_seqnums.shape)
        # （）括号里面的顺序 是由规律的不能乱写 按照前面维度优先的原则来进行 比如(b padsize)维度 [[] [] | [] [] | [] [] |  [] [] |  [] [] ]
        # 按照数据的生成方式 应该做成 b,paasize维度 对于  padszie =2 时 前两组为 nei_seqnums_list_mini 以此类推 最后两组为nei_seqnums_list_maxj
        # 所以比较适合切分为 b, padsize  也就是b在前面 padsize在后面 后面的同理 主要要看数据生成方式
        nei_seqnums= rearrange(nei_seqnums, '(b padsize) (  quad maxij ) -> b padsize quad  maxij ',quad=self.quad,padsize =self.pad_s,b=4)
        # nei_seqnums= rearrange(nei_seqnums, 'b ( quad  maxij ) -> b    quad maxij   ',quad=self.quad  )
        
        # print('nei_seqnums',nei_seqnums )
        # print('nei_seqnums',nei_seqnums.shape )
        return nei_seqnums  #返回维度为 [4个方向, padsize, quad, 最大行/列数（maxij）]
        

    def forward(self, x ):
        if self.pad_s == 0:
            return x
        else:
            # x 的输入纬度为 batch_quad ,channel , height, width 第一纬度是batch和quad的乘积 
            # 返回也需要这个纬度 
            assert x.ndim ==4, f"x must be 4 dim  now x {x.shape}"
            batch_quad ,channel , height, width = x.shape
            # level 可以根据输入的高度计算出来 不需要再输入参数 但是要确保池化都是2的倍数
            current_level = self.get_level(height)
            x =  rearrange(x, '(b q) c h w -> b c q h w',q=self.quad)
            assert height == width, "height and width must be equal"
            MaxI = MaxJ = height
            assert self.pad_s < MaxI, f'padding的值{self.pad_s}不能超过当前等级的最大行号{MaxI}'
            new_data = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s), 'constant', 0) 
            
            current_level = self.get_level(height)

            # NOTE torch 不支持uint32 
            # 按照原始图像的行列号计算需要pading的 q i j（对应最大最小行、列）
            seqnum_codes_mini = torch.as_tensor([(q_nei, 0, j) for q_nei in range(1, 11) for j in range(MaxJ)],dtype=self.dtype )
            seqnum_codes_maxi = torch.as_tensor([(q_nei, MaxI-1, j) for q_nei in range(1, 11) for j in range(MaxJ)],dtype=self.dtype )
            seqnum_codes_minj = torch.as_tensor([(q_nei, i, 0) for q_nei in range(1, 11) for i in range(MaxI)],dtype= self.dtype )
            seqnum_codes_maxj = torch.as_tensor([(q_nei, i, MaxJ-1) for q_nei in range(1, 11) for i in range(MaxI)],dtype=self.dtype )

            # 利用padding后的图像对应的行列号 计算需要插入padding数据的q i j 
            indexi0new = torch.as_tensor([(q_nei, i, j+self.pad_s) for i in range(self.pad_s-1, -1, -1) for q_nei in range(1, 11) for j in range(MaxJ)] ,dtype=self.dtype)

            indeximaxnew = torch.as_tensor([(q_nei, MaxI+self.pad_s+i, j+self.pad_s) for i in range(self.pad_s) for q_nei in range(1, 11) for j in range(MaxJ)] ,dtype=self.dtype)
            indexj0new = torch.as_tensor([(q_nei, i+self.pad_s, j) for j in range(self.pad_s-1, -1, -1) for q_nei in range(1, 11) for i in range(MaxI)] ,dtype=self.dtype)
            indexjmaxnew = torch.as_tensor([(q_nei, i+self.pad_s, MaxJ+self.pad_s+j) for j in range(self.pad_s) for q_nei in range(1, 11) for i in range(MaxI)],dtype=self.dtype)
            # print("===========================")
            # print('ico_pad forward x',x.shape)
            # print('self.pad_s',self.pad_s)
            # print('MaxJ',MaxJ)
            # print('current_level',current_level)
            # print(' indexi0new[:, 0] ', indexi0new .shape)
            # print('self.getq_dim(indexi0new[:, 0])',self.getq_dim(indexi0new[:, 0]).shape)
            # print('seqnum_codes_mini',np.max(seqnum_codes_mini.numpy()))

            # print("===========================")
            # 计算padding的数据 indexi0new生成的索引顺序要与padding_data_mini的顺序相同
            padding_data_mini, _ = self.get_padding_data(seqnum_codes_mini, 'mini', current_level, x, pad_size=self.pad_s)
            padding_data_maxi, _ = self.get_padding_data(seqnum_codes_maxi, 'maxi', current_level, x, pad_size=self.pad_s)
            padding_data_minj, _ = self.get_padding_data(seqnum_codes_minj, 'minj', current_level, x, pad_size=self.pad_s)
            padding_data_maxj, _ = self.get_padding_data(seqnum_codes_maxj, 'maxj', current_level, x, pad_size=self.pad_s)
         


            new_data[:, :, self.getq_dim(indexi0new[:, 0]) , indexi0new[:, 1], indexi0new[:, 2]] = padding_data_mini 
            new_data[:, :, self.getq_dim(indeximaxnew[:, 0]), indeximaxnew[:, 1], indeximaxnew[:, 2]] = padding_data_maxi
            new_data[:, :, self.getq_dim(indexj0new[:, 0]), indexj0new[:, 1], indexj0new[:, 2]] = padding_data_minj
            new_data[:, :, self.getq_dim(indexjmaxnew[:, 0]), indexjmaxnew[:, 1], indexjmaxnew[:, 2]] = padding_data_maxj
            new_data =  rearrange(new_data, 'b c q h w -> (b q) c h w',q=self.quad)
            # print('new_data',new_data[:,:,self.getq_dim(indexi0new[:,0]),indexi0new[:,1],indexi0new[:,2]].shape)
            return new_data

    # 后面是一些辅助函数
    def get_level(self,H):
        level = math.log2(H)
        assert level == int(level), f"H {H} and W {H} must be 2^n"
        return int(level)
     # 注意存在两个q求get_nei时要使用1~10的q编码 但是实际存储时 q的维度时0-9
    def getq_dim(self,q):
        return  torch.as_tensor(q-1,dtype=self.dtype)
    def getq_quad(self,q):
        return q+1
    def getneitable(self,tablepath):
        """ 生成临近表
        Args:
            tablepath (_type_):  临近表的路径
        Returns:
            _type_: 返回临近表
        """

        neitable = pd.read_csv(tablepath,header=None)
        neitable.columns = ['ori','up','left','down','right' ]
        return neitable
    # 获得每个面有多少格元
    def getoffsetPerQuad(self,level):
        # factor = math.pow(2,level)
        factor = 2**level
        offsetPerQuad =factor*factor
        return offsetPerQuad
    def getseqnum(self,q,i,j,level):
        """根据 q i j 计算seqnum编码
        Args:
            q (torch.tensor): dggrid中定义的quadnum 对于菱形属于1~11 注意与存储矩阵的q维度区分开 q_dim 属于0~10
            i (torch.tensor): 行号
            j (torch.tensor): 列号
            level (int ): 当前层级
        Returns:
            torch.tensor: 返回seqnmum编码
        """
        offsetPerQuad =  self.getoffsetPerQuad(level)
        numj = self.getmaxj(level)+1
        seqnum = (q-1)*offsetPerQuad + i*numj + j + 1
        return seqnum
    def getmaxj(self,level):
        # maxj = math.pow(2,level)-1
        maxj = 2**level-1
        return  maxj
    # 得到某一层级所有格元数
    def gettotalnum(self,level):
        totalnum = 10*self.getoffsetPerQuad(level)
        return totalnum
    def getqij(self,seqnum,level):
        """# 根据seqnum的到qij
        Args:
            seqnum (torch.tensor):  seqnum从1开始  tensor
            level (int):  指定层级
        Returns:
            _type_: 返回1q i j 每一个都是一个 tensor 数组
        """
        seqnum = seqnum-1 # seqnum默认从1开始 但是计算的时候从0开始比较好计算
        # assert seqnum < gettotalnum(level), "seqnum must be less than totalnum"
        offsetPerQuad = int(self.getoffsetPerQuad(level))
        # https://pytorch.org/docs/stable/generated/torch.div.html#torch.div
        # rounding_mode 三个参数 默认为None 还有"trunc" 以及floor
        #  trunc是向0取整  floor属于向下取整 在负数时会有区别其余时候没有
        q = torch.div(seqnum,offsetPerQuad,rounding_mode='floor')+1
        # q = seqnum//int(offsetPerQuad) + 1 #0和11号面是特殊面 正常的菱形面从1开始 UserWarning: __floordiv__ is deprecated
        seqnumout =  seqnum -((q - 1) * self.getoffsetPerQuad(level) )
        seqnumout =  seqnumout.type_as(seqnum)

        # seqnum =seqnum-1 # seqnum默认从1开始 但是计算的时候从0开始比较好计算
        numj = self.getmaxj(level)+1
        # i =  seqnumout // numj UserWarning: __floordiv__ is deprecated
        i = torch.div(seqnumout,numj,rounding_mode='floor')
        j =  seqnumout % numj
        return (q,i,j)

# def test_icopad_1():
#     # 测试padding是否正确
#     table_root= '/home/dls/data/openmmlab/letter2/neitable'
#     pad_size = 1
#     icopd = IcoPadding(pad_size,table_root)
#     batch = 1
#     channel = 1
#     quadnum = 10
#     level = 1
#     H = W = int(math.pow(2,level))
#     x = torch.randn( batch, channel,quadnum,H,W )
#     new_data_index = icopd.test_padding_forward(x,level)
#     new_data = icopd(x,level)
#     print(new_data_index[0,0,0]  )
#     data_split = x[0,0]
#     index_mapping = np.array(list(np.ndindex(data_split.shape)))
#     seqnum_index = icopd.getseqnum(icopd.getq_quad(index_mapping[:,0]) ,index_mapping[:,1],index_mapping[:,2],level)
#     print('seqnum_index',seqnum_index)
#     print(new_data[0,0,0]  )
#     print(x[0,0].flatten() )
# def test_icopad_2():
#     # 测试cuda 等卷积网络
#     table_root= '/home/dls/data/openmmlab/letter2/neitable'
#     pad_size = 1
#     batch = 1
#     channel = 1
#     quadnum = 10
#     level = 4
#     icopd = IcoPadding(pad_size,table_root).cuda()
#     model1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1).cuda()
#     H = W = int(math.pow(2,level))
#     x = torch.randn( batch, channel,quadnum,H,W ).cuda()
#     x_pad = icopd(x,level)
#     outs =[]
#     for q in range(quadnum):
#         input = x_pad[:,:,q,:,:]
#         print('input',input.device)
#         out_data = model1(input)
#         print('out_data',out_data.device)
#
#         outs.append(out_data)
#     outs = torch.stack(outs,dim=2)
#     print('out',outs.shape)
#
#
# def test_icopad_3():
#     # 测试cuda 等卷积网络 改变了icopad的输入 测试一下
#     batch = 20
#     channel = 1
#     quadnum = 10
#     level = 7
#     kernel_size = 5
#     pad_size = kernel_size//2
#     icopd = IcoPad(pad_size ).cuda()
#     model1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=kernel_size,stride=1).cuda()
#     H = W = int(math.pow(2,level))
#     x = torch.randn( batch*quadnum, channel,H,W ).cuda()
#     print('x',x.shape)
#
#     x_pad = icopd(x )
#     print('x_pad',x_pad.shape)
#
#     out = model1(x_pad)
#     print('out',out.shape)
# if __name__ == '__main__':
#   test_icopad_3()
#

 
