# model settings

norm_cfg = dict(type='BN')

act_cfg=dict(type='Swish')

conv_cfg  = dict(type ='Conv2d')


data_preprocessor = dict(
    type='IcoDataPreProcessor' 
 )
in_channels =22 
base_channels = 16
backbone=dict(
    type='IcoUnet',
    in_channels = in_channels,
    base_channels = base_channels,
    num_stages = 3,
    strides = (1,1,1),
    enc_num_convs = (2, 2, 2 ),
    dec_num_convs = (2, 2) ,
    downsamples = (True, True),
    enc_dilations = (1, 1, 1 ),
    dec_dilations = (1, 1 ),
    padding_mode = dict( type='IcoPad'),
    norm_cfg=norm_cfg,
    act_cfg=act_cfg,
    upsample_cfg=dict(type='IcoInterpConv') 
)

### 设置decode的参数 
decode_kernel_size = 3 
channels = base_channels//2
dilation=1
num_classes=None
separate =True #均值和方差是否分开卷积
num_convs =1 #分开卷积 则这数必须大于1
concat_input=False #为了节省计算量 这个参数直接设置为false
drop_rate = 0
erp2igd_dict = dict(
    k_size=decode_kernel_size*decode_kernel_size, 
    delta=0.5,
    rootpath='/home/dls/data/openmmlab/letter2/mmsegmentation/index_table',
    dggrid_level=7,p=0,dggird_type='FULLER4D') 


# 设置 loss
lam_wei = 0.5 #第一个loss的权重 ，另一个为 1-lam_wei
loss_decode  = [ dict(type='SmoothL1Loss',lam_w= lam_wei),dict(type='CrpsGaussianLoss',lam_w= 1-lam_wei)]
    
#设置decode_head
decode_head=dict(
            type='IcoFCNHead' ,
            in_channels=base_channels,
            channels=base_channels,
            num_classes = num_classes,
            num_convs = num_convs,
            concat_input = concat_input,
            dropout_ratio=drop_rate,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_cfg = conv_cfg,
            separate=separate,
            loss_decode=loss_decode,
            erp2igd_dict=erp2igd_dict,
            in_index=-1,            
        )
# self.parse_losses model的基类里 这个函数负责对所有的loss进行处理 主要是进行平均和相加
# def loss_name(self): 只要带有loss_的都会被加入到backward graph中 多个loss的话会进行简单的相加 在self.parse_losses里完成相加
#     """Loss Name.

#     This function must be implemented and will return the name of this
#     loss function. This name will be used to combine different loss items
#     by simple sum operation. In addition, if you want this loss item to be
#     included into the backward graph, `loss_` must be the prefix of the
#     name.

#     Returns:
#         str: The name of this loss item.
#     """
#     return self._loss_name
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    decode_head=decode_head,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
