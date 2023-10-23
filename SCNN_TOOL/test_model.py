import torch
from mmengine.runner import Runner
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os,sys
import os.path as osp
sys.path.append(os.getcwd())
from torchsummary import summary
import torchvision
# from mmengine.config import Config, DictAction
from mmengine.logging import print_log
# from mmengine.runner import Runner
# from mmengine import DefaultScope
from mmseg.registry import RUNNERS,MODELS,DATASETS
from mmseg.utils import register_all_modules
from SCNN_IDG_ENS10 import IcoPad
from mmengine.runner import Runner
# model settings
from mmcv.cnn import build_padding_layer 
from SCNN_IDG_ENS10 import IcoConvModule
from SCNN_IDG_ENS10 import  ENS10Dataset
from torch.utils.data import  DataLoader
from SCNN_IDG_ENS10 import encoder_v2
from SCNN_IDG_ENS10 import IcoDataPreProcessor
from SCNN_IDG_ENS10 import IcoFCNHead
from mmseg.structures import SegDataSample 
from mmengine.structures import PixelData

# 注册所有模块 openmmlab系列 已经不再显式调用 这个了 而是继承在了mmengine中进行懒加载 
# 只要在config中设置好 default_scope 为mmseg等 下游库  然后把函数卸载 mmseg对应的文件夹中（比如 model的backbone里）
#  然后调用runner即可
# runner里关键程序   感觉下面这些代码之后 也可以不使用register_all_modules了
# # if default_scope is not None:
#         default_scope = DefaultScope.get_instance(  # type: ignore
#         self._experiment_name,
#             scope_name=default_scope)
#     self.default_scope = default_scope
register_all_modules() 


def test_icopadding_func():
    padding =2
    # 注册完成的IcoPad 可以直接使用build_padding_layer
    model_dict = dict(
        type='IcoPad',
        dilation = 2,
    )
    model = build_padding_layer(model_dict, padding)
    # model = MODELS.build( model_dict)
    print(model)
    x =torch.randn(1*10,3, 16,16)
    print(model(x).shape)

def test_upIcoBasicConvBlock():
    MODELS.register_module('IcoPad', module=IcoPad)
    # 测试build_upsample_layer中是否需要传入padding_mode参数
    upsample_cfg =dict(type='IcoInterpConv')
    in_channels = 3
    skip_channels = 3
    out_channels = 3
    num_convs = 2
    stride = 1
    with_cp = False
    conv_cfg = None
    norm_cfg = dict(type='BN')
    act_cfg = dict(type='ReLU')
    padding_mode = dict(type= 'IcoPad') 
    # model = build_upsample_layer(
    #             cfg=upsample_cfg,
    #             in_channels=in_channels,
    #             out_channels=skip_channels,
    #             with_cp=with_cp,
    #             norm_cfg=norm_cfg,
    #             act_cfg=act_cfg)
    conv = IcoConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            padding_mode = padding_mode)
    print(conv)

def test_icounet():
    num_stages=3#包含最初的一层把inchannel 转化为base_channel的层
    strides=(1, 1, 1 )
    enc_num_convs=(2, 2, 2 )
    dec_num_convs=(2, 2) 
    downsamples=(True, True)
    enc_dilations=(1, 1, 1 )
    dec_dilations=(1, 1 )
    base_channels = 32
    padding_mode= dict( type='IcoPad')
    # erp2igd_dict =dict(k_size=8,
    #                     dgg_kn_erp ='/home/dls/data/openmmlab/letter2/ico_baye_cnn/index_table/ISEA4D_6_0.0_90.0_k8l6_equ_0.5.pkl',
    #                     dggrid_level = 6,
    #                                )
    model_dict = dict(
        type='IcoUnet',
        in_channels = 22,
        num_stages = num_stages,
        strides = strides,
        enc_num_convs = enc_num_convs,
        dec_num_convs = dec_num_convs,
        downsamples = downsamples,
        enc_dilations = enc_dilations,
        dec_dilations = dec_dilations,
        base_channels = base_channels,
        padding_mode =padding_mode,
        # erp2igd_dict = erp2igd_dict
    )
    # print(MODELS)
    model = MODELS.build(model_dict).cuda()
    x = torch.randn(32,22,361,720).cuda()
    out = model(x)
    # print(model)
    # from tqdm import tqdm
    # for i in tqdm(range(10)):
    #     x = torch.randn(32,22,361,720).cuda()
    #     out = model(x)

def testdataset():
    ann_file = '/home/dls/data/openmmlab/letter2/mmsegmentation/SCNN_TOOL/data/t2m/ann/train.json'
    data_root = '/home/dls/data/openmmlab/letter2/mmsegmentation/SCNN_TOOL/data/t2m/ann'
    test_mode =True 
    pipeline = [dict(type= 'PackIcoInputs')]
    dataset_dict = dict(type = 'ENS10Dataset',data_root = data_root,ann_file = 'train.json',test_mode=test_mode,pipeline=pipeline)
    dataset = DATASETS.build(dataset_dict)
    print(len(dataset))
    # print(dataset[0])
    packed_results= dataset[0]
    
    print(packed_results.keys())
    
def test_Dataset_old():
    # 不使用mmseg的dataloader格式了 步骤复杂 直接给runner传真正的dataloader
    dataset_dict =  dict(type = 'ENS10GridDataset',data_path  = '/media/dls/WeatherData/ENS10/meanstd/',target_var = 't2m',return_time=True,dataset_type='train')
    dataset = DATASETS.build(dataset_dict)
    print(len(dataset))
    packed_results= dataset[0]
    # print(packed_results.keys())
# main 函数
def test_dataloader():
  
        pass
def test_all():
    
    train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset =  dict(type = 'ENS10GridDataset',
                     data_path  = '/media/dls/WeatherData/ENS10/meanstd/',
                     target_var = 't2m',
                     return_time=True,
                     dataset_type='train')
    ) 
    data_preprocessor = dict(
    type='IcoDataPreProcessor',
    pad_val=0,
    seg_pad_val=0)
    
    data_path = '/media/dls/WeatherData/ENS10/meanstd/'
    target_var= 't2m'
    batch_size = 1
    # trainloader = DataLoader(ENS10GridDataset(data_path= data_path,
                                                #   target_var=  target_var,
                                                #   dataset_type='train'), batch_size, shuffle=True, num_workers=4, pin_memory=True)
    num_stages=3#包含最初的一层把inchannel 转化为base_channel的层
    strides=(1, 1, 1 )
    enc_num_convs=(2, 2, 2 )
    dec_num_convs=(2, 2) 
    downsamples=(True, True)
    enc_dilations=(1, 1, 1 )
    dec_dilations=(1, 1 )
    base_channels = 32
    padding_mode= dict( type='IcoPad')
    strides=(1, 1, 1  )
    in_channels = 22
    base_channels = 32
    norm_cfg=dict(type='BN') 
    act_cfg=dict(type='ReLU') 
    upsample_cfg=dict(type='IcoInterpConv') 
    
    model_dict = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
            type='IcoUnet',
            in_channels = in_channels,
            base_channels = base_channels,
            num_stages = num_stages,
            strides = strides,
            enc_num_convs = enc_num_convs,
            dec_num_convs = dec_num_convs,
            downsamples = downsamples,
            enc_dilations = enc_dilations,
            dec_dilations = dec_dilations,
            padding_mode =padding_mode,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg
        ),
        decode_head=dict(
            type='FCNHead' ,
            in_channels=base_channels,
            channels=base_channels,
            num_classes = 1,
        )
    )
    x = torch.randn(1*10,3,64,64)
    # print(MODELS)
    model = MODELS.build(model_dict)
    print('test_all model ',model.data_preprocessor)
    model.cuda()
    trainloader=Runner.build_dataloader(train_dataloader)
    print('test_all trainloader ',trainloader)
    for batch_idx, data_batch in enumerate(trainloader):
        # 把databatch转化为模型需要的格式 包含 inputs 和datasampler 同时把数据搬运到gpu上
        data = model.data_preprocessor(data_batch, True)
        print(data.keys())
        if isinstance(data, dict):
            print('test_all in',data['inputs'].device )
            results = model.extract_feat( data['inputs'] )
        elif isinstance(data, (list, tuple)):
            results = model.extract_feat(*data )
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        for i in range(len(data['inputs'])):
            print(f'test_all out {i}',results[i].shape )
        break

def test_decode_head():
    
    in_channels=32
    
    channels=16
    
    num_convs=1
    kernel_size=3
    concat_input=False
    dilation=1
    num_classes=None
    separate =True #均值和方差是否分开卷积
    lam_wei = 0.5
    loss_decode  = [ dict(type='SmoothL1Loss',lam_w= lam_wei),dict(type='CrpsGaussianLoss',lam_w= 1-lam_wei)]
    
    erp2igd_dict = dict(k_size=kernel_size*kernel_size, delta=0.5,rootpath='/home/dls/data/openmmlab/letter2/mmsegmentation/index_table',dggrid_level=5,p=0,dggird_type='FULLER4D') 
    decodehead   = IcoFCNHead(in_channels=in_channels,channels=channels,num_convs=num_convs,
                           kernel_size=kernel_size,concat_input=concat_input,dilation=dilation,
                           num_classes=num_classes,separate=separate,erp2igd_dict=erp2igd_dict,loss_decode=loss_decode)  
    
    batch =10
    batch_Samples =[]
    for i in range(batch):
        sample = SegDataSample()
        gt_sem_seg_data = dict(data= torch.randn(1,361,720))
        sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        batch_Samples.append(sample)
      
        
    x = [torch.randn(batch,in_channels,64,64)]
    
    out = decodehead.loss(x,batch_Samples,None)
    print('test_decode_head out', out)
    # out = decodehead(x)
 
    # print('test_decode_head out',out.shape)
def cal_flops():
    class mymodel(torch.nn.Module):
        def __init__(self,inchannel):
            super().__init__()
            self.conv = torch.nn.Conv2d(inchannel, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3) ) 
        def forward(self,x):
            return self.conv(x)
    class mymodel_UNFLOD(torch.nn.Module):
        def __init__(self,inchannel):
            super().__init__()
            self.conv = torch.nn.Conv2d(inchannel, 32, kernel_size=(1, 49), stride=(1, 1), padding=0 ) 
        def forward(self,x):
            return self.conv(x)
    from thop import profile


    inchannel = 3
    H =W = 64 #菱形的高和宽
    model = mymodel(inchannel=inchannel)
        # Model
    print('==> Building model..')

    dummy_input = torch.randn(1, inchannel, 361, 720)
    out = model(dummy_input)
    print('out model',out.shape)
    flops, params = profile(model, (dummy_input,))
    
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
    
    # model  = mymodel().cuda()
    model1 =   mymodel(inchannel=inchannel)
    print('==> Building model1..')
    dummy_input1 = torch.randn(10,inchannel, H, W)
    out1 = model1(dummy_input1)
    print('out model1',out1.shape)
    flops, params = profile(model1, (dummy_input1,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
    model2 =   mymodel_UNFLOD(inchannel=inchannel)
    print('==> Building model1..')
    dummy_input2 = torch.randn(1,inchannel, 10*H*W, 49)
    out2 = model2(dummy_input2)
    print('out model2',out2.shape)
    flops, params = profile(model2, (dummy_input2,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # summary(model , input_size=(3, 361, 720))
    # summary(model1 , input_size=(30, 64, 64))
    # summary(model1 , input_size=(30, 128, 128))
    
    
    
def test_encoder_v2():
    in_channels=3
    out_channels=32 
    kernel_size=(7, 7)
    bias: bool = True  
    dggs_type = "FULLER4D" 
    dggrid_level = 6 
    grid_mode = "bilinear" 
    model_encoder =  encoder_v2(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,bias=bias,dggs_type=dggs_type,dggrid_level=dggrid_level,grid_mode=grid_mode)
    loss_decode =[dict(type ='SmoothL1Loss' ),dict(type='CrpsGaussianLoss')]
    
    x = torch.randn(1,3,90,180)
    out = model_encoder(x)
    print('test_encoder_v2 out',out.shape)
if __name__ == '__main__':
    # test_upIcoBasicConvBlock()
    # test_icounet()
    # test_encoder_v2()
# runner = Runner(    model=model_dict,
#     work_dir='./work_dir' )
# runner.build_model(model_dict)
    # test_Dataset()
    # test_all()
    # cal_flops()
    # test_decode_head()
    testdataset()