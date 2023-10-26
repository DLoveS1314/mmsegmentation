_base_ = [
    '../_base_/dataset_config.py',
    '../_base_/default_runtime.py',
    '../_base_/schedule.py',
    '../_base_/model_config.py'
]

# Dataset 
train_batch = 8
test_batch = 16
data_root = '/home/dls/data/openmmlab/letter2/mmsegmentation/SCNN_TOOL/data/t2m/ann'
indices=600 #调试用  一个epoch 运行多少次图片
# train_dataloader = dict(dataset = dict(data_root =data_root,indices= indices ),batch_size=train_batch)
train_dataloader = dict(dataset = dict(data_root =data_root ),batch_size=train_batch)
test_dataloader = dict(dataset = dict(data_root =data_root),batch_size=test_batch)
val_dataloader = dict(dataset = dict(data_root =data_root),batch_size=test_batch)

#model 
# 设置 loss
lam_wei = 1.0 #第一个loss的权重 ，另一个为 1-lam_wei
# loss_decode  = [ dict(type='MSE_VAR',lam_w= lam_wei),dict(type='CrpsGaussianLoss',lam_w= 1-lam_wei)]
loss_decode  = [  dict(type='CrpsGaussianLoss',lam_w= 1.0)]

# 设置 idg2erp_dict

knnnei = 9 #需要继承 不然就报错
dggrid_level =7

idg2erp_dict = dict(
    kernel_size=knnnei , 
    delta=0.5, #经纬度的间隔
    rootpath='/home/dls/data/openmmlab/letter2/mmsegmentation/index_table',
    dggrid_level=dggrid_level,p=0,dggird_type='FULLER4D',useAttention=False) 
 
# 设置backbone

erp2igd_dict = dict( kernel_size=knnnei,
        delta=0.5, #经纬度的间隔
    rootpath='/home/dls/data/openmmlab/letter2/mmsegmentation/index_table',
    dggrid_level=dggrid_level,p=0,dggird_type='FULLER4D',useAttention=False
    ) 

backbone=dict(erp2igd_dict=erp2igd_dict)
decode_head = dict(idg2erp_dict=idg2erp_dict,loss_decode=loss_decode)

# 设置 data_preprocessor
mean   = '/media/dls/WeatherData/ENS10/normalized/ENS10_sfc_mean_mean.npy'
std    = '/media/dls/WeatherData/ENS10/normalized/ENS10_sfc_mean_std.npy'
std_mean = '/media/dls/WeatherData/ENS10/normalized/ENS10_sfc_std_mean.npy'
std_std = '/media/dls/WeatherData/ENS10/normalized/ENS10_sfc_std_std.npy'
data_preprocessor = dict(
    type='IcoDataPreProcessor' ,
    mean=mean,
    std=std,
    std_mean=std_mean,
    std_std=std_std
 )
model = dict( backbone=backbone,
        decode_head = decode_head,
        data_preprocessor=data_preprocessor
)

# 设置学习率
optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01) 
optim_wrapper = dict(
    optimizer =optimizer,
    paramwise_cfg=dict(norm_decay_mult=0.0))
# 加速训练
# optim_wrapper=dict(
#         type='AmpOptimWrapper',
#         # 如果你想要使用 BF16，请取消下面一行的代码注释
#         # dtype='bfloat16',  # 可用值： ('float16', 'bfloat16', None)
#         optimizer=optimizer,
#         paramwise_cfg=dict(norm_decay_mult=0.0)) 
warmup_epochs = 2
max_epochs = 200
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-05,
        by_epoch=True,
        end=warmup_epochs,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs-5,
        eta_min=1e-05,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs)
]
# train, val, test setting

train_cfg = dict(  max_epochs=max_epochs )


# 2.0的模型编译功能
cfg=dict(compile=True)
# dict(
#     optimizer=dict(type='Adam', lr=0.002, weight_decay=0.0001),
#     paramwise_cfg=dict(norm_decay_mult=0.0))

# Hook
custom_hooks = [
    dict(type='EmptyCacheHook', after_epoch=True),
    dict(
        # type='EarlyStoppingHook',
        type = 'MyEarlyStoppingHook',
        monitor='crps/crps',
        rule = 'less',
        patience=15),
    # dict(type = 'EMAHook')
]

# 如果希望指定恢复训练的路径，除了设置 resume=True，还需要设置 load_from 参数。需要注意的是，
# 如果只设置了 load_from 而没有设置 resume=True，则只会加载 checkpoint 中的权重并重新开始训练，
# 而不是接着之前的状态继续训练。
# load_from='/home/dls/data/openmmlab/letter2/mmsegmentation/work_dirs/1024/best_crps_crps_epoch_195.pth' 
resume =False