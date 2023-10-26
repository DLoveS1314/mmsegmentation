# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.002, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.0))
# learning policy
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=20000,
#         by_epoch=False)
# ]
warmup_epochs = 5
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-05,
        by_epoch=True,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=195,
        eta_min=1e-05,
        by_epoch=True,
        begin=5,
        end=200)
]
# train, val, test setting
val_interval = 2
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict() ##可以没有 但是不能为空！！！
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=val_interval,max_keep_ckpts = 2, save_best ='crps/crps',rule = 'less'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
