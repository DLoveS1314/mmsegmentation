# dataset settings
dataset_type = 'ENS10Dataset'
# mean = [33.2922, 33.6510, 31.0398] 
# std = [59.9131, 59.8729, 55.4354]
# bgr_mean = mean[::-1]
# bgr_std = std['std'][::-1]
# train_pipeline = [dict(type='spherePackClsInputs')]

test_pipeline = [# dict(type='RandomCrop', crop_size=224, padding=0),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type= 'PackIcoInputs')]

train_pipeline=[       
        dict(type= 'PackIcoInputs')
 ]
 
data_root = '/home/dls/data/openmmlab/letter2/mmsegmentation/SCNN_TOOL/data/t2m/ann'
train_batch = 118
test_batch = 256
num_workers =8
data_root = ''
train_dataloader =  dict(
        dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        pipeline=train_pipeline,
        test_mode=False
        ),
        # collate_fn=dict(type='default_collate'),
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_size= train_batch,
        num_workers=num_workers
    )
val_dataloader =  dict(
        dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        pipeline=test_pipeline,
         test_mode=True ),
        # collate_fn=dict(type='default_collate'),
        sampler=dict(type='DefaultSampler', shuffle=False),
        batch_size=test_batch,
        num_workers=num_workers
    )
test_dataloader =  dict(
        dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        pipeline=test_pipeline,
         test_mode=True,
        ),
        collate_fn=dict(type='default_collate'),
        sampler=dict(type='DefaultSampler', shuffle=False),
        batch_size=test_batch,
        num_workers=num_workers
    )
# val_evaluator = dict(type='Accuracy', topk=(1,5))
test_evaluator = [dict(type='Accuracy'),dict(type='SingleLabelMetric')]
val_evaluator =test_evaluator
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=3)
val_cfg = dict()
test_cfg = dict() ##可以没有 但是不能为空！！！