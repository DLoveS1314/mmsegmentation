# dataset settings
dataset_type = 'ENS10Dataset'
# mean = [33.2922, 33.6510, 31.0398] 
# std = [59.9131, 59.8729, 55.4354]
# bgr_mean = mean[::-1]
# bgr_std = std['std'][::-1]
# train_pipeline = [dict(type='spherePackClsInputs')]

test_pipeline = [# dict(type='RandomCrop', crop_size=224, padding=0),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type= 'PackIcoInputs',meta_keys=('out_mean','out_std'))]

train_pipeline=[       
        dict(type= 'PackIcoInputs',meta_keys=('out_mean','out_std'))
 ]
 
train_batch = 16
test_batch = 32
num_workers = 16
data_root = ''#需要继承使用
pin_memory=True
persistent_workers=True
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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
test_dataloader =  dict(
        dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        pipeline=test_pipeline,
         test_mode=True,
        ),
        # collate_fn=dict(type='default_collate'),
        sampler=dict(type='DefaultSampler', shuffle=False),
        batch_size=test_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
# val_evaluator = dict(type='Accuracy', topk=(1,5))
test_evaluator = [dict(type='CRPS_metric') ]
val_evaluator =test_evaluator
