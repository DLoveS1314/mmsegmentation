import os,sys
sys.path.append(os.getcwd())

import random
# from config import config
from tqdm import tqdm
import shutil
import json
from SCNN_IDG_ENS10.Dataset_old1 import  ENS10GridDataset


import numpy as np
from mmseg.structures import SegDataSample
# 摘抄自prepare_dggrid_data.ipynb 他的那个里面是错误的
def creatann2json(data_root,label_root,ann_path, variables):
    """_summary_
        准备创建的数据集存储结构如下
        ├── data
            |—— annotations
            |   │── t2m
            |   │   ├── test.json
            |   │   ├── train.json
            |   │   ├── val.json
            |   │── t850
            |   │   ├── test.json
            |   │   ├── train.json
            |   │   ├── val.json
            |   │── z500
            |   │   ├── test.json
            |   │   ├── train.json
            |   │   ├── val.json
            │   ├── t2m(每个数据集存一个文件夹)
            │   │   ├── img_dir
            │   │   │   ├── t2m_1988.npy
            │   │   │   │
            │   │   ├── label_dir
            │   │   │   ├── t2mlabel_1988.npy
            │   │   │   │ 
            │   ├── t850
            │   │   ├── img_dir
            │   │   │   ├── t850_1988.npy
            │   │   │   │
            │   │   ├── label_dir
            │   │   │   ├── t850label_1988.npy
            │   │   │   │ 
            │   ├── z500
            │   │   ├── img_dir
            │   │   │   ├── z500_1988.npy
            │   │   │   │
            │   │   ├── label_dir
            │   │   │   ├── z500label_1988.npy
            │   │   │   │ 
    Args:
        data_root (_type_): 数据所在根目录 文件命名形式为为 预测变量_年份.npy 如 t2m_1988.npy
        label_root (_type_): 标签所在根目录 文件命名形式为为 预测变量_年份.npy 如 t2mlabel_1988.npy
        ann_path (_type_): _description_
    """    
    random.seed(888)
    # ratio,ratio1 分别代表 训练集（trian）和验证级(val)的比率，测试集在两者中间（test）
    ratio ='0.8'
    # 按照数据规则 定义变变量
    classes = [] 
    data_list_train =[]
    data_list_test =[]
    data_list_val=[]
    picnum=0
    trainlen=tstlen=vallen=0
    fliename = os.listdir(data_root)
    labelname = os.listdir(label_root)
    print('总的文件个数{}'.format(len(fliename)))
    print('总的标签个数{}'.format(len(labelname)))
    data_list_train = []
    data_list_test =[]
    data_list_val = []
    label_list_train =[]
    label_list_test =[]
    label_list_val = []
    # file_indexes = ['0101', '0104', '0108', '0111', '0115', '0118', '0122', '0125', '0129', '0201', '0205', '0208', '0212', '0215', '0219', '0222', '0226', '0301', '0305', '0308', '0312', '0315', '0319', '0322', '0326', '0329', '0402', '0405', '0409', '0412', '0416', '0419', '0423', '0426', '0430', '0503', '0507', '0510', '0514', '0517', '0521', '0524', '0528', '0531', '0604', '0607', '0611', '0614', '0618', '0621', '0625', '0628', '0702', '0705', '0709', '0712', '0716', '0719', '0723', '0726', '0730', '0802', '0806', '0809', '0813', '0816', '0820', '0823', '0827', '0830', '0903', '0906', '0910', '0913', '0917', '0920', '0924', '0927', '1001', '1004', '1008', '1011', '1015', '1018', '1022', '1025', '1029', '1101', '1105', '1108', '1112', '1115', '1119', '1122', '1126', '1129', '1203', '1206', '1210', '1213', '1217', '1220', '1224', '1227', '1231']

    # for year in range(1998,2018):
    #     for day in file_indexes:
    #         time = str(year)+day
    #         if year <2015:
    #                 data_list_train += [x for x in fliename if str(time) in x]
    #                 label_list_train += [x for x in labelname if str(time) in x]
    #         elif year ==2015:#val
    #             data_list_val += [x for x in fliename if str(time) in x]
    #             label_list_val += [x for x in labelname if str(time) in x]
    #         else:#test
    #             data_list_test += [x for x in fliename if str(time) in x]
    #             label_list_test += [x for x in labelname if str(time) in x]
    
    for file in fliename:
        time = file.split('_')[-1].split('.')[0]
        year = int(time[:4])
        if year <2015:
                data_list_train.append(file)
                label_list_train.append(file.replace('img','label').replace('_','label_'))
        elif year ==2015:#val
            data_list_val.append(file)
            label_list_val.append(file.replace('img','label').replace('_','label_'))
        else:#test
            data_list_test.append(file)
            label_list_test.append(file.replace('img','label').replace('_','label_'))
    # 把data_list_train 和label_list_train 放到train.txt
    # 创建 train.txt 文件并将数据写入
    with open('train.txt', 'w') as file:
        for img_path, label_path in zip(data_list_train, label_list_train):
            # 将图像路径和标签路径写入文件，用空格分隔
            file.write(f'{img_path} {label_path}\n')
            
    # 把data_list_test 和label_list_test 放到test.txt
    with open('test.txt', 'w') as file:
        for img_path, label_path in zip(data_list_test, label_list_test):
            # 将图像路径和标签路径写入文件，用空格分隔
            file.write(f'{img_path} {label_path}\n')

    # 把data_list_val 和label_list_val 放到val.txt
    with  open('val.txt', 'w') as file:
        for img_path, label_path in zip(data_list_val, label_list_val):
            # 将图像路径和标签路径写入文件，用空格分隔
            file.write(f'{img_path} {label_path}\n')
    print('训练集的文件个数{}'.format(len(data_list_train)))
    print('训练集的标签个数{}'.format(len(label_list_train)))   
    print('测试集的文件个数{}'.format(len(data_list_test)))
    print('测试集的标签个数{}'.format(len(label_list_test)))
    print('验证集的文件个数{}'.format(len(data_list_val)))
    print('验证集的标签个数{}'.format(len(label_list_val)))
    print('总的文件个数{}'.format(len(data_list_train)+len(data_list_test)+len(data_list_val)))
    print('总的标签个数{}'.format(len(label_list_train)+len(label_list_test)+len(label_list_val)))
    # return
    train_dict=[]
    test_dict=[]
    val_dict=[]
    # 把数据和标签的路径拼接成字典 数据的key为img_path 标签的key为gt_seg_map
    for data,label in zip(data_list_train,label_list_train):
        data_list_instance = {'img_path':os.path.join(data_root,data),"gt_seg_map": os.path.join(label_root,label) }
        train_dict.append(data_list_instance)
    for data,label in zip(data_list_test,label_list_test):
        data_list_instance = {'img_path':os.path.join(data_root,data),"gt_seg_map": os.path.join(label_root,label) }
        test_dict.append(data_list_instance)
    for data,label in zip(data_list_val,label_list_val):
        data_list_instance = {'img_path':os.path.join(data_root,data),"gt_seg_map": os.path.join(label_root,label) }
        val_dict.append(data_list_instance)
 
    metainfo ={'variables':variables}

    # random.shuffle(train_dict)
    # random.shuffle(test_dict)
    # random.shuffle(val_dict)

    
    with open(os.path.join(ann_path,'train.json'),'w+') as ftrain:
        dataall = {'metainfo':metainfo,'data_list':train_dict}
        json.dump(dataall,ftrain,sort_keys=True, indent=4, separators=(',', ':'))##格式化输出，使得输出能够每块数据加个空格 

    with open(os.path.join(ann_path,'test.json'),'w+') as ftest:
        dataall = {'metainfo':metainfo,'data_list':test_dict}
        json.dump(dataall,ftest,sort_keys=True, indent=4, separators=(',', ':'))
    
    with open(os.path.join(ann_path,'val.json'),'w+') as fval:
        dataall = {'metainfo':metainfo,'data_list':val_dict}
        json.dump(dataall,fval,sort_keys=True, indent=4, separators=(',', ':'))



# root ='/home/dls/data/openmmlab/DGGRID_CNN_NET/UCMerced_LandUse/Images'
# root =root = '/home/dls/data/openmmlab/mmclassification/data/UCMerced_LandUse/data'
# ann_save_path ='/home/dls/data/openmmlab/mmclassification/data/UCMerced_LandUse/kflod_ann'
# creatann2jsonforkflod(root,ann_save_path)
def creat_json():
    """_summary_
        ├── data
            |—— annotations
            |   │── t2m
            |   │   ├── test.json
            |   │   ├── train.json
            |   │   ├── val.json
            |   │── t850
            |   │   ├── test.json
            |   │   ├── train.json
            |   │   ├── val.json
            |   │── z500
            |   │   ├── test.json
            |   │   ├── train.json
            |   │   ├── val.json
            │   ├── t2m(每个数据集存一个文件夹)
            │   │   ├── img_dir
            │   │   │   ├── t2m_1988.npy
            │   │   │   │
            │   │   ├── label_dir
            │   │   │   ├── t2mlabel_1988.npy
            │   │   │   │ 
            │   ├── t850
            │   │   ├── img_dir
            │   │   │   ├── t850_1988.npy
            │   │   │   │
            │   │   ├── label_dir
            │   │   │   ├── t850label_1988.npy
            │   │   │   │ 
            │   ├── z500
            │   │   ├── img_dir
            │   │   │   ├── z500_1988.npy
            │   │   │   │
            │   │   ├── label_dir
            │   │   │   ├── z500label_1988.npy
    """    

    # 按照文件结构 循环创建t2m、t850、z500数据集的json文件
    # root = '/media/dls/WeatherData/ENS10/data'
    root = '/home/dls/Desktop'
    outroot = '/home/dls/data/openmmlab/letter2/mmsegmentation/SCNN_IDG_ENS10/data'
    # for var in ["t2m", "t850", "z500"]:
    for var in ["t2m" ]:
        
        # 数据集根目录
        data_root = os.path.join(root, var,'img')
        # 标签根目录
        label_root = os.path.join(root, var,'label')
        # json文件保存路径
        ann_path  = os.path.join(outroot, var,'ann')
        # 创建文件夹
        if not os.path.exists(ann_path):
            os.makedirs(ann_path)
        # 创建json文件
        if var == "t2m":
            variables = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']
        else:
            variables  = ["Z", "T", "Q", "W", "D", "U", "V"]
        creatann2json(data_root, label_root, ann_path,variables=variables)
        print("{} is done!".format(var))
        print("=====================================")


def getdatafromnc():
    # 从nc文件中提取数据 保存为npy文件 命名的原则是变量_年份.npy 标签的命名是变量label_年份.npy
    # 循环遍历所有的变量和标签
    data_path = '/media/dls/WeatherData/ENS10/meanstd/'
    target_vars = ['t850','z500','t2m' ]
    dataset_type = ['train','val','test']
    out_dir = '/media/dls/WeatherData/ENS10/data'
    for var in target_vars:
        for dat_tpye in dataset_type:
            # 读取数据
            data = ENS10GridDataset(data_path=data_path,target_var = var,return_time=True,dataset_type=dat_tpye,normalized=False)
            bar = tqdm(range(len(data)))
            for i in  bar:
                packed_results = data[i]
                # img = packed_results['inputs']
                data_sample:SegDataSample= packed_results['data_samples']
                label = data_sample.gt_sem_seg.data
                time = data_sample.metainfo['time']
                # 把时间字符串转化为年份+月份+日期的形式 1997-12-02--->19971202
                time = time.split('-')
                time = ''.join(time)
                # bar.set_postfix(time=time)
                # outimgname = os.path.join(out_dir,var,'img',var+'_'+time+'.npy')
                outlabelname = os.path.join(out_dir,var,'label',var+'label_'+time+'.npy')
                # os.makedirs(os.path.dirname(outimgname),exist_ok=True)
                os.makedirs(os.path.dirname(outlabelname),exist_ok=True)
                # 保存数据
                # np.save(outimgname,img)
                np.save(outlabelname,label)
                bar.set_description("Processing {}".format(var+'_'+time+'.npy'))

 
def get_outvar_mean_std():
    data_path = '/media/dls/WeatherData/ENS10/meanstd/'
    target_vars = ['t850','z500','t2m' ]
    dataset_type = ['train','val','test']
    out_dir = '/media/dls/WeatherData/ENS10/data/outvarmeanstd'
    for var in target_vars:
        for dat_tpye in dataset_type:
            # 读取数据
            data = ENS10GridDataset(data_path=data_path,target_var = var,return_time=True,dataset_type=dat_tpye,normalized=False)
            bar = tqdm(range(len(data)))
            for i in  bar:
                packed_results = data[i]
                # img = packed_results['inputs']
  
                mean = packed_results['scale_mean']
                std = packed_results['scale_std']
                time = packed_results['time']
                # 把时间字符串转化为年份+月份+日期的形式 1997-12-02--->19971202
                time = time.split('-')
                time = ''.join(time)
                # bar.set_postfix(time=time)
                # outimgname = os.path.join(out_dir,var,'img',var+'_'+time+'.npy')
                outvar_mean_name = os.path.join(out_dir,var,f'outvarmeanstd',var+'outvarm_'+time+'.npy')
                outvar_std_name = os.path.join(out_dir,var,f'outvarmeanstd',var+'outvars_'+time+'.npy')
                
                # os.makedirs(os.path.dirname(outimgname),exist_ok=True)
                os.makedirs(os.path.dirname(outvar_mean_name),exist_ok=True)
                # 保存数据
                # np.save(outimgname,img)
                np.save(outvar_mean_name,mean)
                np.save(outvar_std_name,std)
                
                bar.set_description(" outvar  {}".format(var+'_'+time+'.npy'))
if __name__ == "__main__":
#    getdatafromnc()
    # creat_json()
    get_outvar_mean_std()