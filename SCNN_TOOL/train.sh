conda activate pytorch2.0
cd /home/dls/data/openmmlab/letter2/mmsegmentation
configpath=/home/dls/data/openmmlab/letter2/mmsegmentation/SCNN_TOOL/config/t2m/1024.py 
python ./SCNN_TOOL/scnn_train.py  ${configpath}  