HH=`pwd`
#
# cd $HH

conda create -n mmdet python=3.7 -y
# conda init bash
# bash
# conda activate mmdettry
export PATH=/opt/conda/envs/mmdet/bin:$PATH
echo $PATH
which python

pip install cython
pip install numpy
pip install torch
pip install torchvision
pip install pycocotools
pip install mmcv
pip install matplotlib
pip install terminaltables
pip install lvis
# cd lvis-api/
# python setup.py develop

cd $HH
python setup.py develop

# cd $HH
OMP_NUM_THREADS=3 ./tools/dist_train.sh configs/faster_rcnn_r50_fpn_1x_lvis.py 8
