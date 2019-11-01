from lvis.lvis import LVIS
import numpy as np
import pickle
import pdb
import os
import torch
from pycocotools.coco import COCO

def get_split():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 10:
            bin10.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)


def ana_param():

    cate2insnum_file = './data/lvis/cate2insnum.pkl'
    if False: # os.path.exists(cate2insnum_file):
        with open(cate2insnum_file, 'rb') as f:
            cate2insnum = pickle.load(f)

    else:
        train_ann_file = './data/lvis/lvis_v0.5_train.json'
        val_ann_file = './data/lvis/lvis_v0.5_val.json'

        lvis_train = LVIS(train_ann_file)
        lvis_val = LVIS(val_ann_file)
        train_catsinfo = lvis_train.cats
        val_catsinfo = lvis_val.cats

        train_cat2ins = [v['instance_count'] for k, v in train_catsinfo.items()]
        train_cat2ins = [0] + train_cat2ins
        val_cat2ins = [v['instance_count'] for k, v in val_catsinfo.items()]
        val_cat2ins = [0] + val_cat2ins

        cate2insnum = {}
        cate2insnum['train'] = np.array(train_cat2ins, dtype=np.int)
        cate2insnum['val'] = np.array(val_cat2ins, dtype=np.int)

        with open(cate2insnum_file, 'wb') as fout:
            pickle.dump(cate2insnum, fout)

    checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_lvis/latest.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']
    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_bias = param['bbox_head.fc_cls.bias'].numpy()

    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    savelist = [cls_fc_weight_norm,
                cls_fc_bias]

    with open('./data/lvis/r50_param_ana.pkl', 'wb') as fout:
        pickle.dump(savelist, fout)


def ana_coco_param():

    train_ann_file = './data/coco/annotations/instances_train2017.json'
    val_ann_file = './data/coco/annotations/instances_val2017.json'

    coco_train = COCO(train_ann_file)
    coco_val = COCO(val_ann_file)

    cat2insnum_train = np.zeros((91,), dtype=np.int)
    for k, v in coco_train.imgToAnns.items():
        for term in v:
            cat2insnum_train[term['category_id']] += 1

    cat2insnum_val = np.zeros((91,), dtype=np.int)
    for k, v in coco_val.imgToAnns.items():
        for term in v:
            cat2insnum_val[term['category_id']] += 1

    cat2insnum_train = cat2insnum_train[np.where(cat2insnum_train > 0)[0]]
    cat2insnum_val = cat2insnum_val[np.where(cat2insnum_val > 0)[0]]
    cat2insnum_train = np.hstack((np.zeros((1,), dtype=np.int), cat2insnum_train))
    cat2insnum_val = np.hstack((np.zeros((1,), dtype=np.int), cat2insnum_val))

    checkpoint_file = './data/download_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']
    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_bias = param['bbox_head.fc_cls.bias'].numpy()

    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    savedict = {'train_ins': cat2insnum_train,
                'val_ins': cat2insnum_val,
                'weight': cls_fc_weight_norm,
                'bias': cls_fc_bias}

    with open('./localdata/cocoparam.pkl', 'wb') as fout:
        pickle.dump(savedict, fout)

def load_checkpoint():

    # checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_lvis/latest.pth'
    checkpoint_file = 'data/download_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']

    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_bias = param['bbox_head.fc_cls.bias'].numpy()
    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    reg_weight = param['bbox_head.fc_reg.weight'].numpy()
    reg_bias = param['bbox_head.fc_reg.bias'].numpy()
    reg_weight_norm = np.linalg.norm(reg_weight, axis=1)
    reg_weight_norm = reg_weight_norm.reshape((81, 4)).mean(axis=1)
    reg_bias = reg_bias.reshape((81, 4)).mean(axis=1)


    savedict = {'cls_weight': cls_fc_weight_norm,
                'cls_bias': cls_fc_bias,
                'reg_weight': reg_weight_norm,
                'reg_bias': reg_bias}

    with open('./localdata/r50_weight_coco.pkl', 'wb') as fout:
        pickle.dump(savedict, fout)

def get_mask():
    train_ann_file = './data/lvis/lvis_v0.5_train.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    mask = np.zeros((1231, ), dtype=np.int)

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 100:
            mask[cid] = 1

    mask_torch = torch.from_numpy(mask)
    torch.save(mask_torch, './data/lvis/mask.pt')


def trymapping():

    mask = torch.load('./data/lvis/mask.pt')

    ids = np.array([0,0,2,0,1], dtype=np.int)
    cls_ids = torch.from_numpy(ids)

    for i in range(5):
        new_ids = mask[cls_ids]
        print(new_ids)

if __name__ == '__main__':

    # ana_param()
    # get_mask()
    # trymapping()
    # ana_coco_param()
    load_checkpoint()