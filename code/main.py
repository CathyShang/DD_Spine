"""
--Submit version

Code for lumbar competetion of Tianchi.
Baseline from https://github.com/wolaituodiban/spinal_detection_baseline.git

by @mruniquejj
2020 08 14
"""

import json
import time
import os
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from code.core.disease.data_loader import DisDataLoader
from code.core.disease.evaluation import Evaluator
from code.core.disease.model import DiseaseModelBase
from code.core.key_point import KeyPointModel, NullLoss
from code.core.structure import construct_studies

from code.nn_tools import torch_utils
import argparse
import matplotlib.pyplot as plt  # 绘制训练过程的loss和accuracy曲线

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ========== setting paths ======================

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='DDT lumbar code')

    parser.add_argument("--mode", default="traintest", choices=['traintest', 'train', 'test'],
                        help="Running mode.\n traintest - train and test.\n train - only train and save the model.\n test - only load the model and test.")
    return parser.parse_args()


train_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train150'#'./data/DatasetA/train/lumbar_train150'
train_anno_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train150_annotation.json'#'./data/DatasetA/train/lumbar_train150_annotation.json'
vali_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train51/'#'./data/DatasetA/train/lumbar_train51/'
vali_anno_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train51_annotation.json'#'./data/DatasetA/train/lumbar_train51_annotation.json'

test_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_testB50/'#'./data/DatasetB/test/lumbar_testB50/'
test_anno_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/testB50_series_map.json'#'./data/DatasetB/test/testB50_series_map.json'


if __name__ == '__main__':
    args = arg_parse()
    mode = args.mode            # train and test

    start_time = time.time()
    backbone = resnet_fpn_backbone('resnet34', True) #  resnet101  resnet50
    kp_model = KeyPointModel(backbone)
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))

    if mode == 'test':
        dis_model.load_state_dict(torch.load('/home/wangzy/Documents/Shangzq/DDT_code_roundA/data/External/models/DDT_60200908210756.dis_model'))  # load model  0804_kpBCELoss_60ite.dis_model

    dis_model.cuda()
    # print(dis_model)

    epoch = 60 #40
    # ========= train ==============
    if 'train' in mode:
        print('construct TRAINING studies')
        train_studies, train_annotation, train_counter = construct_studies(
            train_path, train_anno_path, multiprocessing=True)
        print('construct VALIDATION studies')
        valid_studies, valid_annotation, valid_counter = construct_studies(
            vali_path,vali_anno_path , multiprocessing=True)

        train_dataloader = DisDataLoader(
            train_studies, train_annotation, batch_size=8, num_workers=15, num_rep=10, prob_rotate=0.5, max_angel=20,
            sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.sagittal_size, k_nearest=0
        )
    
        valid_evaluator = Evaluator(
            dis_model, valid_studies, vali_anno_path, num_rep=20, max_dist=6,
        )
    
        step_per_batch = len(train_dataloader)
        print('step_per_batch:', step_per_batch)
        optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
        max_step = epoch * step_per_batch
        print('Max_step:', max_step)
    
        fit_result= torch_utils.fit(
            dis_model,
            train_data=train_dataloader,
            valid_data=None,
            optimizer=optimizer,
            max_step=max_step,
            loss=NullLoss(),
            metrics=[valid_evaluator.metric],
            is_higher_better=True,
            evaluate_per_steps=step_per_batch,
            evaluate_fn=valid_evaluator,
        ) # , Loss_record
        # print(fit_result.shape())
        # print(Loss_record.shape())
        # plt.plot(Loss_record, 60, '.-')
        # plt.xlabel('Test loss vs. epoches')
        # plt.ylabel('Test loss')
        # plt.show()
        # plt.savefig("loss.jpg")
        # print(fit_result.shape)
        torch.save(dis_model.cpu().state_dict(), 'data/External/models/DDT_'+str(epoch)+time.strftime('%y%m%d%H%M%S')+'.dis_model')
        print('task completed, {} seconds used'.format(time.time() - start_time))
    # ========= test ==============
    if 'test' in mode:
        print('construct Testing studies from:',test_path)
        if test_anno_path:
            test_anno = json.load(open(test_anno_path, 'r'))

        test_t2_series = {}
        for anno in test_anno:
            test_t2_series[anno['studyUid']] = anno['seriesUid']

        testA_studies = construct_studies(test_path, multiprocessing=True)
        result = []
        for study in testA_studies.values():
            study.t2_sagittal_uid = test_t2_series[study.study_uid]
            # frame = study.t2_sagittal_middle_frame
            # frame.image.save('./pic/'+study.study_uid + '.jpg')
            result.append(dis_model.eval()(study, True))

        with open('submit/DDT_'+str(epoch)+time.strftime('%y%m%d%H%M%S')+'.json', 'w') as file:
            json.dump(result, file)
        print('task completed, {} seconds used'.format(time.time() - start_time))
