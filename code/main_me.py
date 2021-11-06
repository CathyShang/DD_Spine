## 打开dicom 并将其转换为 png
# import json
# import time
# import os
# import torch
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# import os
# import SimpleITK as sitk
# import pydicom
# import numpy as np
# import cv2
# from tqdm import tqdm
#
#
# from code.core.disease.data_loader import DisDataLoader
# from code.core.disease.evaluation import Evaluator
# from code.core.disease.model import DiseaseModelBase
# from code.core.key_point import KeyPointModel, NullLoss
# from code.core.structure import construct_studies
#
# from code.nn_tools import torch_utils
# import argparse
# import matplotlib.pyplot as plt  # 绘制训练过程的loss和accuracy曲线

train_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train150'#'./data/DatasetA/train/lumbar_train150'
train_anno_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train150_annotation.json'#'./data/DatasetA/train/lumbar_train150_annotation.json'
vali_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train51/'#'./data/DatasetA/train/lumbar_train51/'
vali_anno_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train51_annotation.json'#'./data/DatasetA/train/lumbar_train51_annotation.json'

test_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_testB50/'#'./data/DatasetB/test/lumbar_testB50/'
test_anno_path = '/home/wangzy/Documents/Shangzq/spinal_detection/data/testB50_series_map.json'#'./data/DatasetB/test/testB50_series_map.json'

# def dicom2png(dicm_path, list_tag):
#     reader = sitk.ImageFileReader()
#     reader.LoadPrivateTagsOn()
#     reader.SetFileName(dicm_path)
#     reader.ReadImageInformation()
#     return [reader.GetMetaData(t) for t in list_tag]
#       return image

## dicom_2_png
import os
import SimpleITK
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

def is_dicom_file(filename):

    #判断某文件是否是dicom格式的文件
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def load_patient(src_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = pydicom.read_file(src_dir + '/' + s)
            slices.append(instance)

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array


if __name__ == '__main__':
	#dicom文件目录
    dicom_dir = '/home/wangzy/Documents/Shangzq/spinal_detection/data/lumbar_train51/study2'#'dcm/'
    # 读取dicom文件的元数据(dicom tags)
    slices = load_patient(dicom_dir)
    print('The number of dicom files : ', len(slices))
    # 提取dicom文件中的像素值
    image = get_pixels_hu_by_simpleitk(dicom_dir)
    for i in tqdm(range(image.shape[0])):
    	#输出png文件目录
        img_path = "/home/wangzy/Documents/Shangzq/spinal_detection/data/dcm_2_png/img_" + str(i).rjust(4, '0') + "_i.png"
        org_img =image[i]*20
        # 保存图像数组
        cv2.imwrite(img_path, org_img)




