import random

import SimpleITK as sitk
import numpy as np
import torch
import torchvision.transforms.functional as tf
from torchvision import utils as vutils
import torchvision
from PIL import Image
import cv2
import time
from ..data_utils import resize, rotate, gen_distmap,gen_heatmap, gasuss_noise,my_sagital_crop,my_flip,my_translate
from ..dicom_utils import DICOM_TAG


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))

        return getattr(self, attr_name)

    return _lazy_property


def str2tensor(s: str) -> torch.Tensor:
    """

    :param s: numbers separated by '\\', eg.  '0.71875\\0.71875 '
    :return: 1-D tensor
    """
    return torch.tensor(list(map(float, s.split('\\'))))


def unit_vector(tensor: torch.Tensor, dim=-1):
    norm = (tensor ** 2).sum(dim=dim, keepdim=True).sqrt()
    return tensor / norm


def unit_normal_vector(orientation: torch.Tensor):
    temp1 = orientation[:, [1, 2, 0]]
    temp2 = orientation[:, [2, 0, 1]]
    output = temp1 * temp2[[1, 0]]
    output = output[0] - output[1]
    return unit_vector(output, dim=-1)


class DICOM:
    """
    解析dicom文件
    属性：
        study_uid：检查ID
        series_uid：序列ID
        instance_uid：图像ID
        series_description：序列描述，用于区分T1、T2等
        pixel_spacing: 长度为2的向量，像素的物理距离，单位是毫米
        image_position：长度为3的向量，图像左上角在人坐标系上的坐标，单位是毫米
        image_orientation：2x3的矩阵，第一行表示图像从左到右的方向，第二行表示图像从上到下的方向，单位是毫米？
        unit_normal_vector: 长度为3的向量，图像的单位法向量，单位是毫米？
        image：PIL.Image.Image，图像
    注：人坐标系，规定人体的左边是X轴的方向，从面部指向背部的方向表示y轴的方向，从脚指向头的方向表示z轴的方向
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.error_msg = ''

        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetImageIO('GDCMImageIO')
        reader.SetFileName(file_path)
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            pass

        try:
            self.study_uid = reader.GetMetaData(DICOM_TAG['studyUid'])
        except RuntimeError:
            self.study_uid = ''

        try:
            self.series_uid: str = reader.GetMetaData(DICOM_TAG['seriesUid'])
        except RuntimeError:
            self.series_uid = ''

        try:
            self.instance_uid: str = reader.GetMetaData(DICOM_TAG['instanceUid'])
        except RuntimeError:
            self.instance_uid = ''

        try:
            self.series_description: str = reader.GetMetaData(DICOM_TAG['seriesDescription'])
        except RuntimeError:
            self.series_description = ''

        try:
            self._pixel_spacing = reader.GetMetaData(DICOM_TAG['pixelSpacing'])
        except RuntimeError:
            self._pixel_spacing = None

        try:
            self._image_position = reader.GetMetaData(DICOM_TAG['imagePosition'])
        except RuntimeError:
            self._image_position = None

        try:
            self._image_orientation = reader.GetMetaData(DICOM_TAG['imageOrientation'])
        except RuntimeError:
            self._image_orientation = None

        try:
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image)[0]
            # rescale the range of image array and cast it to np.uint8.
            # though SimpleITK has similar method, it may crash with python 3.7 and ubuntu 20.04
            # and the reason is unknown right now. thus, i choose to write my own cast code.
            array = array.astype(np.float64)
            array = (array - array.min()) * (255 / (array.max() - array.min()))
            array = array.astype(np.uint8)

            # array = cv2.equalizeHist(array)  # 直方图均衡化，作用于整张图片

            self.image: Image.Image = tf.to_pil_image(array)
        except RuntimeError:
            self.image = None

    @lazy_property
    def pixel_spacing(self):
        if self._pixel_spacing is None:
            return torch.full([2, ], fill_value=np.nan)
        else:
            return str2tensor(self._pixel_spacing)

    @lazy_property
    def image_position(self):
        if self._image_position is None:
            return torch.full([3, ], fill_value=np.nan)
        else:
            return str2tensor(self._image_position)

    @lazy_property
    def image_orientation(self):
        if self._image_orientation is None:
            return torch.full([2, 3], fill_value=np.nan)
        else:
            return unit_vector(str2tensor(self._image_orientation).reshape(2, 3))

    @lazy_property
    def unit_normal_vector(self):
        if self.image_orientation is None:
            return torch.full([3, ], fill_value=np.nan)
        else:
            return unit_normal_vector(self.image_orientation)

    @lazy_property
    def t_type(self):
        if 'T1' in self.series_description.upper():
            return 'T1'
        elif 'T2' in self.series_description.upper():
            return 'T2'
        else:
            return None

    @lazy_property
    def plane(self):
        if torch.isnan(self.unit_normal_vector).all():
            return None
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 0., 1.])).abs() > 0.75:
            # 轴状位，水平切开
            return 'transverse'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([1., 0., 0.])).abs() > 0.75:
            # 矢状位，左右切开
            return 'sagittal'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 1., 0.])).abs() > 0.75:
            # 冠状位，前后切开
            return 'coronal'
        else:
            # 不知道
            return None

    @lazy_property
    def mean(self):
        if self.image is None:
            return None
        else:
            return tf.to_tensor(self.image).mean()

    @property
    def size(self):
        """

        :return: width and height
        """
        if self.image is None:
            return None
        else:
            return self.image.size

    def pixel_coord2human_coord(self, coord: torch.Tensor) -> torch.Tensor:
        """
        将图像上的像素坐标转换成人坐标系上的坐标
        :param coord: 像素坐标，Nx2的矩阵或者长度为2的向量
        :return: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        """
        return torch.matmul(coord * self.pixel_spacing, self.image_orientation) + self.image_position

    def point_distance(self, human_coord: torch.Tensor) -> torch.Tensor:
        """
        点到图像平面的距离，单位为毫米
        :param human_coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为N的向量或者标量
        """
        return torch.matmul(human_coord - self.image_position, self.unit_normal_vector).abs()

    def projection(self, human_coord: torch.Tensor) -> torch.Tensor:
        """
        将人坐标系中的点投影到图像上，并输出像素坐标
        :param human_coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return:像素坐标，Nx2的矩阵或者长度为2的向量
        """
        cos = torch.matmul(human_coord - self.image_position, self.image_orientation.transpose(0, 1))
        return (cos / self.pixel_spacing).round()

    def transform(self, pixel_coord: torch.Tensor,
                  size=None, prob_rotate=0, max_angel=0, distmap=False, tensor=True, colorjitter=True ) -> (torch.Tensor, torch.Tensor):
        """
        Data augmaentation and distmap generation.
        返回image tensor和distance map
        :param pixel_coord:[11,2]
        tensor([[115,  41],
        [109,  62],
        [102,  84],
        [ 98, 109],
        [103, 133],
        [115,  30],
        [111,  53],
        [103,  75],
        [ 98,  97],
        [ 99, 122],
        [109, 146]])
        :param size:
        :param colorjitter:是否采用颜色变换(亮度对比度饱和度)
        :param prob_rotate:是否旋转的概率，0-1之间
        :param max_angel:
        :param distmap: 是否返回distmap
        :param tensor: 如果True，那么返回图片的tensor，否则返回Image, to_tensor会将0-255转成0-1，如果mode为L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1
        :return:
        """
        image, pixel_spacing = self.image, self.pixel_spacing

        # color transform
        if colorjitter:
            colorjitter = torchvision.transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2), saturation=(0.5, 2),hue=0)
            image = colorjitter(image)

        # noise
        if np.random.rand() < prob_rotate:
            image = gasuss_noise(image)

        # affine
        if np.random.rand() < prob_rotate:
            chooseflag = np.random.randint(0,3)
            if chooseflag == 0:
                image, pixel_coord = my_sagital_crop(image, pixel_coord)        # crop

                # image = np.array(image)
                # for p in pixel_coord:
                #     image[p[1], p[0]] = 255
                # image = Image.fromarray(image)
                # image.save('./pic/crop' + time.strftime('%y%m%d%H%M%S') + '.jpg')
            # if chooseflag == 1:
            #     image, pixel_coord = my_flip(image, pixel_coord)                # hflip

                # image = np.array(image)
                # for p in pixel_coord:
                #     image[p[1], p[0]] = 255
                # image = Image.fromarray(image)
                # image.save('./pic/flip' + time.strftime('%y%m%d%H%M%S') + '.jpg')

            if chooseflag == 2:
                image, pixel_coord = my_translate(image, pixel_coord)           # translate

                # image = np.array(image)
                # for p in pixel_coord:
                #     image[p[1],p[0]] = 255
                # image=Image.fromarray(image)
                # image.save('./pic/trans' + time.strftime('%y%m%d%H%M%S') + '.jpg')

        # resize
        if size is not None:
            image, pixel_spacing, pixel_coord = resize(size, image, pixel_spacing, pixel_coord)
        # rotate
        if max_angel > 0 and random.random() <= prob_rotate:
            angel = random.randint(-max_angel, max_angel)
            image, pixel_coord = rotate(image, pixel_coord, angel)

        if tensor:
            image = tf.to_tensor(image)                 # to_tensor会将0-255转成0-1，如果mode为L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1

        pixel_coord = pixel_coord.round().long()

        if distmap:
            # distmap = gen_distmap(image, pixel_spacing, pixel_coord)
            distmap = gen_heatmap(image, pixel_coord, sigma = 1)
            # print('distmap.shape:', distmap.shape)
            # for i in range(11):
            #     vutils.save_image(distmap[i].clone().detach(),'./pic/heat'+time.strftime('%y%m%d%H%M%S') + '.jpg')
            return image, pixel_coord, distmap
        else:
            return image, pixel_coord

'''
from matplotlib import pyplot as plt
# 读取图像测试transform
image = Image.open('dist2.jpg')
image.convert('I')
image.show(title='original')
colorjitter = torchvision.transforms.ColorJitter(brightness=(0.5,2), contrast=(0.5,2), saturation=(0.5,2), hue=0)
imagec = colorjitter(image)
imagec.show()

# noise
imagen = gasuss_noise(imagec)
imagen.show()
# crop
coord = torch.tensor([[240,97],[230,320]])
imagep = np.array(imagec).copy()
imagep[coord[0,1]-1:coord[0,1]+2,coord[0,0]-1:coord[0,0]+2] = 255
imagep[coord[1,1]-1:coord[1,1]+2,coord[1,0]-1:coord[1,0]+2] = 255
plt.imshow(imagep,cmap='gray')
plt.show()

imagep2, coordn = my_sagital_crop(imagep,coord)
imagep2[coordn[0,1]-1:coordn[0,1]+2,coordn[0,0]-1:coordn[0,0]+2] = 255
imagep2[coordn[1,1]-1:coordn[1,1]+2,coordn[1,0]-1:coordn[1,0]+2] = 255
imagep2b,_,_ = resize(imagep.shape,Image.fromarray(imagep2),[1,1],torch.tensor(coordn))
plt.imshow(imagep2,cmap='gray')
plt.show()
# hflip
imagef, coordf = my_flip(imagec, coord)
imagef = np.array(imagef)
imagef[coordf[0,1]-1:coordf[0,1]+2,coordf[0,0]-1:coordf[0,0]+2] = 255
imagef[coordf[1,1]-1:coordf[1,1]+2,coordf[1,0]-1:coordf[1,0]+2] = 255
plt.imshow(imagef,cmap='gray')
plt.show()
# translate
coord = torch.tensor([[240,97],[230,320]])
imaget, coordt = my_translate(imagec, coord)
imaget = np.array(imaget)
imaget[coordt[0,1]-1:coordt[0,1]+2,coordt[0,0]-1:coordt[0,0]+2] = 255
imaget[coordt[1,1]-1:coordt[1,1]+2,coordt[1,0]-1:coordt[1,0]+2] = 255
plt.imshow(imaget,cmap='gray')
plt.show()

'''


