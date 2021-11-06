"""
使用矢状面分类，与t_class_model选一个用，并更换输入disease-model中的模型
"""
import random
import torch
import numpy as np
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from ..data_utils import SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_DISEASE_ID,SPINAL_VERTEBRA_ID, SPINAL_DISC_ID
from torchvision import transforms as tt
from torchvision.transforms import functional as tf
from torchvision import utils as vutils
import time
import cv2
from PIL import Image


class ClassModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN,num_vertebra_diseases: int = len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases: int = len(SPINAL_DISC_DISEASE_ID),num_vertebra_points: int = len(SPINAL_VERTEBRA_ID),
                 num_disc_points: int = len(SPINAL_DISC_ID),pixel_mean=0.5, pixel_std=1, disc_size = (64,64),max_angel=45):
        super().__init__()
        self.backbone = backbone
        self.maxpool = torch.nn.MaxPool2d(8, stride=4)
        # self.conv2d = torch.nn.Conv2d(self.backbone.out_channels, 1, kernel_size=1)
        self.linear = torch.nn.Linear(1000, 5)
        self.num_vertebra_diseases=num_vertebra_diseases
        self.num_disc_diseases = num_disc_diseases
        self.num_vertebra_points = num_vertebra_points
        self.num_disc_points = num_disc_points
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean))
        self.register_buffer('pixel_std', torch.tensor(pixel_std))
        self.disc_size = disc_size
        self.crossloss = torch.nn.CrossEntropyLoss()
        self.max_angel = max_angel

    def _cal_scores(self, image):
        """
        :param image: (batch, channel,height, width)
        :return:
        """
        feature_maps = self.backbone(image)         # (8,1000)
        scores = self.linear(feature_maps)          # (8,5)

        return scores

    def disc_augment(self, image: Image):
        """
        disc 图像增强
        :param image: PIL image.
        :return: PIL image after augment.
        """
        # 直方图均衡化
        '''
        image = np.asarray(image)
        image = 255*image
        image = image.astype(np.uint8)
        image = cv2.equalizeHist(image)
        image = Image.fromarray(image)
        '''

        # 提高亮度与对比度
        # image = tf.adjust_contrast(image, 2)    # 提高对比度，数值为整数，0对应灰度，1对应原图
        # image = tf.adjust_brightness(image, 2)  # 增亮，1代表原图

        # 随机调整亮度和对比度
        tf_color = tt.ColorJitter(brightness=[0.8, 1.5], contrast=[0.8, 1.5], saturation=0, hue=0)
        image = tf_color(image)

        return image

    def disc_proposal(self,images, v_labels, d_labels ):
        """
        计算关键点的图片
        :param images: ([8, 3, 512, 512])
        :param v_labels: ([8, 5, 3])
        :param d_labels: ([8, 6, 3])
        :return: disc_images:(8,6,3,64,64）
        """
        disc_images = torch.zeros(d_labels.shape[0], d_labels.shape[1], 1, self.disc_size[0],self.disc_size[1])  # （8,6,3,64,64）

        # 计算每个disc点的框大小
        disc_h = torch.zeros(d_labels.shape[0], 10)  # (8,6)
        for p in range(5):
            disc_h[:, 2*p] = torch.abs(d_labels[:, p, 1] - v_labels[:, p, 1])
            disc_h[:, 2*p+1] = torch.abs(d_labels[:, -(p+1), 1] - v_labels[:, -(p+1), 1])

        disc_h, _ = torch.sort(disc_h,dim=1)
        disc_h = disc_h[:,3:7].mean(dim=1)

        # 获取每个点对应的框并rezise成统一大小
        for i in range(images.shape[0]):
            for point in range(d_labels.shape[1]):
                height_p = 2 * disc_h[i]
                width_p = 3 * disc_h[i]
                top_p = d_labels[i, point, 1] - height_p / 2
                left_p = d_labels[i, point, 0] - width_p / 3

                # 图像增强+裁切
                im = tf.to_pil_image(images[i].cpu())
                if self.training: im = self.disc_augment(im)
                disc_crop = tf.crop(im, int(top_p), int(left_p), int(height_p), int(width_p))

                # 上面3个点逆时针转，最下面的椎间盘顺时针转旋转,正数表示逆时针
                if point < 3:
                    angel = random.randint(0, int(self.max_angel/2))
                    disc_crop = tf.rotate(disc_crop, angel)
                if point == d_labels.shape[1]-1:
                    angel = random.randint(-self.max_angel, -20)
                    disc_crop = tf.rotate(disc_crop, angel)

                disc_images[i, point] = tf.to_tensor(tf.resize(disc_crop, self.disc_size))

        # for ii in range(6):
        #     vutils.save_image(disc_images[:,ii].clone().detach(), './pic/dist'+str(ii) + time.strftime('%y%m%d%H%M%S') + '.jpg')

        return disc_images

    def forward(self, images, v_labels=None, d_labels=None, t_masks=None):
        """

                :param images: sagittal_images      ([8, 1, 512, 512]), value range (0,1)
                :param v_labels: ([8, 5, 3])
                :param d_labels: ([8, 6, 3])
                :param t_masks:
                :return:
        """

        images = images.to(self.pixel_mean.device)
        # images = (images - self.pixel_mean) / self.pixel_std

        disc_images = self.disc_proposal(images, v_labels, d_labels)    # (8,6,-,disc_size,disc_size）
        disc_images = disc_images.expand(-1, -1, 3, -1, -1)
        disc_images = disc_images.to(self.pixel_mean.device)
        disc_images.requires_grad=True


        # train
        if self.training:

            disc_loss = []
            for i in range(d_labels.shape[1]):
                image = disc_images[:, i, :, :, :]                  # ([8,3,64,64])
                scores = self._cal_scores(image)                    # (8,5)

                label = d_labels[:, i, -1]                          # ([8])

                disc_loss.append(self.crossloss(scores, label.cuda()))      # 这里默认求平均得到一个值，只有reduction 为None时输出维度与输入相同

            disc_loss = torch.tensor(disc_loss, requires_grad=True, device=self.pixel_mean.device)    # size ([6])
            disc_loss = torch.mean(disc_loss)

            return disc_loss

        # inference
        else:
            '''
            scores = torch.zeros(images.shape[0],self.num_disc_points,self.num_disc_diseases)    # (batch,6,5)
            for i in range(images.shape[1]):
                image = images[:, i, 0, :, :, :]                    # ([batch,3,512,512])
                score = self._cal_scores(image)                     # (batch,5)
                max_ind = torch.argmax(score,dim=-1).unsqueeze(dim=1)
                for point in range(len(scores)):
                    scores[point,i,max_ind[point]] = 1
            '''

            scores = torch.zeros(self.num_disc_points, self.num_disc_diseases)  # (6,5)
            for i in range(disc_images.shape[1]):
                image = disc_images[:, i, :, :, :]       # ([8,6,3,64,64])
                score = self._cal_scores(image)          # (1,5)
                max_ind = torch.argmax(score, dim=-1)
                scores[i, max_ind] = 1

            return scores



