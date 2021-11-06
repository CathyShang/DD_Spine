"""
使用横断面进行分类，修改了backbone后这里的层数也要修改
如果要使用矢状面的分类，需要调用class_model,并修改disease-model 中 from .class_model import ClassModel为from .t_class_model import ClassModel

by @mruniquejj
"""

import torch
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from ..data_utils import SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_DISEASE_ID,SPINAL_VERTEBRA_ID, SPINAL_DISC_ID

class ClassModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN,num_vertebra_diseases: int = len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases: int = len(SPINAL_DISC_DISEASE_ID),num_vertebra_points: int = len(SPINAL_VERTEBRA_ID),
                 num_disc_points: int = len(SPINAL_DISC_ID),pixel_mean=0.5, pixel_std=1):
        super().__init__()
        self.backbone = backbone
        self.maxpool = torch.nn.MaxPool2d(8, stride=4)
        self.conv2d = torch.nn.Conv2d(self.backbone.out_channels, 1, kernel_size=1)
        self.linear = torch.nn.Linear(225, 5)
        self.num_vertebra_diseases=num_vertebra_diseases
        self.num_disc_diseases = num_disc_diseases
        self.num_vertebra_points = num_vertebra_points
        self.num_disc_points = num_disc_points
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean))
        self.register_buffer('pixel_std', torch.tensor(pixel_std))

    def _cal_scores(self, image):
        """
        :param image: (batch, channel,height, width)
        :return:
        """
        feature_pyramids = self.backbone(image)
        feature_maps = feature_pyramids['0']
        feature_maps = self.conv2d(feature_maps)
        feature_maps_pool = self.maxpool(feature_maps)
        scores = feature_maps_pool.flatten(start_dim=1)
        scores = self.linear(scores)

        return scores

    def forward(self, images, v_labels=None, d_labels=None, t_masks=None):
        """
        :param images: transverse_images
        """

        # images = images[~t_masks]
        images = images.to(self.pixel_mean.device)
        images = images.expand(-1, -1, -1, 3, -1, -1)

        # train
        if self.training:
            transverse_loss = []
            for i in range(images.shape[1]):
                image = images[:, i, 0, :, :, :]
                scores = self._cal_scores(image)

                label = d_labels[:, i, -1]
                labels = torch.zeros_like(scores)
                for column in range(len(labels)):
                    labels[column, label[column]] = 1

                bceloss = torch.nn.BCEWithLogitsLoss()
                loss = bceloss(scores, labels)

                transverse_loss.append(loss)

            return transverse_loss
        # inference
        else:

            scores = torch.zeros(self.num_disc_points, self.num_disc_diseases)
            for i in range(images.shape[1]):
                image = images[:, i, 0, :, :, :]
                score = self._cal_scores(image)
                max_ind = torch.argmax(score, dim=-1)
                scores[i, max_ind] = 1

            return scores



