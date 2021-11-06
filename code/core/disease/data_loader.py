from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ..data_utils import gen_mask
from ..structure import DICOM, Study


class DisDataSet(Dataset):      # map-style dataset
    """
    prob_rotate: 数据增强参数-旋转的概率
    max_angel： 数据增强参数-最大旋转角度
    """
    def __init__(self,
                 studies: Dict[Any, Study],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 prob_rotate: float,
                 max_angel: float,
                 num_rep: int,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 k_nearest: int,
                 max_dist: int):
        self.studies = studies
        self.annotations = []
        for k, annotation in annotations.items():
            study_uid, series_uid, instance_uid = k
            if study_uid not in self.studies:
                continue
            study = self.studies[study_uid]
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                self.annotations.append((k, annotation))

        self.prob_rotate = prob_rotate
        self.max_angel = max_angel
        self.num_rep = num_rep
        self.sagittal_size = sagittal_size
        self.transverse_size = transverse_size
        self.k_nearest = k_nearest
        self.max_dist = max_dist

    def __len__(self):
        return len(self.annotations) * self.num_rep

    def __getitem__(self, item) -> (Study, Any, (torch.Tensor, torch.Tensor)):  # 获取study, id, annotations
        item = item % len(self.annotations)
        key, (v_annotation, d_annotation) = self.annotations[item]
        return self.studies[key[0]], key, v_annotation, d_annotation        # 这里的key包含（study_uid, series_uid, instance_uid）

    def collate_fn(self, data) -> (Tuple[torch.Tensor], Tuple[None]):       # study, key, v_anno, d_anno
        sagittal_images, transverse_images, vertebra_labels, disc_labels, distmaps = [], [], [], [], []
        v_masks, d_masks, t_masks = [], [], []
        for study, key, v_anno, d_anno in data:
            v_mask = gen_mask(v_anno)                                       # 这里有标签的点为True，没标签的点为False
            d_mask = gen_mask(d_anno)
            v_masks.append(v_mask)
            d_masks.append(d_mask)

            transverse_image, t_mask = study.t2_transverse_k_nearest(
                d_anno[:, :2], k=self.k_nearest, size=self.transverse_size, max_dist=self.max_dist,
                prob_rotate=self.prob_rotate, max_angel=self.max_angel
            )

            t_masks.append(t_mask)          # 没有的位置标为TRUE
            transverse_images.append(transverse_image)

            # seriesuid确定矢状面序列
            dicom: DICOM = study[key[1]][key[2]]
            pixel_coord = torch.cat([v_anno[:, :2], d_anno[:, :2]], dim=0)
            sagittal_image, pixel_coord, distmap = dicom.transform(
                pixel_coord, self.sagittal_size, self.prob_rotate, self.max_angel, distmap=True)

            distmaps.append(distmap)
            sagittal_images.append(sagittal_image)

            v_label = torch.cat([pixel_coord[:v_anno.shape[0]], v_anno[:, 2:]], dim=-1)
            d_label = torch.cat([pixel_coord[v_anno.shape[0]:], d_anno[:, 2:]], dim=-1)
            vertebra_labels.append(v_label)
            disc_labels.append(d_label)

        sagittal_images = torch.stack(sagittal_images, dim=0)
        distmaps = torch.stack(distmaps, dim=0)
        transverse_images = torch.stack(transverse_images, dim=0)
        vertebra_labels = torch.stack(vertebra_labels, dim=0)
        disc_labels = torch.stack(disc_labels, dim=0)
        v_masks = torch.stack(v_masks, dim=0)
        d_masks = torch.stack(d_masks, dim=0)
        t_masks = torch.stack(t_masks, dim=0)

        data = (sagittal_images, transverse_images, distmaps, vertebra_labels, disc_labels, v_masks, d_masks, t_masks)
        label = (None, )
        return data, label


class DisDataLoader(DataLoader):
    # TODO 添加一些sampling的方法
    def __init__(self, studies, annotations, batch_size, sagittal_size, transverse_size, k_nearest,
                 num_workers=0, prob_rotate=False, max_angel=0, max_dist=6, num_rep=1, pin_memory=False):
        dataset = DisDataSet(studies=studies, annotations=annotations, sagittal_size=sagittal_size,
                             transverse_size=transverse_size, k_nearest=k_nearest, prob_rotate=prob_rotate,
                             max_angel=max_angel, num_rep=num_rep, max_dist=max_dist)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory, collate_fn=dataset.collate_fn)

