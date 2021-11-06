import json
import math
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple
from scipy import ndimage

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from .dicom_utils import read_one_dcm

PADDING_VALUE: int = 0


def read_dcms(dcm_dir, error_msg=False) -> (Dict[Tuple[str, str, str], Image.Image], Dict[Tuple[str, str, str], dict]):
    """
    读取文件夹内的所有dcm文件
    :param dcm_dir:
    :param error_msg: 是否打印错误信息
    :return: 包含图像信息的字典，和包含元数据的字典
    """
    dcm_paths = []
    for study in os.listdir(dcm_dir):
        study_path = os.path.join(dcm_dir, study)
        for dcm_name in os.listdir(study_path):
            dcm_path = os.path.join(study_path, dcm_name)
            dcm_paths.append(dcm_path)

    with Pool(cpu_count()) as pool:
        async_results = []
        for dcm_path in dcm_paths:
            async_results.append(pool.apply_async(read_one_dcm, (dcm_path,)))

        images, metainfos = {}, {}
        for async_result in tqdm(async_results, ascii=True):
            async_result.wait()
            try:
                metainfo, image = async_result.get()
            except RuntimeError as e:
                if error_msg:
                    print(e)
                continue
            key = metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            del metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            images[key] = tf.to_pil_image(image)
            metainfos[key] = metainfo

    return images, metainfos


def get_spacing(metainfos: Dict[Tuple[str, str, str], dict]) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    从元数据中获取像素点间距的信息
    :param metainfos:
    :return:
    """
    output = {}
    for k, v in metainfos.items():
        spacing = v['pixelSpacing']
        spacing = spacing.split('\\')
        spacing = list(map(float, spacing))
        output[k] = torch.tensor(spacing)
    return output


with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_vertebra_id.json'), 'r') as file:
    SPINAL_VERTEBRA_ID = json.load(file)

with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_disc_id.json'), 'r') as file:
    SPINAL_DISC_ID = json.load(file)

assert set(SPINAL_VERTEBRA_ID.keys()).isdisjoint(set(SPINAL_DISC_ID.keys()))

with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_vertebra_disease.json'), 'r') as file:
    SPINAL_VERTEBRA_DISEASE_ID = json.load(file)

with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_disc_disease.json'), 'r') as file:
    SPINAL_DISC_DISEASE_ID = json.load(file)


def read_annotation(path) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """

    :param path:
    :return: 字典的key是（studyUid，seriesUid，instance_uid）
             字典的value是两个矩阵，第一个矩阵对应锥体，第二个矩阵对应椎间盘
             矩阵每一行对应一个脊柱的位置，前两列是位置的坐标(横坐标, 纵坐标)，之后每一列对应一种疾病
             坐标为0代表缺失
             ！注意图片的坐标和tensor的坐标是转置关系的
    """
    with open(path, 'r') as annotation_file:
        # non_hit_count用来统计为被编码的标记的数量，用于预警
        non_hit_count = {}
        annotation = {}
        for x in json.load(annotation_file):
            study_uid = x['studyUid']

            assert len(x['data']) == 1, (study_uid, len(x['data']))
            data = x['data'][0]
            instance_uid = data['instanceUid']
            series_uid = data['seriesUid']

            assert len(data['annotation']) == 1, (study_uid, len(data['annotation']))
            points = data['annotation'][0]['data']['point']

            vertebra_label = torch.full([len(SPINAL_VERTEBRA_ID), 3],
                                        PADDING_VALUE, dtype=torch.long)
            disc_label = torch.full([len(SPINAL_DISC_ID), 3],
                                    PADDING_VALUE, dtype=torch.long)
            for point in points:
                identification = point['tag']['identification']
                if identification in SPINAL_VERTEBRA_ID:    # 如果是椎骨的位置
                    position = SPINAL_VERTEBRA_ID[identification]
                    diseases = point['tag']['vertebra']

                    vertebra_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease in SPINAL_VERTEBRA_DISEASE_ID:
                            disease = SPINAL_VERTEBRA_DISEASE_ID[disease]
                            vertebra_label[position, 2] = disease
                elif identification in SPINAL_DISC_ID:      # 如果是椎间盘的位置
                    position = SPINAL_DISC_ID[identification]
                    diseases = point['tag']['disc']

                    disc_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease in SPINAL_DISC_DISEASE_ID:
                            disease = SPINAL_DISC_DISEASE_ID[disease]
                            disc_label[position, 2] = disease
                elif identification in non_hit_count:
                    non_hit_count[identification] += 1
                else:
                    non_hit_count[identification] = 1

            annotation[study_uid, series_uid, instance_uid] = vertebra_label, disc_label
    if len(non_hit_count) > 0:
        print(non_hit_count)
    return annotation


def resize(size: Tuple[int, int], image: Image.Image, spacing: torch.Tensor, *coords: torch.Tensor):
    """

    :param size: [height, width]，height对应纵坐标，width对应横坐标
    :param image: 图像
    :param spacing: 像素点间距
    :param coords: 标注是图像上的坐标，[[横坐标,纵坐标]]，横坐标从左到有，纵坐标从上到下
    :return: resize之后的image，spacing，annotation
    """
    # image.size是[width, height]
    height_ratio = size[0] / image.size[1]
    width_ratio = size[1] / image.size[0]

    ratio = torch.tensor([width_ratio, height_ratio])
    spacing = spacing / ratio
    coords = [coord * ratio for coord in coords]
    image = tf.resize(image, size)

    output = [image, spacing] + coords
    return output


def rotate_point(points: torch.Tensor, angel, center: torch.Tensor) -> torch.Tensor:
    """
    将points绕着center顺时针旋转angel度
    :param points: size of（*， 2）
    :param angel:
    :param center: size of（2，）
    :return:
    """
    if angel == 0:
        return points
    angel = angel * math.pi / 180
    while len(center.shape) < len(points.shape):
        center = center.unsqueeze(0)
    cos = math.cos(angel)
    sin = math.sin(angel)
    rotate_mat = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32, device=points.device)
    output = points - center
    output = torch.matmul(output, rotate_mat)
    return output + center


def rotate_batch(points: torch.Tensor, angels: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    将一个batch的点，按照不同的角度和中心转旋
    :param points: (num_batch, num_points, 2)
    :param angels: (num_batch,)
    :param centers: (num_batch, 2)
    :return:
    """
    centers = centers.unsqueeze(1)
    output = points - centers

    angels = angels * math.pi / 180
    cos = angels.cos()
    sin = angels.sin()
    rotate_mats = torch.stack([cos, sin, -sin, cos], dim=1).reshape(angels.shape[0], 1, 2, 2)
    output = output.unsqueeze(-1)
    output = output * rotate_mats
    output = output.sum(dim=-1)
    return output + centers


def rotate(image: Image.Image, points: torch.Tensor, angel: int) -> (Image.Image, torch.Tensor):
    center = torch.tensor(image.size, dtype=torch.float32) / 2
    return tf.rotate(image, angel), rotate_point(points, angel, center)


def gen_distmap(image: torch.Tensor, spacing: torch.Tensor, *gt_coords: torch.Tensor, angel=0):
    """
    先将每个像素点的坐标顺时针旋转angel之后，再计算到标注像素点的物理距离
    :param image: height * weight
    :param gt_coords: size of（*， 2）
    :param spacing:
    :param angel: 
    :return:
    """

    coord = torch.where(image.squeeze() < np.inf)
    # 注意需要反转横纵坐标
    center = torch.tensor([image.shape[2], image.shape[1]], dtype=torch.float32) / 2
    coord = torch.stack(coord[::-1], dim=1).reshape(image.size(1), image.size(2), 2)
    coord = rotate_point(coord, angel, center)
    dists = []
    for gt_coord in gt_coords:
        gt_coord = rotate_point(gt_coord, angel, center)
        dist = []
        for point in gt_coord:
            dist.append((((coord - point) * spacing) ** 2).sum(dim=-1).sqrt())
        dist = torch.stack(dist, dim=0)
        dists.append(dist)
    '''

    coord = torch.ones(image.shape[1], image.shape[2])

    dists = []
    for gt_coord in gt_coords:
        dist = []
        for point in gt_coord:
            coord[point[0],point[1]] = 0
            d = ndimage.distance_transform_edt(coord, sampling=list(spacing.numpy()))
            dist.append(torch.tensor(d))

        dist = torch.stack(dist, dim=0)
        dists.append(dist)
    '''

    if len(dists) == 1:
        return dists[0]
    else:
        return dists

def gen_heatmap(image: torch.Tensor, joints: torch.Tensor, sigma=2):
    """
    产生高斯热图
    :param image: image
    :param joints:  [num_joints, 2] coord of key points
    :param sigma: 影响着生成的高斯热图中每个点的大小和分布
    :param heatmap_size: 热图大小，alphapose里用的是图像大小的四分之一
    :return: 高斯热图，target[128,128], target_weight[num_joints,1] (1: visible, 0: invisible)
    """

    image_size = image[0].shape
    heatmap_size = image_size
    
    num_joints = joints.shape[0]
    feat_stride = np.array(image_size) / np.array(heatmap_size)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return torch.tensor(target)   #, target_weight



def gen_mask(coord: torch.Tensor):  # 之前将缺少标签的点坐标设为了0，这里没有标签的点设为False
    return (coord.index_select(-1, torch.arange(2, device=coord.device)) != PADDING_VALUE).any(dim=-1)

def sp_noise(image,prob):
    '''

    添加椒盐噪声,输入为PIL.image

    prob:噪声比例

    '''
    image = np.array(image)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    return Image.fromarray(output)

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声,输入为PIL.image
        mean : 均值
        var : 方差
    '''
    image = np.array(image)
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)

    return Image.fromarray(out)

def my_sagital_crop(image, coord):
    """
    矢状位图像随机裁剪
    image: PIL image
    coord: key point, torch.tensor
    """
    image = np.array(image)
    h, w = image.shape
    min_dis = 0.9*(np.min([coord[0,1], h-coord[-1,1]]))      # 防止裁切掉关键点所在区域
    scale_range = min_dis/h
    scale = np.random.rand()*scale_range
    h_side = np.int(h * scale)       # 纵向单边距
    w_side = np.int(w * scale)       # 横向单边距
    l = w_side
    r = w - l
    t = h_side
    b = h - t
    image_crop = image[t:b,l:r]

    coord[:, 0] -= l
    coord[:, 1] -= t

    return Image.fromarray(image_crop), coord

def my_flip(image, coord):
    """
    带关键点的图像翻转.
    :param image: PIL iamge
    :param coord: torch.tensor,第一列为x,第二列y
    :return:
    """
    # horizontal flip
    image = tf.hflip(image)
    h, w = image.size
    coord[:, 0] = w - coord[:, 0]   # 水平翻转只变x,即第一列

    return image, coord

def my_translate(image, coord):
    """
    矢状位图像随机平移
    image: PIL image
    coord: key point, torch.tensor
    """
    h, w = image.size
    min_dis = 0.8*(np.min([coord.min(), h-coord[:,1].max()]))      # 防止裁切掉关键点所在区域
    scale_range = min_dis/h
    scale = np.random.rand()*scale_range
    h_side = np.int(h * scale)       # 纵向单边距
    w_side = np.int(w * scale)       # 横向单边距

    if np.random.rand()<0.5:
        image = tf.affine(image, angle=0, translate=(w_side,0),scale=1,shear=0) # 水平平移,,fillcolor=255
        coord[:, 0] += w_side
    else:
        image = tf.affine(image, angle=0, translate=(0,h_side),scale=1,shear=0)
        coord[:, 1] += h_side

    return image, coord

