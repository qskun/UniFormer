from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from . import augs_TIBA as img_trsform


def build_additional_strong_transform():
    strong_aug_nums = 3     # 随机选择三种强增强
    flag_use_rand_num = True
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)  # 对图像进行深拷贝，不会影响原图像的大小

        trs_form_strong = build_additional_strong_transform()

        img_s1 = trs_form_strong(img_s1)
        img_s2 = trs_form_strong(img_s2)

        # if random.random() < 0.8:  # 随机生成的浮点数是否小于0.8，即以80%的概率执行下面的颜色抖动操作。
        #     img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)  # 随机改变图像的亮度、对比度、饱和度和色相。
        # img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)  # 随机灰度转换
        # img_s1 = blur(img_s1, p=0.5)  # 模糊处理，p为灰度处理的概率
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)  # 生成CutMix操作所需的区域框。img_s1.size[0]表示图像宽度。CutMix是一种数据增强技术。

        # if random.random() < 0.8:  # 同样的方法获取第二份强增强图像img_s2
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))  # 创建一个ignore_mask，其大小与原始掩码相同，但所有值都设置为 0。

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)  # 标准化处理

        mask = torch.from_numpy(np.array(mask)).long()  # 将mask转换为longTensor
        ignore_mask[mask == 254] = 255  # 将ignore_mask中掩码值为254转换为255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
