from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class Dataprocess(data.Dataset):

    def __init__(self, image_dir, attr_path, transform, mode):
        """Initialize and preprocess Shadow dataset"""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess shadow attribute file."""
        # 打开标签txt文件
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]  # rstrip删除字符串末尾字符，默认空格；open('r') 只读模式
        all_attr_names = lines[0].split()  # all_attr_names表示left、mid、right shadow类别集合
        # for i, attr_name in enumerate(all_attr_names):  # 将属性名称和对应索引分别存入两个列表中
        #     self.attr2idx[attr_name] = i
        #     self.idx2attr[i] = attr_name

        lines = lines[1:]  # 每行为图片名称及其对应标签
        random.seed(1234)
        random.shuffle(lines)  # 每一行为一张图片，打乱图片顺序
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]  # 图片名
            values = split[1:]  # 图片对应的属性标签

            # for attr_name in self.selected_attrs:  # 创建训练选用的属性和索引的一一对应关系
            #     idx = self.attr2idx[attr_name]
            #     label.append(values[idx] == '1')  # 将训练选用的属性标签置为1

            if (i + 1) < 20:  # 此时的i为总图片数量，取其中20张做测试集数据，其余都为训练集数据
                self.test_dataset.append([filename, values])
            else:
                self.train_dataset.append([filename, values])

        print('Finished preprocessing shadow dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        if label[0] == 1:
            image = Image.open(os.path.join(self.image_dir, 'left', filename))
        if label[1] == 1:
            image = Image.open(os.path.join(self.image_dir, 'mid', filename))
        if label[2] == 1:
            image = Image.open(os.path.join(self.image_dir, 'right', filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images