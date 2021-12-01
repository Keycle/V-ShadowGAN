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
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split()
        # for i, attr_name in enumerate(all_attr_names):
        #     self.attr2idx[attr_name] = i
        #     self.idx2attr[i] = attr_name

        lines = lines[1:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            # for attr_name in self.selected_attrs:
            #     idx = self.attr2idx[attr_name]
            #     label.append(values[idx] == '1')

            if (i + 1) < 20:
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
