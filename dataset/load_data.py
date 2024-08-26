"""
    DFU数据集信息: 5955个有标签训练数据, 其中2552个none, 2555个infection, 621个both, 227个ischaemia
              形成11:11:3:1
              3994个无标签训练数据, 500个无标签验证数据, 5734个无标签测试数据
    ISIC数据集信息: 
"""

import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset.randaugment import RandAugmentPC, RandAugmentMC
from dataset.dfuaugment import DFUAugment
from torchvision.datasets import ImageFolder
import csv
from collections import Counter

label_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_test(img_dir, size):
    '''加载test数据, 并且生成迭代器'''
    # 获取图片路径
    dataname = img_dir.split('/')[-3]
    if dataname == 'DFU':
        dataset = ImageFolder(img_dir, transform=label_transform)
        loader = DataLoader(dataset, batch_size=size) #  replacement 可以重复采样
    elif dataname == 'ISIC':
        dataset = []
        label_dir = img_dir + 'label.csv'
        f = open(label_dir, 'r')
        reader = csv.DictReader(f)  # 以字典形式读取
        # 寻找标签
        for row in reader:
            img_path = img_dir+'images/'+ row['image']+'.jpg'
            index = list(row.values()).index('1.0')
            dataset.append((img_path, int(index-1)))   # index 1, 2 变成0, 1
        f.close()
        dataset = TDataset(imgs=dataset, transform=label_transform)
        loader = DataLoader(dataset, batch_size=size)
    return loader

def load_train_dataset(img_dir):
    '''
        顺序读图片文件路径以及标签
        return:
            label_imgs:[(img_path, label), ...]
            unlabel_imgs:[img_path, ...]
    '''
    dataname = img_dir.split('/')[-3]
    if dataname == 'DFU':
        dataset = ImageFolder(img_dir)
        label = dataset.imgs[0:5955]      # 1043
        unlabel = dataset.imgs[5955:]
    elif dataname == 'ISIC':
        label_dir = img_dir + 'label.csv'
        label = []
        unlabel = []
    
        f = open(label_dir, 'r')
        reader = csv.DictReader(f)  # 以字典形式读取
        # 寻找标签
        for row in reader:
            img_path = img_dir+'images/'+row['image']+'.jpg'
            index = list(row.values()).index('1.0')
            label.append((img_path, index-1))   # index 1, 2 变成0, 1
        f.close()
        # 划分无标签数据
        label, unlabel = get_unlabel_dataset(label)

    return label, unlabel

def generate_loader(imgs, size, islabel, name=''):
    # 此时数据样本为[(tensor(C,H,W),int(class_id)), (tensor,int), ... ]
    if islabel is True: # 有标签
        dataset = TDataset(imgs=imgs, transform=Transform())
    else:               # 无标签
        dataset = TDataset(imgs=imgs, transform=TransformFixMatch())
    # 生成dataset类的迭代对象
    # 在这一步数据变成[[tensor(batch_size, C, W ,H), tensor(class_id)], ...]
    # shuffle=True 每个epoch对数据重新排序
    loader = DataLoader(dataset, batch_size=size, shuffle=True) #  replacement 可以重复采样
    return loader

def get_unlabel_dataset(label_imgs, f=0.4):
    """imgs
        返回第 i+1 折 (i = 0 -> k-1)
        交叉验证时所需要的训练和验证数据, X_train为训练集, X_valid为验证集
    """
    label = [row[1] for row in label_imgs]
    unlabel_imgs = []
    per_unlabel_number = Counter(label)  # dict
    for k in per_unlabel_number.keys():
        per_unlabel_number[k] *= f
    for i, img in enumerate(label_imgs):
        if per_unlabel_number[img[1]] >= 1:
            unlabel_imgs.append(img)
            label_imgs.remove(img)
            per_unlabel_number[img[1]] -= 1
    
    return label_imgs, unlabel_imgs

class Transform(object):
    def __init__(self):
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            ])
        self.nomalize = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        strong = self.strong(x)
        return self.nomalize(strong)

class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # DFUAugment(n=2)
            RandAugmentMC(n=2, m=10)
            ])
        self.nomalize = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.nomalize(weak), self.nomalize(strong)

class TDataset(Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]  # list类型
        img = Image.open(img_path)
        img = self.transform(img)   # 无标签的图片会返回一个[weak, strong]
        return index, img, label

    def __len__(self):
        return len(self.imgs)
