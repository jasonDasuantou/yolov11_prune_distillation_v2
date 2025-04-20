# -*- encoding: utf-8 -*-
'''
@File    :   DataLoade.py
@Time    :   2020/08/01 10:58:51
@Author  :   AngYi
@Contact :   angyi_jq@163.com
@Department   :  QDKD shuli
@description : 创建Dataset类，处理图片，弄成trainloader validloader testloader
'''

# here put the import lib
import pandas as pd 
import numpy as np 
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
import torch
random.seed(78)

class CustomDataset(Dataset):
    def __init__(self,data_root,input_width,input_height, kind="train", cls=4):
        super(CustomDataset, self).__init__()  # 在子类进行初始化时，也想继承父类的__init__()就通过super()实现
        self.img_dir = os.path.join(data_root, "images", kind)
        self.lab_dir = os.path.join(data_root, "masks", kind)
        self.image_list = [ os.path.join(self.img_dir, name) for name in os.listdir(self.img_dir)]
        self.label_list = [ os.path.join(self.lab_dir, name) for name in os.listdir(self.lab_dir)]
        self.width = input_width
        self.height = input_height
        self.cls = cls

    def __len__(self):
        return len(self.image_list)

        
    def __getitem__(self,index):
        img = Image.open(self.image_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('RGB')


        img,label  = self.train_transform(img,label,crop_size=(self.width,self.height))

        # assert(img.size == label.size)s
        return img,label




    def train_transform(self,image,label,crop_size=(256,256)):
        '''
        :param image: PIL image
        :param label: PIL image
        :param crop_size: tuple
        '''

        image,label=RandomCrop(crop_size)(image,label) # 第一个括号是实例话对象，第二个是__call__方法
        tfs=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        image=tfs(image)

        label=image2label(self.cls)(label)
        label=torch.from_numpy(label).long()
        return image,label


class RandomCrop(object):
    """
    Crop the given PIL Image at a random location.
    自定义实现图像与label随机裁剪相同的位置
    没办法直接使用transform.resize() 因为是像素级别的标注，而resize会将这些标注变成小数
    """
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label):
        i, j, h, w = self.get_params(img, self.size)
        return img.crop((j,i,j+w,i+h)),label.crop((j,i,j+w,i+h))



class image2label():
    '''
    现在的标签是每张图都是黑色背景，白色边框标记物体，那么要怎么区分飞机和鸟等21类物体，我们需要将标签
    改为背景是[0,0,0],飞机是[1,1,1],自行车是[2,2,2]...
    voc classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
    '''
    def __init__(self,num_classes=4):
        classes = ['background','Split','burr','Pit','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
        # 给每一类都来一种颜色
        colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

        self.colormap=colormap[:num_classes]

        cm2lb=np.zeros(256**3)  # 创建256^3 次方空数组，颜色的所有组合
        for i,cm in enumerate(self.colormap):
            cm2lb[(cm[0]*256+cm[1])*256+cm[2]]=i # 符合这种组合的标记这一类
            # 相当于创建了一个类别的颜色条，这里比较难理解
        self.cm2lb=cm2lb

    def __call__(self, image):
        '''
        :param image: PIL image
        :return:
        '''
        from PIL import Image
        a=np.array(image,dtype=np.uint8)
        img1 = Image.fromarray(a)
        img1.save('1.png')
        if a.max() > 1:
            print('sdadaw')
        image=np.array(image,dtype=np.int64)

        # # 创建新数组 (640, 640)，初始化为0
        # new_array = np.zeros((640, 640), dtype=np.int64)
        #
        # # 筛选满足条件的位置：R=128, G=0, B=0
        # condition1 = (image[:, :, 0] == 128) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
        # condition2 = (image[:, :, 0] == 0) & (image[:, :, 1] == 128) & (image[:, :, 2] == 0)
        # condition3 = (image[:, :, 0] == 128) & (image[:, :, 1] == 128) & (image[:, :, 2] == 0)
        #
        # # 将满足条件的位置映射到新数组上
        # new_array[condition1] = 1
        # new_array[condition2] = 2
        # new_array[condition3] = 3
        # b=np.array(new_array*50,dtype=np.uint8)
        # img2 = Image.fromarray(b)
        # img2.save('2.png')

        idx=(image[:,:,0]*256+image[:,:,1])*256+image[:,:,2]
        label=np.array(self.cm2lb[idx],dtype=np.int64) # 根据颜色条找到这个label的标号
        return label

#根据label结合colormap得到原始颜色数据
class label2image():
    def __init__(self, num_classes=4):
        self.colormap = colormap(256)[:num_classes].astype('uint8')

    def __call__(self, label_pred,label_true):
        '''
        :param label_pred: numpy
        :param label_true: numpy
        :return:
        '''
        pred=self.colormap[label_pred]
        true=self.colormap[label_true]
        return pred,true


#voc数据集class对应的color
def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])
    return cmap

if __name__ == "__main__":
    pass
    # DATA_ROOT = './data/'
    # traindata = CustomTrainDataset(DATA_ROOT,256,256)
    # traindataset = DataLoader(traindata,batch_size=2,shuffle=True,num_workers=0)

    # for i,batch in enumerate(traindataset):
    #     img,label = batch
    #     print(img,label)

    # l1 = Image.open('data/SegmentationClass/2007_000032.png').convert('RGB')

    # label = image2label()(l1)
    # print(label[150:160, 240:250])
