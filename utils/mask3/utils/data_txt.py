# -*- encoding: utf-8 -*-
'''
@File    :   data_txt.py
@Time    :   2020/08/01 10:36:19
@Author  :   AngYi
@Contact :   angyi_jq@163.com
@Department   :  QDKD shuli
@description： 把图片数据从文件夹整理成csv文件，每一行代表其路径
'''

import numpy as np
import pandas as pd 
import os 
import PIL
from PIL import Image




class image2csv(object):
    # 分割训练集 验证集 测试集
    # 做成对应的txt
    def __init__(self,data_root,image_dir,label_dir,slice_data,width_input,height_input):
        self.data_root = data_root
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.slice_train = slice_data[0]
        self.slice_val = slice_data[1]

        self.width = width_input
        self.height = height_input
        
    


    def read_path(self):
        images = []
        labels = []
        for i,im in enumerate(os.listdir(self.image_dir)):
            label_name = im.split('.')[0] + '.png'   # 读取图片的名字，去label里面找，确保两个文件夹都有这个名字的图
           
            if os.path.exists(os.path.join(self.label_dir,label_name)):
                size_w,size_h = Image.open(os.path.join(self.image_dir,im)).size
                size_lw,size_lh = Image.open(os.path.join(self.label_dir,label_name)).size

                if min(size_w,size_lw) > self.width and min(size_h,size_lh)> self.height:
                    images.append(os.path.join(self.image_dir,im))
                    labels.append(os.path.join(self.label_dir,label_name))
                else:
                    continue



            
        
        assert(len(images)==len(labels)) #

        self.data_length = len(images) # 真正两个文件夹都有的图片的长度

        data_path = {
            'image':images,
            'label':labels,
        }
        
        return data_path


    def generate_csv(self):
        data_path = self.read_path() # 存放了路径

        data_path_pd = pd.DataFrame(data_path)
        train_slice_point = int(self.slice_train*self.data_length) # 0.7*len
        validation_slice_point = int((self.slice_train+self.slice_val)*self.data_length) # 0.8*len

        train_csv = data_path_pd.iloc[:train_slice_point,:]
        validation_csv = data_path_pd.iloc[train_slice_point:validation_slice_point,:]
        test_csv = data_path_pd.iloc[validation_slice_point:,:]

        train_csv.to_csv(os.path.join(self.data_root,'train.csv'),header=None,index=None)
        validation_csv.to_csv(os.path.join(self.data_root,'val.csv'),header = None,index = None)
        test_csv.to_csv(os.path.join(self.data_root,'test.csv'),header=False,index = False)


if __name__ == "__main__":
    DATA_ROOT =  './data/'
    image = os.path.join(DATA_ROOT,'JPEGImages')
    label = os.path.join(DATA_ROOT,'SegmentationClass')
    slice_data = [0.7,0.1,0.2] #  训练 验证 测试所占百分比
    WIDTH = 256
    HEIGHT  =  256
    tocsv = image2csv(DATA_ROOT,image,label,slice_data,WIDTH,HEIGHT)
    tocsv.generate_csv()




        



