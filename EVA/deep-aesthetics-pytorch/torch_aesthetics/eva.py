import os
import shutil
from typing import List, Tuple

import torch
import torchvision.transforms as T
import numpy as np
import scipy.io
from PIL import Image
import pandas as pd
#from torchvision.datasets.folder import default_loader
def load_transforms(
    input_shape: Tuple[int, int] = (256, 256),
) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape),
        T.ToTensor()
    ])

def load_transforms_flip(
    input_shape: Tuple[int,int] = (256,256),
) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

class EVA(torch.utils.data.Dataset):

    attributes = [
        'score',
        'visual',
        'composition',
        'quality',
        'semantic',
    ]

    labels_file = "votes.csv"

    def __init__(
        self,
        image_dir: str = "./data/eva-dataset-master/images/EVA_together",
        labels_dir: str = "./data/eva-dataset-master/data",
        transforms: T.Compose = load_transforms()
    ):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        #self.files, self.labels = self.load_split()
        self.df = self.load_split()

    def load_split(self): 
        labels_file = os.path.join(self.labels_dir, self.labels_file)

        # 初始化一个空列表来存储解析后的数据  
        data_list = []  
  
        # 读取文件并逐行处理  
        with open(labels_file, 'r') as file:  
            for line in file:  
                # 去除行尾的换行符并分割等号  
                parts = line.strip().split('=')  
          
                # 由于最后一个元素是用户代理字符串，我们不需要它，因此排除在外  
                parts = parts[:-1]  
          
                # 定义字段名  
                field_names = ['image_id', 'user_id', 'score', 'difficulty', 'visual', 'composition', 'quality', 'semantic']            
                # 如果parts的长度小于字段数量，用None填充缺失的字段  
                parts.extend([None] * (len(field_names) - len(parts)))  
          
                # 将parts转换为字典  
                row_data = dict(zip(field_names, parts))  
          
                # 将字符串类型的数字字段转换为浮点数  
                for key in ['score', 'difficulty', 'visual', 'composition', 'quality', 'semantic']:  
                    if row_data[key].replace('.', '', 1).isdigit():  # 简单的检查来确认是否可以转换为浮点数  
                        row_data[key] = float(row_data[key])  
          
                # 如果factor字段也是数字，可以同样处理  
            
                # 假设vote_time是数据中的最后一个字段，但你没有在parts中包含它  
                # 如果需要，可以从原始line中解析它（但需要注意，这可能需要更复杂的字符串处理）  
          
                # 将解析后的数据添加到列表中  
                data_list.append(row_data)  
  
        # 将数据列表转换为DataFrame  
        df = pd.DataFrame(data_list)
        df.drop(0,inplace=True)
        
        df[['score','visual','composition','quality','semantic']]=df[['score','visual','composition','quality','semantic']].astype(float)
        grouped = df.groupby('image_id')[['score','visual','composition','quality','semantic']].mean().reset_index()
        grouped[['score','visual','composition','quality','semantic']]=grouped[['score','visual','composition','quality','semantic']].astype(float).round(2)

        return grouped  
        #return grouped['image_id'],grouped[['score','visual','composition','quality','semantic']]
      
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        #x = Image.open(self.files[idx]).convert("RGB")
        row = self.df.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        x = Image.open(image_path).convert("RGB")
        #image = default_loader(image_path)
        x = self.transforms(x)
    
        score = np.array(row['score'], np.float32)
        composition = np.array(row['composition'], np.float32)
        visual = np.array(row['visual'], np.float32)
        quality = np.array(row['quality'], np.float32)
        semantic = np.array(row['semantic'], np.float32)

        #y = row[['score','visual','composition','quality','semantic']]
        y =  np.array([score,visual,composition,quality,semantic])  
        y = torch.from_numpy(y)
        return x,y,image_path

EVA()
