import csv  
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义文件名和拆分比例  
input_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_image_labeled_by_each_worker.csv'  
train_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_GIAA_mean_train.csv'  
test_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_GIAA_mean_test.csv'  
split_ratio = 0.8  # 80%用于训练，20%用于测试  

data = pd.read_csv(input_file)

# 计算每张照片的平均值、标准差和方差
statistics = data.groupby("imagePair")["score"].agg(
    mean=lambda x: round(x.mean(), 6),
    std=lambda x: round(x.std(), 6),
    var=lambda x: round(x.var(), 6)
).reset_index()

# 拆分为训练集和测试集
train_set, test_set = train_test_split(statistics, test_size=0.2, random_state=42)

# 保存结果到文件
statistics.to_csv("photo_statistics.csv", index=False)
train_set.to_csv(train_file, index=False)
test_set.to_csv(test_file, index=False)

print("完成：统计数据保存为 'photo_statistics.csv'，训练集为 'train_set.csv'，测试集为 'test_set.csv'")
