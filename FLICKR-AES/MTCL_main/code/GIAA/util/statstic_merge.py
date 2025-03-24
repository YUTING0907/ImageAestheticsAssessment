import csv  
import random  
# 定义文件名和拆分比例  
label_dir = '/home/ps/temp/model/FLICKR-AES/label/'
input_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_image_score.txt'  
train_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_GIAA_train.csv'  
test_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_GIAA_test.csv'  
split_ratio = 0.8  # 80%用于训练，20%用于测试  

import pandas as pd

statistics_file = label_dir + "photo_statistics.csv"  # 替换为你的 photo_statistics.csv 文件路径

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
photo_statistics = pd.read_csv(statistics_file)

# 确保文件中都有 imagePair 列
assert "imagePair" in train_data.columns
assert "imagePair" in test_data
assert "imagePair" in photo_statistics

# 合并 train.csv 和 photo_statistics.csv
train_merged = pd.merge(train_data, photo_statistics, on="imagePair", how="left")

# 合并 test.csv 和 photo_statistics.csv
test_merged = pd.merge(test_data, photo_statistics, on="imagePair", how="left")

# 保存结果到新文件
train_merged.to_csv(label_dir+"train_with_statistics.csv", index=False)
test_merged.to_csv(label_dir+"test_with_statistics.csv", index=False)

print("完成：已保存 'train_with_statistics.csv' 和 'test_with_statistics.csv'")

