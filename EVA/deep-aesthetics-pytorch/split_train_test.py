import csv  
import random
import pandas as pd
from sklearn.model_selection import train_test_split
# 定义文件名和拆分比例 
root_dir = '/home/ps/temp/model/EVA/deep-aesthetics-pytorch/data/eva-dataset-master/data/'
input_file = root_dir+'image_content_category.csv'  
train_file = root_dir+'image_content_category_train.csv'  
test_file = root_dir+'image_content_category_test.csv'  

def read_txt_tocsv():
    split_ratio = 0.8  # 80%用于训练，20%用于测试  
    # 读取txt文件内容  
    with open(input_file, 'r') as file:  
        lines = file.readlines()            
        # 打乱数据顺序  
        random.shuffle(lines)  
        # 计算训练集和测试集的大小  
        train_size = int(len(lines) * split_ratio)  
        test_size = len(lines) - train_size  
        # 准备写入文件  
        with open(train_file, 'w', newline='') as train_csv, open(test_file, 'w', newline='') as test_csv:  
            train_writer = csv.writer(train_csv)  
            test_writer = csv.writer(test_csv)                              
            # 写入训练集  
            for i in range(train_size):  
                train_writer.writerow(lines[i].strip().split())  
            # 写入测试集  
            for i in range(test_size):  
                test_writer.writerow(lines[train_size + i].strip().split())  
    print(f"数据已拆分为 {train_file} 和 {test_file}")

def to_train_val_csv():
    # 读取原始 CSV 文件
    data = pd.read_csv(input_file)
    # 设置训练集和验证集的比例，例如 80% 的数据用于训练，20% 的数据用于验证
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    # 将训练集和验证集分别保存为 train.csv 和 val.csv
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(test_file, index=False)
    print("数据已成功拆分并保存为 train.csv 和 val.csv")

if __name__ == '__main__':
    to_train_val_csv()
