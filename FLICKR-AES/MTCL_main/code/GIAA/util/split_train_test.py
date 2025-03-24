import csv  
import random  
# 定义文件名和拆分比例  
input_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_image_score.txt'  
train_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_GIAA_train.csv'  
test_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_GIAA_test.csv'  
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
