import csv  
import os
input_image_dir = '/home/ps/temp/model/FLICKR-AES/image'
# 定义txt文件的路径  
input_csv_file = '/home/ps/temp/model/FLICKR-AES/label/FLICKR-AES_image_labeled_by_each_worker.csv'  
# 读取txt文件内容  
with open(input_csv_file, 'r', newline='', encoding='utf-8') as file:  
    reader = csv.DictReader(file)  # 使用DictReader方便按列名访问数据 
    worker_data = {}  # 创建一个字典来存储以worker为键的数据列表
    files = os.listdir(input_image_dir)
    # 创建一个字典来存储以worker为键的数据列表  
    # 遍历每一行，将数据添加到对应的worker字典中  
    for row in reader: 
        
        worker = row['worker']
        image_pair = row['imagePair']
        score = row['score']

        # 将数据添加到对应的worker字典中
        if worker not in worker_data:
            worker_data[worker] = []
        if image_pair in files:
            print(image_pair)
            worker_data[worker].append([image_pair, score]) 

# 为每个worker创建一个csv文件，并写入数据  
for worker, data in worker_data.items():  
    csv_file_name = f'/home/ps/temp/model/aesthetic2/MTCL_main/code/MTCL/FlickrAES_TrainUser/{worker}.csv'  
    csv_file_path = os.path.join(os.getcwd(), csv_file_name)  # 获取当前工作目录并拼接文件名  
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:  
        csv_writer = csv.writer(csv_file)
        # 写入数据
        csv_writer.writerows(data)  

print(f"数据已拆分为以worker命名的csv文件。")
