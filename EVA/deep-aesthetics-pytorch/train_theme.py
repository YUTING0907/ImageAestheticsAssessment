"""
训练(GPU)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from torchvision import transforms
from skimage import transform
import random
from torch.utils.data.dataloader import default_collate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image / 1.0  # / 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class Dataset_GIAA(Dataset):
    """Images dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
        img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
        im = Image.open(img_name+'.jpg').convert('RGB')
        if im.mode == 'P':
            im = im.convert('RGB')
        image = np.asarray(im)
        # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        rating = self.images_frame.iloc[idx, 1:]
        sample = {'image': image, 'rating': rating}

        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data():
    data_dir = os.path.join(r'/home/ps/temp/model/EVA/deep-aesthetics-pytorch/data/eva-dataset-master/data')
    data_image_dir = os.path.join(r'/home/ps/temp/model/EVA/deep-aesthetics-pytorch/data/eva-dataset-master/images/EVA_together')
    data_image_train_dir = os.path.join(data_dir, 'image_content_category_train.csv')
    data_image_test_dir = os.path.join(data_dir, 'image_content_category_test.csv')

    transformed_dataset_train = Dataset_GIAA(
        csv_file=data_image_train_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(256, 256)),
             RandomHorizontalFlip(0.5),
             RandomCrop(
                 output_size=(224, 224)),
             Normalize(),
             ToTensor(),
             ])
    )
    transformed_dataset_valid = Dataset_GIAA(
        csv_file=data_image_test_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(224, 224)),
             Normalize(),
             ToTensor(),
             ])
    )
    data_train = DataLoader(transformed_dataset_train, batch_size=64,
                            shuffle=True, num_workers=8, collate_fn=my_collate, drop_last=False)
    data_valid = DataLoader(transformed_dataset_valid, batch_size=64,
                            shuffle=True, num_workers=8, collate_fn=my_collate, drop_last=False)

    df_train = pd.read_csv(data_image_train_dir)
    # 获取数据行数
    df_test = pd.read_csv(data_image_test_dir)
    
    train_num = len(df_train)
    valid_num = len(df_test)
    return data_train, data_valid,train_num,valid_num


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    data_train, data_valid,train_num,val_num = load_data()
    batch_size = 64    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算
    nw = 0

    """ 测试数据集图片"""
    # def imshow(img):
    #     img = img / 2 + 0.5
    #     np_img = img.numpy()
    #     plt.imshow(np.transpose(np_img, (1, 2, 0)))
    #     plt.show()
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = models.resnet34(True)    # 实例化网络
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 6)  # (6分类)
    #net.fc = nn.Sequential(nn.Linear(in_channel, 1), nn.Sigmoid())

    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.000001)

    epochs = 20
    save_path = "./ResNet34_GPU.pth"
    best_accuracy = 0.0
    train_steps = len(data_train)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(data_train, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data['image'],data['rating']
            optimizer.zero_grad()
            logits = net(images.to(device).float())
            loss = loss_function(logits, labels.to(device).squeeze().long()-1) # 计算损失函数
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss
                                                                      )
        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(data_valid, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data['image'],val_data['rating']
                outputs = net(val_images.to(device).float())
                
                predict_y = torch.max(outputs, dim=1)[1]
                
                predict_y = predict_y.unsqueeze(1).unsqueeze(1)
            
                acc += torch.eq(predict_y, val_labels.to(device)-1).sum().item()

        print("------------")
        print(acc)
        print(val_num)
        val_accuracy = acc / val_num
        print("[epoch %d ] train_loss: %3f    val_accurancy: %3f" %
              (epoch + 1, running_loss / train_steps, val_accuracy))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(net,save_path)
            #torch.save(net.state_dict(), save_path)
    print("Finished Training.")

if __name__ == '__main__':
    main()



