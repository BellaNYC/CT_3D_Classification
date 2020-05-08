import read_data
from torchvision import transforms,datasets
import resnet as model
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
from PIL import ImageFile,Image
import os,shutil
import random
import time


ImageFile.LOAD_TRUNCATED_IMAGES = True
batch_size=2


def test(model, test_path):
    sum_time=0
    model.eval()
    lines = list(open(test_path, 'r'))
    sum=0
    for video_data in lines:
        input, lable=read_data.get_frame_data(video_data)
        input=np.array(input).transpose(3,0,1,2)
        input=[input]
        lable=[lable]
        inputs, lables = Variable(torch.Tensor(input)).cuda(),Variable(torch.Tensor(lable)).cuda()#, Variable(lable).cuda()
        outputs = model(inputs.cuda())
        _, predicted = torch.max(outputs, 1)
        lables = lables.to(device=torch.device("cuda:0"), dtype=torch.int64)
        if lables[0]==predicted[0]:
            sum=sum+1
    return sum/len(lines)



num_classes = 2
train_path='./data/train_data.txt'
test_path='./data/test_data.txt'
val_path='./data/val_data.txt'
model_resnet = model.resnet18(
                num_classes=400,
                shortcut_type='A',
                sample_size=112,
                sample_duration=16)
model_resnet = model_resnet.cuda()
model_resnet = nn.DataParallel(model_resnet, device_ids=[0])
model_resnet.module.conv1=nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
model_resnet.module.conv1=model_resnet.module.conv1.cuda()
model_resnet.module.fc = nn.Linear(3584,
                                   num_classes)
model_resnet.module.fc = model_resnet.module.fc.cuda()
print(model_resnet)

model_resnet.load_state_dict(torch.load('./model/model_11_0_0.875.pkl'))
print(test(model_resnet,val_path))