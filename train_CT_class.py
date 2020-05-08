import read_data
import resnet as model
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
from PIL import ImageFile
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

def train_model(model,train_path,test_path,num_opochs=5000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(num_opochs):
        running_loss = 0.0
        model.train(True)
        exp_lr_scheduler.step()
        lines= list(open(train_path, 'r'))
        steps=len(lines)/batch_size
        random.shuffle(lines)
        for i in range(int(steps)):
            # get the inputs
            inputs, lables = read_data.get_data(lines,batch_size,i)
            # wrap them in Variable
            inputs, lables = Variable(torch.Tensor(inputs)).cuda(), Variable(torch.Tensor(lables)).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            _, predicted = torch.max(outputs, 1)
            lables=lables.to(device=torch.device("cuda:0"), dtype=torch.int64)
            correct = (predicted == lables).sum()
            acc_train=float(correct) / batch_size

            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            # print statistics
            print('[%d, %5d] loss: %.3f  time:  %.3f' %
                  (epoch + 1, i + 1, loss,end_time-start_time))
            running_loss += loss

            if i % 10 == 0:
                acc = test(model, test_path)
                print('test [%d, %5d] loss: %.3f acc:  %.3f' %
                      (epoch + 1, i + 1, running_loss / 10,acc))
                running_loss = 0.0
                torch.save(model.state_dict(), 'model/model_'+str(epoch)+'_'+str(i)+'_'+str(acc)+'.pkl')

    print('Finished Training')


num_classes = 2
pretrain_path='./resnet-18-kinetics.pth'
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
print('loading pretrained model {}'.format(pretrain_path))
pretrain = torch.load(pretrain_path)
assert 'resnet-18' == pretrain['arch']
model_resnet.load_state_dict(pretrain['state_dict'])
model_resnet.module.conv1=nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
model_resnet.module.conv1=model_resnet.module.conv1.cuda()
model_resnet.module.fc = nn.Linear(3584,
                                   num_classes)
model_resnet.module.fc = model_resnet.module.fc.cuda()
print(model_resnet)



train_model(model_resnet, train_path, val_path)
