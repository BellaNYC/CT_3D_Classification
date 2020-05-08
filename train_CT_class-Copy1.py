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


def test(model, test_path, loss_func):
    model.train(False)
    test_correct=0
    loss_total =0
    lines = list(open(test_path, 'r'))
    for video_data in lines:
        input, lable=read_data.get_frame_data(video_data)
        input=np.array(input).transpose(3,0,1,2)
        input=[input]
        lable=[lable]
        inputs, lables = Variable(torch.Tensor(input)).cuda(),Variable(torch.Tensor(lable)).cuda()#, Variable(lable).cuda()
        outputs = model(inputs.cuda())
        lables = torch.tensor(lables, dtype=torch.long, device=torch.device("cuda:0"))
        test_loss = loss_func(outputs, lables)
        loss_total += test_loss
        
        prob = outputs.exp()
        _, predicted = torch.max(outputs, 1)
        lables = lables.to(device=torch.device("cuda:0"), dtype=torch.int64)
        if lables==predicted:
            test_correct+=1
            
    return float(test_correct/len(lines)), float(loss_total/len(lines))

def train_model(model,train_path,test_path,num_epochs=100):
    accuracy_train = []
    loss_train = []
    best_acc = 0
    loss_test=[]
    accuracy_test=[]
    criterion = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(num_epochs):
        train_correct = 0
        running_img = 0
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
            lables = torch.tensor(lables, dtype=torch.long, device=torch.device("cuda:0"))
            loss = criterion(outputs, lables)
            prob = outputs.exp()
            _, predicted = torch.max(outputs, 1)
            lables=lables.to(device=torch.device("cuda:0"), dtype=torch.int64)
            
            train_correct+= torch.sum(predicted==lables).item()
            running_img +=batch_size
    
            
            
            # statistics
            running_loss += loss.item()*batch_size # edit
            
            loss.backward()
            optimizer.step()
            
        acc_train_epoch=float(train_correct / running_img)
        loss_train_epoch=float(running_loss / running_img)
            # print statistics
        print('[%d] Train loss: %.3f  acc: %.3f  time:  %.3f' %
              (epoch + 1, loss_train_epoch,acc_train_epoch, end_time-start_time))
     # Testing model
        loss_train.append(loss_train_epoch)
        accuracy_train.append(acc_train_epoch)
        
        test_acc, test_loss = test(model, test_path, criterion)             
          
        print('Test [%d] loss: %.3f acc:  %.3f' %
              (epoch + 1, test_loss, test_acc))
        loss_test.append(test_loss)
        accuracy_test.append(test_acc)
        
        if test_acc >best_acc:
            best_acc= test_acc
            best_model_weights= model.state_dict()
        torch.save(model.state_dict(), 'model/model_'+str(epoch)+'_'+str(i)+'_'+str(test_acc)+'.pkl')
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), 'Bestmodel.pkl')
    return model, accuracy_train, loss_train, accuracy_test, loss_test
    print('Finished Training')
    
def val_model(model,val_path,loss_fn):
    model.eval()
    val_correct=0
    val_loss=0
    lines = list(open(val_path, 'r'))
    labellist =[]
    predlist = []
    for video_data in lines:
        input, lable=read_data.get_frame_data(video_data)
        input=np.array(input).transpose(3,0,1,2)
        input=[input]
        lable=[lable]
        labellist.append(lable)
        inputs, lables = Variable(torch.Tensor(input)).cuda(),Variable(torch.Tensor(lable)).cuda()#, Variable(lable).cuda()
        outputs = model(inputs.cuda())
        lables = torch.tensor(lables, dtype=torch.long, device=torch.device("cuda:0"))
        loss = loss_fn(outputs, lables)
#         print(outputs)
        prob = outputs.exp()
        _, predicted = torch.max(outputs, 1)
        lables = lables.to(device=torch.device("cuda:0"), dtype=torch.int64)
        if lables==predicted:
            val_correct+=1
        val_loss +=loss
        prob = (prob.cpu().detach().numpy()).squeeze()
        predlist.append(prob[1])
         
    return float(val_correct/len(lines)), float(val_loss/len(lines)), labellist, predlist

       

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
model_resnet.module.softmax = model_resnet.module.softmax.cuda()
# print(model_resnet)

import sklearn.metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

model, acc_train, loss_train, acc_test, loss_test = train_model(model_resnet, train_path, val_path)
loss_function = torch.nn.CrossEntropyLoss()
acc_val, loss_val, y_true, prob = val_model(model,val_path,loss_function)
print(' Val loss: %.3f  acc: %.3f ' %
      (loss_val,acc_val))
print(y_true, prob)
fpr, recall, thresholds = roc_curve(y_true, prob, pos_label =1)
roc_auc =auc(fpr,recall)
print('AUC is: %.3f'%(roc_auc))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


fig3 = plt.figure()
plt.title('ROC')
plt.plot(fpr,recall,'b', label = 'AUC= %0.3f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[1,0], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('Fall-out(1-specifity)')
plt.ylabel('Recall')
plt.show()
fig3.savefig('ROCtrial_9.png')


fig1 = plt.figure()
plt.plot(loss_train, label = 'loss_train')
plt.plot(loss_test, label = 'loss_test')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.title('Loss value vs Epochs')
plt.legend()
plt.show()
fig1.savefig('loss_9.png')


fig2 = plt.figure()
plt.plot(acc_train, label = 'acc_train')
plt.plot(acc_test, label = 'acc_test')
plt.ylabel('Acc Value')
plt.xlabel('Epoch')
plt.title('Acc value vs Epochs')
plt.legend()
plt.show()
fig2.savefig('acc_9.png')