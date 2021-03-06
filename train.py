import argparse
import sys
import time
import os
import datetime
from unittest import result
from models.resnet import Resnet
from models.vgg import VGG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def accuracy(pred_tensor, y_tensor):
    # accuracy = torch.eq(pred_tensor, y_tensor).sum().item() / len(pred_tensor)
    accuracy = torch.sum(pred_tensor == y_tensor) / len(pred_tensor)
    return accuracy

def evaluate(model, testloader, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    # Starts batch test
    for x_batch, y_batch in testloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        pred = F.softmax(pred, dim=1)
        loss = nn.CrossEntropyLoss()
        loss_output=loss(pred, y_batch).item()
        test_loss += loss_output
        # acc = accuracy(pred, y_batch)
        prediction = torch.argmax(pred, dim=1)
        acc = accuracy(prediction, y_batch)
        test_acc += acc

    test_acc = test_acc/ len(testloader)
    test_loss = test_loss / len(testloader)

    return test_loss, test_acc


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs, batch_size, model_name = opt.num_epochs, opt.batch_size, opt.model_name
    start_epoch, load_model, dataset_path, save_txt = opt.start_epoch, opt.load_model, opt.dataset_path, opt.save_txt
    print( f"%% model name : {model_name}, num epochs : {num_epochs}, batch size : {batch_size} %%")
    
    if model_name == 'resnet':
        number = input("the number of resnet layers(18, 34, 50, 101, 152): ")
        number = int(number)
        model = Resnet(number)
        data_transform = transforms.Compose([
        transforms.ToTensor(), # normalize는 PIL이미지 형태가 아닌 Tensor형태에서 수행되어야하므로 앞에 이 줄을 꼭 넣어줘야함
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    elif model_name == 'vgg':
        number = input("the number of vgg layers(11, 13, 16, 19): ")
        number = int(number)
        model = VGG(number)
        data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # normalize는 PIL이미지 형태가 아닌 Tensor형태에서 수행되어야하므로 앞에 이 줄을 꼭 넣어줘야함
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else: 
        print("it's not appropriate name")
        sys.exit(0)

   
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    # resize 할까말까 고민함 근데? 어차피 마지막에 averagepooling하니까 안해도 된다고 판단
    
    trainset = ImageFolder(root=dataset_path+"/train", transform=data_transform)
    # testset = ImageFolder(root=dataset_path+"/test", transform=data_transform)
    testset = ImageFolder(root=dataset_path+"/val", transform=data_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    # print(trainset.classes) [female, male]
    # print(trainset.class_to_idx) female:0, male:1
    trainloader_length = len(trainloader)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False) # batch size 클 때
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    # nesterov : 현재 위치의 그래디언트 g(\theta_t) 를 이용하는 것이 아니고 현재 위치에서 속도 \mu v_t만큼 전진한 후의 그래디언트 g(\theta_t + \mu v_t) 를 이용합니다. 사람들은 이를 가리켜 선험적으로 혹은 모험적으로 먼저 진행한 후 에러를 교정한다라고 표현합니다. 

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)    
    writer = SummaryWriter()
    start = time.time()
    best_test_acc = 0

    dt = datetime.datetime.now()
    save_model_path = './save_files/'+f"{model_name}_{dt.month}-{dt.day}-{dt.hour}-{dt.minute}"
    os.mkdir(save_model_path)
    if save_txt:
        f = open(save_model_path + '/result.txt','w')
    for epoch in range(start_epoch, num_epochs):
        # Set model in training model
        model.train()
        train_acc = 0
        train_loss = 0
        # Starts batch training
        iter=0
        for x_batch, y_batch in trainloader:
            iter +=1
            msg = '\riteration  %d / %d'%(iter, trainloader_length)
            print(' '*len(msg), end='')
            print(msg, end='')
            time.sleep(0.1)

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            pred = model(x_batch)
            # torch에서 nn.CrossEntropyLoss()는 softmax도 함꼐 이루어지므로 굳이 안해도 됨
            # pred = F.softmax(pred, dim=1)
            
            loss = nn.CrossEntropyLoss()
            loss_output = loss(pred, y_batch)
            # gradient calculation
            loss_output.requires_grad_(True) # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn 해결
            loss_output.backward() 
            # gradient update
            optimizer.step()
            prediction = torch.argmax(pred, dim=1)
            acc = accuracy(prediction, y_batch)
            train_acc += acc
            train_loss += loss_output.item()
        
        # evaluate after 1 epoch training
        torch.cuda.empty_cache()
        test_loss, test_acc = evaluate(model, testloader, device)
        if test_acc > best_test_acc:
            torch.save(model.state_dict(), save_model_path+ f'/{model_name}_{num_epochs}ep_{batch_size}b_best.pt')
            best_test_acc = test_acc
        train_acc = train_acc / trainloader_length
        train_loss = train_loss / trainloader_length
        lr_scheduler.step(test_loss)
        writer.add_scalars('Loss', {'trainloss':train_loss, 'testloss':test_loss}, epoch)
        writer.add_scalars('Accuracy', {'trainacc':train_acc, 'testacc':test_acc}, epoch)
        

        # Metrics calculation
        result_txt = "\nEpoch: %d, loss: %.8f, Train accuracy: %.8f, Test accuracy: %.8f, Test loss: %.8f, lr: %5f" % (epoch+1, train_loss, train_acc, test_acc, test_loss, optimizer.param_groups[0]['lr'])
        f.write(result_txt)
        print(result_txt)
        # print("\nEpoch: %d, loss: %.8f, Train accuracy: %.8f, Test accuracy: %.8f, Test loss: %.8f, lr: %5f" % (epoch+1, train_loss, train_acc, test_acc, test_loss, optimizer.param_groups[0]['lr']))
    finish = time.time()
    print("---------training finish---------")
    total_result_txt = "\nTotal time: %d(sec), Total Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f, Test loss: %.5f" % (finish-start, num_epochs, train_loss, train_acc, test_acc, test_loss)
    # print("\nTotal time: %d(sec), Total Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f, Test loss: %.5f" % (finish-start, num_epochs, train_loss, train_acc, test_acc, test_loss))
    print(total_result_txt)
    f.write(total_result_txt)
    f.close()
    torch.save(model.state_dict(), save_model_path+ f'/{num_epochs}ep_{batch_size}b_final.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_name', type=str, default='resnet', help='the name of model - resnet or vgg16')
    parser.add_argument('--dataset_path', type=str, default='../data_1', help='dataset directory path')
    parser.add_argument('--start_epoch', type=int, default=0, help='the start number of epochs')
    parser.add_argument('--load_model',default=None, type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    opt = parser.parse_args()
    train(opt)