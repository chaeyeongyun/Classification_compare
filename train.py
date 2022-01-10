import argparse
import sys
import time
from models import Resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def accuracy(pred_tensor, y_tensor):
    accuracy = torch.eq(pred_tensor, y_tensor).sum().item() / len(pred_tensor)
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
        # pred = torch.max(pred, dim=1).indices
        # pred = pred.type(torch.FloatTensor)
        # y_batch = y_batch.type(torch.FloatTensor)
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

def cpu_evaluate(cpu_model, testloader, device=torch.device('cpu')):
    cpu_model.eval()
    test_acc = 0
    test_loss = 0
    # Starts batch test
    for x_batch, y_batch in testloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = cpu_model(x_batch)
        pred = F.softmax(pred, dim=1)
        # pred = torch.max(pred, dim=1).indices
        # pred = pred.type(torch.FloatTensor)
        # y_batch = y_batch.type(torch.FloatTensor)
        loss = nn.CrossEntropyLoss()
        loss_output=loss(pred, y_batch).item()
        test_loss += loss_output
        # acc = accuracy(pred, y_batch)
        prediction = torch.argmax(pred, dim=1)
        acc = accuracy(prediction, y_batch)
        test_acc += acc
    test_acc = test_acc/ len(testloader)
    test_loss = test_loss / len(testloader)
    torch.cuda.empty_cache()
    return test_loss, test_acc

 
def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if device == torch.device('cuda'):
    #     print("release all unoccupied cached memory currently held by the caching allocator")
    #     torch.cuda.empty_cache()

    num_epochs, batch_size, model_name = opt.num_epochs, opt.batch_size, opt.model_name
    print( f"%% model name : {model_name}, num epochs : {num_epochs}, batch size : {batch_size} %%")
    if model_name == 'resnet':
        number = input("the number of resnet layers(18, 34, 50, 101, 152): ")
        number = int(number)
        model = Resnet(number).to(device)
        model._initialize_weights()
        # cpu_model = Resnet(number).to(torch.device('cpu'))
        # cpu_model._initialize_weights()
    # elif model_name == 'vgg16':
    #   ...

    # elif model_name == 'inception':
    #   ...

    else: 
        print("it's not appropriate name")
        sys.exit(0)
    
    # resize 할까말까 고민함 근데? 어차피 마지막에 averagepooling하니까 안해도 된다고 판단
    data_transform = transforms.Compose([
        transforms.ToTensor(), # normalize는 PIL이미지 형태가 아닌 Tensor형태에서 수행되어야하므로 앞에 이 줄을 꼭 넣어줘야함
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
     ])
    trainset = ImageFolder(root="../people_data/train", transform=data_transform)
    testset = ImageFolder(root="../people_data/test", transform=data_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    # print(trainset.classes) [female, male]
    # print(trainset.class_to_idx) female:0, male:1
    trainloader_length = len(trainloader)
    if model_name == 'resnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False) 
        # nesterov : 현재 위치의 그래디언트 g(\theta_t) 를 이용하는 것이 아니고 현재 위치에서 속도 \mu v_t만큼 전진한 후의 그래디언트 g(\theta_t + \mu v_t) 를 이용합니다. 사람들은 이를 가리켜 선험적으로 혹은 모험적으로 먼저 진행한 후 에러를 교정한다라고 표현합니다. 


    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    writer = SummaryWriter()
    start = time.time()
    for epoch in range(num_epochs):
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
            # pytorch에서는 gradients들을 추후에 backward를 해줄때 계속 더해주기 때문에 우리는 항상 backpropagation을 하기전에 gradients를 zero로 만들어주고 시작을 해야합니다.
            optimizer.zero_grad()
            
            pred = model(x_batch)
            pred = F.softmax(pred, dim=1)
            # print(pred)
            # pred = torch.argmax(pred, dim=1)
            # pred = torch.max(pred, dim=1).indices
            # print(pred)
            # pred = nn.Sigmoid(pred)
            # pred = pred.type(torch.FloatTensor)
            # y_batch = y_batch.type(torch.FloatTensor)
            loss = nn.CrossEntropyLoss()
            # loss = nn.BCELoss()
            loss_output = loss(pred, y_batch)
            # loss = F.binary_cross_entropy(pred, y_batch) 
            # loss = nn.cross_entropy(pred, y_batch) 
            # gradient calculation
            loss_output.requires_grad_(True) # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn 해결
            loss_output.backward() 
            # gradient update
            optimizer.step()
            prediction = torch.argmax(pred, dim=1)
            # print(prediction)
            acc = accuracy(prediction, y_batch)
            train_acc += acc
            train_loss += loss_output.item()
            # print(loss.item())
            
        
        # evaluate after 1 epoch training
        torch.cuda.empty_cache()
        test_loss, test_acc = evaluate(model, testloader, device)
        train_acc = train_acc / len(trainloader)
        train_loss = train_loss / len(trainloader)
        lr_scheduler.step(test_loss)
        writer.add_scalars('Loss', {'trainloss':train_loss, 'testloss':test_loss}, epoch)
        writer.add_scalars('Accuracy', {'trainacc':train_acc, 'testacc':test_acc}, epoch)

        # Metrics calculation
        print("\nEpoch: %d, loss: %.8f, Train accuracy: %.8f, Test accuracy: %.8f, Test loss: %.8f, lr: %5f" % (epoch+1, train_loss, train_acc, test_acc, test_loss, optimizer.param_groups[0]['lr']))
    finish = time.time()
    print("---------training finish---------")
    print("\nTotal time: %d(sec), Total Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f, Test loss: %.5f" % (finish-start, num_epochs, train_loss, train_acc, test_acc, test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_name', type=str, default='resnet', help='the name of model - resnet or vgg16 or inception')


    opt = parser.parse_args()
    train(opt)