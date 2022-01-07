import argparse
import sys
from models import Resnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def accuracy(pred_tensor, y_tensor):
    temp = pred_tensor - y_tensor
    right_answer = temp.count(0)
    return right_answer / len(pred_tensor)

 
def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_epochs, batch_size, model_name = opt.num_epochs, opt.batch_size, opt.model_name
    
    if model_name == 'resnet':
        number = input("the number of resnet layers: ")
        number = int(number)
        model = Resnet(number).to(device)
    # elif model_name == 'vgg16':
    #   ...

    # elif model_name == 'inception':
    #   ...

    else: 
        print("it's not appropriate name")
        sys.exit(0)
    
    # resize 할까말까 고민함 근데? 어차피 마지막에 averagepooling하니까 안해도 된다고 판단
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
     ])
    trainset = ImageFolder(root="../people_data/train", transform=data_transform)
    testset = ImageFolder(root="../people_data/test", transform=data_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    # print(trainset.classes) [female, male]
    # print(trainset.class_to_idx) female:0, male:1
    
    if model_name == 'resnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False) 
        # nesterov : 현재 위치의 그래디언트 g(\theta_t) 를 이용하는 것이 아니고 현재 위치에서 속도 \mu v_t만큼 전진한 후의 그래디언트 g(\theta_t + \mu v_t) 를 이용합니다. 사람들은 이를 가리켜 선험적으로 혹은 모험적으로 먼저 진행한 후 에러를 교정한다라고 표현합니다. 


    writer = SummaryWriter()
    for epoch in range(num_epochs):
        # Set model in training model
        model.train()
        predictions = []
        train_acc = 0
        # Starts batch training
        for x_batch, y_batch in trainloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            pred = nn.functional.softmax(pred)
            pred = torch.argmax(pred, dim=1)
            pred = pred.type(torch.FloatTensor)
            y_batch = y_batch.type(torch.FloatTensor)
            loss = nn.functional.binary_cross_entropy(pred, y_batch)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            acc = accuracy(pred, y_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_name', type=str, default='resnet', help='the name of model - resnet or vgg16 or inception')


    opt = parser.parse_args()
    train(opt)