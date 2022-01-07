import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch, batch_size = opt.num_epoch, opt.baetch_size
    trainset = ImageFolder(root="../people_data/train")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, transform=노말라이즈?)
    testset = ImageFolder(root="../people_data/test")
    testloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, transform=노말라이즈?)
    
    # print(trainset.classes) [female, male]
    # print(trainset.class_to_idx) female:0, male:1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    opt = parser.parse_args()
    train(opt)