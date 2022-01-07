import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

 
def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch, batch_size = opt.num_epoch, opt.batch_size
    data_transform = transforms.Compose([
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
     ])
    trainset = ImageFolder(root="../people_data/train", transform=data_transform)
    testset = ImageFolder(root="../people_data/test", transform=data_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    # print(trainset.classes) [female, male]
    # print(trainset.class_to_idx) female:0, male:1




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    opt = parser.parse_args()
    train(opt)