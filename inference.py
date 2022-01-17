import argparse
import sys
import torch
import torch.nn as nn
from models.resnet import Resnet
from models.vgg import VGG
import PIL
import torchvision.transforms as transforms

def inference(img_path, model, data_transform, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pretrained_model = opt.pretrained_model
    show_img = opt.show_img
    img = PIL.Image.open(img_path)
    if show_img:
        img.show()
    
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    img = data_transform(img)
    img = torch.unsqueeze(img, 0).to(device)
    pred = model(img)
    prediction = torch.argmax(pred, dim=1).item()
    print(f"---------- {img_path} inference : {prediction} ----------")
    return pred, prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./img/test.jpg',help='the path of image for inference')
    parser.add_argument('--model_name', type=str, default='vgg', help='the name of model - resnet or vgg16')
    parser.add_argument('--pretrained_model', type=str, default='./save_files/vgg_1-13-8-56/vgg_25ep_8b_best.pt', help='the path of pretrained model(.pt)')
    parser.add_argument('--show_img', type=bool, default=False, help='if it''s true, show image')

    opt = parser.parse_args()
    if opt.model_name == 'resnet':
        number = input("the number of resnet layers(18, 34, 50, 101, 152): ")
        number = int(number)
        model = Resnet(number)
        data_transform = transforms.Compose([
        transforms.ToTensor(), # normalize는 PIL이미지 형태가 아닌 Tensor형태에서 수행되어야하므로 앞에 이 줄을 꼭 넣어줘야함
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    elif opt.model_name == 'vgg':
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

    inference(opt.img_path, model, data_transform, opt)