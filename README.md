# Classification_compare
compare classification performance some famous CNN models. 

blog post: https://www.notion.so/chang-ai-study/ResNet-VGG16-ea8611424a384257974651df4ef1b635

---
## Install  
clone repo and use this docker image  
docker Image : https://hub.docker.com/repository/docker/changjoy/ubuntu18.04-cuda10.2-cudnn8-torch1.7.1-opencv4.4.0
```python
$ git clone https://github.com/chaeyeongyoon/Classification_compare.git
```

## Training
```python
$ python3 train.py --num_epochs 30 --batch_size 4 --model_name resnet
```
```python
$ python3 train.py --num_epochs 30 --batch_size 4 --model_name vgg16
```
if you want to use your custom dataset, you can set your dataset path  
```python
$ python3 train.py --dataset_path ../dataset
```
If you set save_txt option to true, you can save training process as a txt file  
```python
$ python3 train.py --save_txt True
```
>Epoch: 1, loss: 0.79928847, Train accuracy: 0.56917864, Test accuracy: 0.72906148, Test loss: 0.61428075, lr: 0.000100  
Epoch: 2, loss: 0.50854653, Train accuracy: 0.75551569, Test accuracy: 0.76788092, Test loss: 0.56308582, lr: 0.000100  
Epoch: 3, loss: 0.40448979, Train accuracy: 0.81891280, Test accuracy: 0.79925555, Test loss: 0.51088951, lr: 0.000100  
...
>
If you want to resume learning with a pre-trained model, you can set start epoch and pretrained model path
```python
$ python3 train.py --start_epoch 10 --load_model ./save_files/vgg_1-13-8-56/last.pt
```
## Inference  
You can predict the class of an image.
```python
$ python3 inference.py --img_path ./img/test.jpg --pretrained_model ./checkpoint/best.pt
```
