# LDCNet
### Learnable Descriptive Convolutional Network for Face Anti-Spoofing (BMVC'22)

## Feature maps extracted by LDC 
![Screenshot](figure/visualize_low_level.png)
Generate by `Low_level_features_visualization.py` 

## Learnable Descriptive Convolution
![Screenshot](figure/LDC_illustration.png)

## Network Architecture
![Screenshot](figure/LDCNet.png)

## Requirements
```
grad_cam==1.3.5
matplotlib==3.5.2
numpy==1.22.3
scikit_learn==1.1.2
torch==1.12.0
torchvision==0.13.0
```

## Training
Step 1: execute `Amap_train.py` to get pretrained model for producing activation map 

Step 2: execute `train.py` to train LDCNet

Step 3: execute `test.py`
