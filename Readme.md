# LDCNet
### Learnable Descriptive Convolutional Network for Face Anti-Spoofing (BMVC'22)

## Feature maps extracted by LDC 
![Screenshot](figure/visualize_low_level.png)
Generate by `Low_level_features_visualization.py` 

## Learnable Descriptive Convolution
![Screenshot](figure/LDC_illustration.png)
![Screenshot](figure/formula.png)

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
Step 1: run `Amap_train.py` to get pretrained model for producing activation map 

Step 2: run `train.py` to train LDCNet

## Testing
run `test.py`

## More visualization: Feature maps extracted by LDC
### Live images:
![Screenshot](low_level_visualize/0001_image_live.png)
![Screenshot](low_level_visualize/0001_featmap.png)
![Screenshot](low_level_visualize/0002_image_live.png)
![Screenshot](low_level_visualize/0002_featmap.png)
### Print images (grid artifacts):
![Screenshot](low_level_visualize/0003_image_print.png)
![Screenshot](low_level_visualize/0003_featmap.png)
![Screenshot](low_level_visualize/0004_image_print.png)
![Screenshot](low_level_visualize/0004_featmap.png)
### Replay images (moiré patterns):
![Screenshot](low_level_visualize/0005_image_replay.png)
![Screenshot](low_level_visualize/0005_featmap.png)
![Screenshot](low_level_visualize/0006_image_replay.png)
![Screenshot](low_level_visualize/0006_featmap.png)

## Citation

If you use the LDC, please cite the paper:

```
@article{huang2022learnable,
  title={Learnable Descriptive Convolutional Network for Face Anti-Spoofing},
  author={Huang, Pei-Kai and Ni, Hui-Yu and Ni, Yan-Qin and Hsu, Chiou-Ting},
  year={2022}
}
```
