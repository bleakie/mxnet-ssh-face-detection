基于mxnet的ssh人脸检测算法（改进版）

**`2019.02.27`**: 初始版本，优化误检


# 1.介绍

1. Reproduce [SSH](https://arxiv.org/abs/1708.03979) (Single Stage Headless Face Detector) with MXNet.

2. Original Caffe code: [https://github.com/deepinsight/insightface/tree/master/SSH]

3. Evaluation on WIDER FACE（原版的结果，改进后的没做测试，应该会更好）:

| Impelmentation     | Easy-Set | Medium-Set | Hard-Set |
| ------------------ | -------- | ---------- | -------- |
| Original Caffe SSH | 0.93123  | 0.92106    | 0.84582  |
| Our SSH Model      | 0.93489  | 0.92281    | 0.84525  |

4. Evaluation on fddb = 98.7%

![Identification results on fddb](https://github.com/bleakie/mxnet-ssh-face-detection/blob/master/image/result/discROC.png)

5. 优化原始版本边缘人脸漏检
![Detection results](https://github.com/bleakie/mxnet-ssh-face-detection/blob/master/image/result/demo_res.jpg)

# 2.安装

## 环境

ubuntu16.04 cuda cudnn mxnet以及python的依赖项等

## 配置

1. Type `make` to build necessary cxx libs（需要更改python版本时需在Makefile修改对应py的版本）

2. Download MXNet VGG16 pretrained model from [here](http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params) and put it under `model` directory.

3. 编译，在rcnn/config.py里修改参数配置

```
config.BBOX_MASK_THRESH = 20 #add mask with in train for little size faces
# config.COLOR_JITTERING = 0.125
config.COLOR_JITTERING = 0 # add augmentation for bright and so on

config.TEST.SCORE_THRESH = 0.5

# scale changed as smallhard face
config.TEST.SCALES = [50, 500, 1000]
config.TEST.PYRAMID_SCALES = [0.75, 1.5]

default.base_lr = 0.004
default.e2e_epoch = 40
```

# 3.Training

```
python train_ssh.py
```
```
算法对代码中blur>1, invalid>0, occlusion>1的人脸都加上mask，这样会减少误捡，但是同样造成漏检
```

## 人脸识别数据集组成

    .
    ├── WIDER_train
    |   └── images
    │       ├── .... 
    │       ├── .... 
    └── ...





