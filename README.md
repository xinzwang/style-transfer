# Style Transfer

Simple demo for paper: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://link.springer.com/content/pdf/10.1007/978-3-319-46475-6_43.pdf)

[1] Johnson J, Alahi A, Fei-Fei L. Perceptual losses for real-time style transfer and super-resolution[C]//European conference on computer vision. Springer, Cham, 2016: 694-711.

## Introduction

Style Transfer is a very interesting task. We can transfer the style of one famous painting to any picture to give them a wonderful visual impression.There are many famous paintings, such as **The Starry Night(Van Gogh, 1889)**, **Sunrise(Monet, 1872)**, **Great Wave(Hokusai, 1831)**, etc. All these paintings are very stylish in composition and color. 

In this task, we only need one target image to complete the style transfer task. We trained three style transfer models on coco dataset, and achieved very good visual effects.

## Visual

![](docs/imgs/visual.png)

## Usage

Transfered images will be saved during training

```bash
# train
python train.py --style_img=night.jpg --data_path=imgs/ 
```
