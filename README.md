# Multilabel-Classification_CNN_and_Vision-transform

## 📌 Overview
This repository implements an Image Multi-Label Classification model, which can identify multiple labels in a single image. Unlike traditional image classification, where an image is assigned a single category, multi-label classification allows an image to have multiple labels simultaneously.

## 🔍 Difference Between Image Classification and Multi-Label Classification
![image](https://github.com/KaushiML3/Multilabel-Classification_CNN_and_Vision-Transformer/blob/main/src_img/Screenshot%20(90).png)



## 🚀 Features

- Uses a deep learning model (CNN, ResNet,VIT etc.) for multi-label classification.
- Implements Binary Cross-Entropy (BCE) Loss to handle multiple labels per image.
- Data preprocessing and augmentation for better generalization.
- Evaluation metrics: F1-score, Precision, Recall, mAP (mean Average Precision).
- Supports custom datasets and pretrained models for fine-tuning.


## Dataset [Link](https://www.kaggle.com/datasets/kaiska/apparel-dataset/data)
The dataset used for training contains 16170 images of  8 different clothing categories in 9 different colours.

This repository contains resources for training a deep learning model to multi-label classification. It includes two main Jupyter notebooks for model training, each implementing a distinct architecture:

1. [Custom Architecture Notebook]() : This notebook demonstrates the use of a custom-built neural network architecture tailored specifically for multi-label classification. Designed for flexibility and simplicity, the custom architecture allows for experimentation and adaptation to varying datasets.
Model inference:
![image](https://github.com/KaushiML3/Multilabel-Classification_CNN_and_Vision-Transformer/blob/main/src_img/cnn.png)

3. [VIT Architecture Notebook]() : In this notebook, I'll show how one can fine-tune any pre-trained vision model from the Transformers library for multi-label image classification.[refference](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Fine_tuning_SigLIP_and_friends_for_multi_label_image_classification.ipynb)
Model inference:
![image](https://github.com/KaushiML3/Multilabel-Classification_CNN_and_Vision-Transformer/blob/main/src_img/sliglip.png)
