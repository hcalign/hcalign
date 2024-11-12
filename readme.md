# Hierarchical Concept Alignment for Text-Video Retrieval: \\ More Levels of Alignment Lead to More Efficient Data Utilizatio

This is an official pytorch implementation of paper "Hierarchical Concept Alignment for Text-Video Retrieval: \\ More Levels of Alignment Lead to More Efficient Data Utilizatio".  This repository provides code and pretrained weights to facilitate the reproduction of the paper's results.

## 1. Install
```
# From CLIP
conda create --name hcalign python=3.8
conda activate hcalign
conda install --yes -c pytorch pytorch=1.7.1 torchvision==0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
```


## 2. Dataset Preparation
We use `preprocess/compress_video.py` to rescale origin videos to a short size of 224.
We organize the data catagory as follows:
```
data
|-- MSRVTT
|   |-- MSRVTT_JSFUSION_test.csv
|   |-- MSRVTT_data.json
|   |-- MSRVTT_train.7k.csv
|   |-- MSRVTT_train.9k.csv
|   |-- videos_compressed
|-- WebVid
|   |-- train_0.5M.json
|   |-- train_1M.json
|   |-- train_2M.json
|   |-- videos_compressed
|   |-- videos
```


## 3. Training
```
bash scripts/run_hcalign_msrvtt_vit32.sh
```

## 4. Inference
+ The pretrained weights can be downloaded at [BaiduYun](https://pan.baidu.com/s/1NXU3bNpaVwpa0ucuTOB8eQ)(password: jgnk)
```
bash scripts/test_hcalign_msrvtt_vit32.sh
```


## Acknowledge
* Our code is based on [ArrowLuo/CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) and [facebookresearch/swav](https://github.com/facebookresearch/swav).

