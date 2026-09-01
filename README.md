# EOD: Explainable outlier generation for Out-of-distribution Detection
This is the source code accompanying the paper **[*EOD*]**

## Informations
- python : 3.9.21
- CUDA : 11.3

## Requirements
```python
pip install -r requirements.txt
```
The related version for PyTorch and related library (e.g. torchvision) is based on cu113. Please adjust it according to the experimental environment.

## Executing code
```python
python train.py --data CIFAR10 \
                --model wrn \
                --start_epoch 80 \
                --thres 0.3 \
                --scale 5
```
**Explanation about arguments**
- --data: ID dataset.
- --model: Backbone architecture.
- --start_epoch: Start epoch for generate virtual outlier.
- --thres: Threshold for distinguish semantic and nuisance region by CAM.
- --scale: Adjust magnitude for FGSM-like virtual outlier generation.

![Framework](./fig/framework.png)

## Acknowledgement
This repository was developed with support from the **서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단** - http://dsplus.uos.ac.kr/
