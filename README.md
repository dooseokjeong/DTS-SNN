# DTS-SNN
This repository is the official implementation of **DTS-SNN: Spiking Neural Networks with Dynamic Time-Surfaces** by Donghyung Yoo, Seongsik Park and Doo Seok Jeong.

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train a DTS-SNN with zero sum temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```train
python main.py --dataset DVS128-Gesture --temporal_kernel ktzs --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode train
python main.py --dataset N-Cars --temporal_kernel ktzs --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode train
python main.py --dataset SHD --temporal_kernel ktzs --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode train
```

To train a DTS-SNN with single exponential temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```train
python main.py --dataset DVS128-Gesture --temporal_kernel kt --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode train
python main.py --dataset N-Cars --temporal_kernel kt --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode train
python main.py --dataset SHD --temporal_kernel kt --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode train
```

## Evaluation

To evaluate a DTS-SNN with zero sum temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```evaluation
python main.py --dataset DVS128-Gesture --temporal_kernel ktzs --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode eval
python main.py --dataset N-Cars --temporal_kernel ktzs --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode eval
python main.py --dataset SHD --temporal_kernel ktzs --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode eval
```

To evaluate a DTS-SNN with single exponential temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```evaluation
python main.py --dataset DVS128-Gesture --temporal_kernel kt --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode eval
python main.py --dataset N-Cars --temporal_kernel kt --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode eval
python main.py --dataset SHD --temporal_kernel kt --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode eval
```

## Results
Our model achieves the following performance on: 

- DVS128-Gesture dataset

| Method                                            | Network                    | Accuracy (%) |
| ------------------------------------------------- | -------------------------- | ------------ |
| DTS-SNN (with zero sum temporal kernel)           | 1600-400-11 (FCN)          | 95.6%        |
| DTS-SNN (with single exponetial temporal kernel)  | 1600-400-11 (FCN)          | 92.82%       |

- N-Cars dataset

| Method                                            | Network                    | Accuracy (%) |
| ------------------------------------------------- | -------------------------- | ------------ |
| DTS-SNN (with zero sum temporal kernel)           | 3000-400-2 (FCN)           | 90.28%       |
| DTS-SNN (with single exponetial temporal kernel)  | 3000-400-2 (FCN)           | 89.47%       |

- SHD dataset

| Method                                            | Network                    | Accuracy (%) |
| ------------------------------------------------- | -------------------------- | ------------ |
| DTS-SNN (with zero sum temporal kernel)           | 350-128-2 (FCN)            | 77.99%       |
| DTS-SNN (with single exponetial temporal kernel)  | 350-128-2 (FCN)            | 66.30%       |
