# DTS-SNN
This repository is the official implementation of DTS-SNN: Spiking Neural Networks with Dynamic Time-Surfaces.

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train a DTS-SNN with zero sum temporal kernel ($$k_{tzs}$$) on DVS128-Gesture or N-Cars or SHD, run this command:
```train
python main.py --dataset DVS128-Gesture --temporal_kernel ktzs --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode train
python main.py --dataset N-Cars --temporal_kernel ktzs --dt 1 --T 100 --batch_size 64 --gpu 0 1 --mode train
python main.py --dataset SHD --temporal_kernel ktzs --dt 1 --T 500 --batch_size 256 --gpu 0 1 --mode train
```

To train a DTS-SNN with single exponential temporal kernel ($k_t$) on DVS128-Gesture or N-Cars or SHD, run this command:
```train
python main.py --dataset DVS128-Gesture --temporal_kernel kt --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode train
python main.py --dataset N-Cars --temporal_kernel kt --dt 1 --T 100 --batch_size 64 --gpu 0 1 --mode train
python main.py --dataset SHD --temporal_kernel kt --dt 1 --T 500 --batch_size 256 --gpu 0 1 --mode train
```

## Evaluation

To evaluate a DTS-SNN with zero sum temporal kernel ($k_tzs$) on DVS128-Gesture or N-Cars or SHD, run this command:
```evaluation
python main.py --dataset DVS128-Gesture --temporal_kernel ktzs --batch_size 16 --mode eval
python main.py --dataset N-Cars --temporal_kernel ktzs --dt 1 --T 100 --batch_size 64 --mode eval
python main.py --dataset SHD --temporal_kernel ktzs --dt 1 --T 500 --batch_size 256 --mode eval
```

To evaluate a DTS-SNN with single exponential temporal kernel ($k_t$) on DVS128-Gesture or N-Cars or SHD, run this command:
```evaluation
python main.py --dataset DVS128-Gesture --temporal_kernel kt --batch_size 16 --mode eval
python main.py --dataset N-Cars --temporal_kernel kt --dt 1 --T 100 --batch_size 64 --mode eval
python main.py --dataset SHD --temporal_kernel kt --dt 1 --T 500 --batch_size 256 --mode eval
```

## Results
Our model achieves the following performance on: 

DVS-128 dataset
| Method                | Network                    | Accuracy (%) |
| --------------------- |--------------------------- | ------------ |
| DTS-SNN with ($k_tzs$)| 1600-400-11 (MLP)          | 95.6%        |
| DTS-SNN with ($k_t$)  | 1600-400-11 (MLP)          | 92.82%       |

N-Cars dataset
| Method                | Network                    | Accuracy (%) |
| --------------------- |--------------------------- | ------------ |
| DTS-SNN with ($k_tzs$)| 3000-400-2 (MLP)           | 90.28%       |
| DTS-SNN with ($k_t$)  | 3000-400-2 (MLP)           | 89.47%       |

SHD dataset
| Method                | Network                    | Accuracy (%) |
| --------------------- |--------------------------- | ------------ |
| DTS-SNN with ($k_tzs$)| 350-128-2 (MLP)            | 77.99%       |
| DTS-SNN with ($k_t$)  | 350-128-2 (MLP)            | 66.30%       |
