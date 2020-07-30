import torch

import scipy
import scipy.signal

from argparse import Namespace

args = Namespace(
    split_ratio=0.2,
    epoch=40,
    batch_size=1,
    kernel=3,
    lr=0.0001,
    shift=12,
    SCNN_checkpoint_path=f'checkpoint/S_CNN/',

    EPSILON=1e-12,
    n_fft=512,
    hop_length=160,
    win_length=512,
    window=scipy.signal.hamming,
    sample_rate=16000,
)
args.FBIN = args.n_fft // 2 + 1
args.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

args.train_noise_type = [
    'cafeteria_babble',
    'crowd-party-adult-med',
    *[f'n{i}' for i in range(1, 101)],
    'street noise',
    'street noise_downtown',
]

args.test_noise_type = [
    'car_noise_idle_noise_60_mph',
    'engine',
    'pinknoise_16k',
    'street',
    'street noise',
    'taiwan_3talker',
    'white',
]

args.test_SNR_type = [
    'n10dB', 'n7dB', 'n6dB', 'n5dB', 'n3dB', 'n1dB', '0dB',
    '1dB', '3dB', '4dB', '5dB', '6dB', '9dB', '10dB', '15dB',
]
