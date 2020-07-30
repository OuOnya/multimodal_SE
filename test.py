import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import os
import librosa
import scipy
import numpy as np
# import librosa.display
import matplotlib.pyplot as plt

# from IPython.display import Audio, display
from tqdm import tqdm
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
from matplotlib import cm
from argparse import Namespace
from multiprocessing import Pool
from collections import OrderedDict

from config import args
from preprocessing import *
from model.model import *

# args.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


def test(model, noise_type, SNR_type, test_sample, old_version=False):
    print(f'{noise_type}, {SNR_type}')

    Sx, phasex, meanx, stdx = wav2spec(
        f'./Dataset/Testing/Noisy/{noise_type}/a1/{SNR_type}/{to_TMHINT_name(test_sample)}.wav'
    )
    noisy = torch.Tensor([Sx.T]).to(args.device)

    Sy, phasey, _, _ = wav2spec(
        f'./Dataset/Testing/Clean/a1/{to_TMHINT_name(test_sample)}.wav')

    if model.use_E:
        elec_data = np.genfromtxt(
            f'./raw data/split electro/E{test_sample:03d}.csv', delimiter=',', dtype=np.float32)
        elec_data = elec_data[:Sy.shape[1] - args.shift, 1:]
        elec_data = np.vstack(
            [np.zeros((args.shift, elec_data.shape[1])), elec_data])
        elec = torch.Tensor([elec_data]).to(args.device)
    else:
        elec = None

    with torch.no_grad():
        Ss, Se, Sy_ = model(noisy, elec)

    if Ss is not None:
        Ss = Ss[0].cpu().detach().numpy().T
    if Se is not None:
        Se = Se[0].cpu().detach().numpy().T
    Sy_ = Sy_[0].cpu().detach().numpy().T

    if noisy is not None:
        enhanced = spec2wav(Sy_, phasex)
    else:
        enhanced = librosa.core.griffinlim(
            10**(Sy_ / 2), n_iter=5, hop_length=args.hop_length, win_length=args.win_length, window=args.window)
    clean = spec2wav(Sy, phasey)
    noisy = spec2wav(Sx, phasex, meanx, stdx)

    sr = args.sample_rate
    print('PESQ: ', pesq(sr, clean, enhanced, 'wb'))
    print('STOI: ', stoi(clean, enhanced, sr, False))
    print('ESTOI:', stoi(clean, enhanced, sr, True))

    # display(Audio(f'../Dataset/Testing/Clean/a1/{to_TMHINT_name(test_sample)}.wav', rate=args.sample_rate, autoplay=False))
    # display(Audio(clean, rate=args.sample_rate, autoplay=False))
    # display(Audio(noisy, rate=args.sample_rate, autoplay=False))
    # display(Audio(enhanced, rate=args.sample_rate, autoplay=False))

    f, axes = plt.subplots(6, 1, sharex=True, figsize=(18, 12))
#     plt.figure(figsize=(18, 3))
#     librosa.display.waveplot(noisy, sr=args.sample_rate)
    axes[0].set_xlim(0, Sy.shape[1])

    axes[0].imshow(Sx, origin='l', aspect='auto', cmap='jet')
    if model.use_E:
        axes[1].imshow(elec_data.T, aspect='auto', cmap=cm.Blues)
    if Ss is not None:
        axes[2].imshow(Ss, origin='l', aspect='auto', cmap='jet')
    if Se is not None:
        axes[3].imshow(Se, origin='l', aspect='auto', cmap='jet')
    axes[4].imshow(Sy_, origin='l', aspect='auto', cmap='jet')
    axes[5].imshow(Sy, origin='l', aspect='auto', cmap='jet')
    plt.tight_layout(pad=0.2)
#     plt.savefig('spectrogram.svg')
    plt.show()


if __name__ == '__main__':

    # To see the spectrogram and performance of some specific models in the test sample
    noise_type = 1
    SNR_type = 0
    sample = 10  # 1 ~ 70

    model_names = [
        # 'S_CNN (Epoch 40)',
        # 'S_CNN (Epoch 40 True)',

        'S_CNN+E V1 (Epoch 40)',

        # 'S_CNN+E EF cat (Epoch 40)',
        # 'S_CNN+E EF cat Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E EF mean (Epoch 40)',
        # 'S_CNN+E EF mean (Epoch 40 True)',
        # 'S_CNN+E EF mean Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E EF mask (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (1 loss) (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (1 loss) (Epoch 40 True)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF mean (1 loss) (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (Epoch 40 True)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF mean (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 EF mean (Epoch 40 True)',


        # 'S_CNN+E LF cat CNN (Epoch 40)',
        # 'S_CNN+E LF mask CNN (Epoch 40)',
        # 'S_CNN+E LF cat (1 loss) (Epoch 40)',
        # 'S_CNN+E LF cat Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E LF cat Linear V2 (1 loss) (Epoch 40 test)',
        # 'S_CNN+E LF mean (1 loss) (Epoch 40)',
        # 'S_CNN+E LF mean Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E LF mask (1 loss) (Epoch 40)',
        # 'S_CNN+E LF cat LSTM Linear (1 loss) (Epoch 40 test)',
        # 'S_CNN+E LF mask CNN (Epoch 40)',
        # 'S_CNN+E LF mask Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 nopad LF cat CNN (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 nopad LF cat CNN (Epoch 40 True)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF cat CNN (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF cat CNN (Epoch 40 True)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF cat (1 loss) (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF cat Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF mean Linear (1 loss) (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF mean CNN (Epoch 40)',
        # 'S_CNN+E CNN_16_32_64 stride_1_3 LF mean (1 loss) (Epoch 40)',
        # 'S_CNN+E S_CNN LF cat CNN (Epoch 40)',
        # 'S_CNN+E S_CNN LF cat CNN (Epoch 40 True)',
    ]

    for model_name in model_names:
        print(f'model_name: {model_name}')
        model = S_CNN().load_model(args.SCNN_checkpoint_path,
                                   f'{model_name}.pt', args.device)
        model.to(args.device)
        test(model, args.test_noise_type[noise_type],
             args.test_SNR_type[SNR_type], sample)
