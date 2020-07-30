import torch

import librosa
import numpy as np

from scipy.io import wavfile
from config import args


def wav2spec(filename, norm=False):
    sr, wave_data = wavfile.read(filename)
    wave_data = wave_data.astype('float')
    wave_data = wave_data / np.max(abs(wave_data))
    D = librosa.stft(wave_data,
                     n_fft=args.n_fft,
                     hop_length=args.hop_length,
                     win_length=args.win_length,
                     window=args.window)

    try:
        S = 2 * np.log10(abs(D))
    except:
        print('D == 0')
        S = 2 * np.log10(abs(D) + args.EPSILON)

    phase = np.exp(1j * np.angle(D))
    if norm:
        mean = np.mean(S, axis=1).reshape(args.FBIN, 1)
        std = np.std(S, axis=1).reshape(args.FBIN, 1)

        if (std == 0).any():
            print('std == 0')
            std += args.EPSILON

        S = (S - mean) / std
    else:
        mean = 0
        std = 1

    return S, phase, mean, std


def spec2wav(S, phase, mean=0, std=1):
    D = np.multiply(10**((S * std + mean) / 2), phase)
    return librosa.istft(D,
                         hop_length=args.hop_length,
                         win_length=args.win_length,
                         window=args.window)


def to_TMHINT_name(sample):
    return f'TMHINT_{(sample-1)//10+1:02d}_{(sample-1)%10+1:02d}'


def cache_clean_data(elec_preprocessor=None, is_training=True, force_update=False, device=args.device):
    '''
    elec_preprocessor:
      Specify how many channels to use. Can be a 2d-tuple or, int value or None.
      Example:
        elec_channel = (1, 124)
        elec_channel = 45

      Default: Not used.

    is_training:
      [True] clean wave data in tensor form with shape: (1, 1, wave signal length)
      [False] clean wave data in numpy form with shape: (wave signal length)

      clean electrical data always in tensor form with shapw: (1, [electrodes/hidden], wave signal length)
    '''
    global dataset

    if isinstance(elec_preprocessor, int):
        elec_channel = (1, elec_preprocessor)
        hidden_size = elec_channel[1] - elec_channel[0] + 1

    elif isinstance(elec_preprocessor, tuple) and len(elec_preprocessor) == 2:
        elec_channel = elec_preprocessor
        hidden_size = elec_channel[1] - elec_channel[0] + 1

    elif elec_preprocessor is not None:
        raise TypeError('Unknown type: elec_preprocessor')

    if not force_update and 'dataset' in globals() and (
        (elec_preprocessor is None) or
            ('elec' in dataset and hidden_size == dataset['elec'][1].size(2))):
        return dataset

    # ===== Initialize dataset =====
    dataset = {'spec': [0]}
    if elec_preprocessor is not None:
        dataset['elec'] = [0]

    # ===== Select [Train/Test] directory =====
    if is_training:
        data_range = range(1, 251)
        data_dir = f'./Dataset/Training/Clean/'
    else:
        data_range = range(1, 71)
        data_dir = f'./Dataset/Testing/Clean/a1/'

    # ===== For all samples =====
    for sample in data_range:
        if sample == 103:
            # broken data
            dataset['spec'].append(0)
            if 'elec' in dataset:
                dataset['elec'].append(0)
            continue

        # ===== Load wave data and nomarlize =====
        if is_training:
            sample_name = f'{data_dir}{sample}.wav'
        else:
            sample_name = f'{data_dir}{to_TMHINT_name(sample)}.wav'

        Sy, phasey, _, _ = wav2spec(sample_name)

        if is_training:
            # tensor shape: (1, 1, wave signal length)
            dataset['spec'].append(torch.Tensor([Sy.T]).to(device))
        else:
            # numpy shape: (wave signal length)
            dataset['spec'].append(spec2wav(Sy, phasey))

        if 'elec' in dataset:
            # ===== Load electrical data =====
            if is_training:
                filename = f'./raw data/split electro/E{sample+70:03d}.csv'
            else:
                filename = f'./raw data/split electro/E{sample:03d}.csv'
            elec_data = np.genfromtxt(
                filename, delimiter=',', dtype=np.float32)

            # ===== Extract channels and time shift =====
            # final numpy shape: (electrodes, signal length)
            elec_data = elec_data[:Sy.shape[1] - args.shift,
                                  elec_channel[0]:elec_channel[1] + 1]
            elec_data = np.vstack(
                [np.zeros((args.shift, elec_data.shape[1])), elec_data])

            # ===== Use auto_encoder compress electrical data =====
            # final numpy shape: (hidden, elec signal length)
#             if auto_encoder:
#                 with torch.no_grad():
#                     # numpy shape: (electrodes, elec signal length) => tensor shape: (elec signal length, 1, hidden)
#                     elec_data = auto_encoder.encoder(torch.Tensor(elec_data.T).unsqueeze(1).to(args.device))

#                     # tensor shape: (elec signal length, 1, hidden) => numpy shape: (hidden, elec signal length)
#                     elec_data = elec_data.squeeze(dim=1).cpu().detach().numpy().T

            # ===== Resample to 16k =====
            # final numpy shape: ([electrodes/hidden], wave signal length)
#             elec_data = cv2.resize(elec_data, dsize=(wave_data.shape[0], elec_data.shape[0]),
#                                    interpolation=cv2.INTER_CUBIC)
#                                    interpolation=cv2.INTER_NEAREST)

            # final numpy shape: (1, [electrodes/hidden], wave signal length)
            dataset['elec'].append(torch.Tensor([elec_data]).to(device))

            # final numpy shape: ([electrodes/hidden], wave signal length)
#             dataset['elec'].append(elec_data)
    return dataset


def load_data(noise_type, sample, norm=False, is_training=True):
    '''
    return (noisy wave, [elec/None], clean wave) in Tensor form
    '''
    if is_training:
        noisy_file = f'./Dataset/Training/Noisy/{noise_type}/{sample}.wav'
    else:
        noisy_file = f'./Dataset/Testing/Noisy/{noise_type}/{to_TMHINT_name(sample)}.wav'

    Sx, phasex, _, _ = wav2spec(noisy_file, norm)

    if 'elec' in dataset:
        return (Sx, phasex, dataset['elec'][sample], dataset['spec'][sample])

    return (Sx, phasex, None, dataset['spec'][sample])
