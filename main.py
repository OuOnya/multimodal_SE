import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import wavfile
# from pesq import pesq
# from pystoi import stoi
from matplotlib import cm

from config import args
from model.model import *
from preprocessing import *


def training(model, loss_fn, optimizer, filename):
    try:
        os.makedirs(args.SCNN_checkpoint_path)
    except:
        pass

    saved_epoch = 0
    min_valid_loss = 1e10
    loss_hist = {'train loss': [], 'valid loss': []}
    with tqdm(range(1, args.epoch + 1)) as pbar1, \
            tqdm(total=len(args.train_noise_type)) as pbar2, \
            tqdm(total=len(args.train_noise_type)) as pbar3:
        for epoch in pbar1:
            loss_hist['train loss'] = []

            pbar2.reset()
            for n, noise_type in enumerate(args.train_noise_type):
                bs = 0
                loss = 0

                pbar2.set_description_str(
                    f'(Epoch {epoch}) noise type: {noise_type}')
                for sample in range(1, 224):
                    if sample == 103:
                        continue

                    Sx, _, elec, clean = load_data(noise_type, sample)
                    noisy = torch.Tensor([Sx.T]).to(args.device)

                    enhan = model(noisy, elec)
                    loss += model.get_loss(loss_fn, enhan, clean)
                    bs += 1
                    if bs >= args.batch_size:
                        loss /= bs
                        loss_item = loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loss_hist['train loss'].append(loss_item)
                        loss = 0
                        bs = 0

                    pbar2.refresh()
                    pbar1.refresh()

                if bs != 0:
                    loss /= bs
                    loss_item = loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_hist['train loss'].append(loss_item)
                    loss = 0
                    bs = 0

                pbar2.set_postfix(loss=loss_item)
                pbar2.update()

            # ===== Validation =====
            pbar3.reset()
            valid_loss = 0
            valid_sample = 0
            for n, noise_type in enumerate(args.train_noise_type):
                pbar3.set_description_str(f'noise type: {noise_type}')
                for sample in range(224, 251):
                    Sx, _, elec, clean = load_data(
                        noise_type, sample, norm=model.use_norm)
                    noisy = torch.Tensor([Sx.T]).to(args.device)

                    with torch.no_grad():
                        enhan = model(noisy, elec)
                        valid_loss += model.get_loss(loss_fn,
                                                     enhan, clean).item()
                    valid_sample += 1

                    pbar3.set_postfix(valid_loss=valid_loss / valid_sample)
                    pbar1.refresh()
                pbar3.update()

            valid_loss /= valid_sample
            loss_hist['valid loss'].append(valid_loss)

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                saved_epoch = epoch
                model.save_model(args.SCNN_checkpoint_path,
                                 f'{filename} (Epoch {args.epoch}).pt')

            pbar3.set_description_str(
                f'Saved Epoch: {saved_epoch}, min valid loss: {min_valid_loss}')

            # plt.figure()
            plt.plot(loss_hist['train loss'])
            plt.plot(np.linspace(0, len(loss_hist['train loss']), len(
                loss_hist['valid loss'])), loss_hist['valid loss'])
            # plt.xlabel('iteration')
            # plt.ylabel('loss')
            plt.legend(['Train', 'Valid'])
            plt.tight_layout(pad=0.2)
            # if epoch == 1:
            #     plt.savefig(args.SCNN_checkpoint_path + f'{filename} (Epoch 1).svg')
            plt.show()


if __name__ == '__main__':
    filename = 'S_CNN+E LF cat LSTM Linear (1 loss)'
    model = S_CNN(
        use_E=True,
        # E2Spectrogram=nn.Sequential(
        #     Unsqueeze(1),
        #     nn.Conv2d(1, 16, (3, 3), stride=(1, 3), padding=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, (3, 3), stride=(1, 3), padding=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (3, 3), stride=(1, 3), padding=(1, 1)),
        #     nn.ReLU(),

        #     Reshape(64, 5),
        #     nn.Linear(64 * 5, 257),
        # ),
        E2S=nn.Sequential(
            nn.Linear(124, 257),
        ),
        is_late_fusion=True,
        fusion_type='cat',
        fusion_channel=2,
        Fusion_layer=nn.Sequential(
            nn.LSTM(514, 128, batch_first=True),
            nn.Linear(128, 257),
        ),
        # Fusion_layer=nn.Sequential(
        #     Unsqueeze(0),
        #     nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 1, (3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(),

        #     Reshape(1, 257),
        #     nn.Linear(257, 257),
        # ),
    )  # .load_model(args.SCNN_checkpoint_path, f'{filename} (Epoch 40).pt')
    model = S_CNN().load_model(args.SCNN_checkpoint_path, f'S_CNN+E V1 (Epoch 40).pt')
    model.to(args.device)

    print('Caching data ...')
    cache_clean_data(elec_preprocessor=(1, 124),
                     is_training=True, force_update=False)
    print('Cached!')

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training(model, loss_fn, optimizer, filename)
