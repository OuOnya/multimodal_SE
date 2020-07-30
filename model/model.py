import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class Reshape(nn.Module):
    def __init__(self, channel, height):
        super(Reshape, self).__init__()
        self.channel = channel
        self.height = height

    def forward(self, x):
        return x.permute(0, 2, 3, 1).reshape(1, -1, self.channel * self.height)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class S_CNN(nn.Module):
    def __init__(self, use_E=False, E2S=None, Fusion_layer=None, is_late_fusion=False, fusion_type='cat', fusion_channel=0):
        '''
        elec_channel: 2d-tuple or AutoEncoder
        early_fusion: Boolean
        '''
        super(S_CNN, self).__init__()
        self.use_E = use_E

        _in_channel = 2 if use_E and not is_late_fusion and fusion_type in 'concatenate' else 1

        self.S2S = self.__init_CNN(_in_channel)

        if self.use_E:
            if isinstance(E2S, nn.Module):
                self.E2S = E2S
            else:
                self.E2S = nn.Linear(124, 257)

        self.Fusion_layer = Fusion_layer
        self.is_late_fusion = is_late_fusion
        self.fusion_type = fusion_type
        self.fusion_channel = fusion_channel

    def __init_CNN(self, in_channel):
        return nn.Sequential(
            Unsqueeze(0),
            nn.Conv2d(in_channel, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=(1, 3), padding=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 3), padding=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 3), padding=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 3), padding=(1, 1)),
            nn.ReLU(),

            Reshape(128, 4),
            nn.Linear(512, 128),
            nn.Linear(128, 257),
        )

    def forward(self, s, e=None):
        '''
            return (Spectral Enhancement, Electronic Enhancement, Fusion Enhancement)
        '''
        if self.use_E and e is not None:
            h_e = self.E2S(e)

            if isinstance(self.E2S, nn.LSTM):
                h_e = h_e[0]

            if not self.is_late_fusion:
                if s is not None:
                    if self.fusion_type in 'concatenate':
                        h = torch.cat((s, h_e), dim=self.fusion_channel)

                    elif self.fusion_type in 'mean':
                        h = (s + h_e) / 2
                        h_e = h
                    elif self.fusion_type in 'mask':
                        h = torch.mul(s, h_e)
                        h_e = h

                return None, h_e, self.S2S(h)
            else:
                h_s = self.S2S(s)

                if self.fusion_type in 'concatenate':
                    h = torch.cat((h_s, h_e), dim=self.fusion_channel)

                elif self.fusion_type in 'mean':
                    h = (h_s + h_e) / 2
                elif self.fusion_type in 'mask':
                    h = torch.mul(h_s, h_e)
                    h_e = h

                for layer in self.Fusion_layer:
                    h = layer(h)
                    if isinstance(layer, nn.LSTM):
                        h = h[0]

                return h_s, h_e, h

        else:
            return None, None, self.S2S(s)

    def get_loss(self, loss_fn, pred_y, true_y):
        return loss_fn(pred_y[2], true_y)
#         loss = 0
#         for _y in pred_y:
#             if _y is not None:
#                 loss += loss_fn(_y, true_y)
#         return loss

    def load_model(self, checkpoint_path, filename, device='cuda' if torch.cuda.is_available() else 'cpu'):
        try:
            state_dict = torch.load(
                os.path.join(checkpoint_path, filename),
                map_location=device
            )

            self.use_E = state_dict.get('use_E', False)
            if self.use_E:
                if 'E2S' in state_dict:
                    self.E2S = state_dict['E2S']

                else:
                    self.E2S = nn.Linear(124, 257)

            self.is_late_fusion = state_dict.get(
                'is_late_fusion', self.is_late_fusion)
            self.fusion_type = state_dict.get('fusion_type', self.fusion_type)
            self.fusion_channel = state_dict.get(
                'fusion_channel', self.fusion_channel)

            _in_channel = 2 if self.use_E and not self.is_late_fusion and self.fusion_type in 'concatenate' else 1
            self.S2S = self.__init_CNN(_in_channel)

            if 'S2S_state_dict' in state_dict:
                self.S2S.load_state_dict(
                    state_dict['S2S_state_dict'], strict=False)

            if 'Fusion_layer' in state_dict:
                self.Fusion_layer = state_dict['Fusion_layer']

        except Exception as e:
            print(f'Error! Can not load model {filename}.')
            print(e)

        return self

    def save_model(self, checkpoint_path, filename):
        state_dict = {
            'S2S_state_dict': self.S2S.state_dict(),
            'use_E': self.use_E,
            'is_late_fusion': self.is_late_fusion,
            'fusion_type': self.fusion_type,
            'fusion_channel': self.fusion_channel,
        }

        if self.use_E:
            state_dict['E2S'] = self.E2S
        if self.is_late_fusion:
            state_dict['Fusion_layer'] = self.Fusion_layer

        torch.save(state_dict, os.path.join(checkpoint_path, filename))
