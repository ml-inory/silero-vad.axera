# -*- coding:utf-8 -*-
# @FileName  :silero.py
# @Time      :2024/11/6 21:30
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import torch
from torch import nn
import numpy as np


class STFT(nn.Module):

    def __init__(self, ):
        super(STFT, self).__init__()
        self.filter_length = 256
        self.padding = nn.ReflectionPad1d((0, 64))
        self.forward_basis_buffer = nn.Conv1d(in_channels=1, out_channels=258, kernel_size=256, stride=128, padding=0,
                                              bias=False)

    def transform_(self, input_data):
        x = self.padding(input_data).unsqueeze(1)
        x = self.forward_basis_buffer(x)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = x[:, :cutoff, :6]
        imag_part = x[:, cutoff:, :6]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        return magnitude

    def forward(self, x):
        return self.transform_(x)
    
    def forward_export(self, x):
        x = x.unsqueeze(1)
        x = self.forward_basis_buffer(x)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = x[:, :cutoff, :6]
        imag_part = x[:, cutoff:, :6]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        return magnitude


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Encoder, self).__init__()
        self.reparam_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.reparam_conv(x)
        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Conv1d(128, 1, kernel_size=1, padding=0, dilation=1),
                                     nn.Sigmoid())

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state
    
    def forward_export(self, x, state):
        x = x.squeeze(-1)
        h, c = self.rnn(x, (state[0], state[1]))

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state


class SileroVAD(nn.Module):
    def __init__(self):
        super(SileroVAD, self).__init__()
        self.stft = STFT()
        self.encoder = nn.Sequential(Encoder(in_channels=129, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     Encoder(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     Encoder(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     Encoder(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.decoder = Decoder()

    def reset_states(self, batch_size=1):
        self._state = torch.zeros([0])
        self._context = torch.zeros([0])
        self._last_sr = 0
        self._last_batch_size = 0

    def forward(self, data, sr):

        num_samples = 512 if sr == 16000 else 256

        if data.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {data.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = data.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, data], dim=1)

        x = self.stft(x)
        print(f"stft.size() = {x.size()}")
        x = self.encoder(x)
        x, self._state = self.decoder(x, self._state)


        self._context = data[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return x


class SileroVADforExport(nn.Module):
    def __init__(self):
        super(SileroVADforExport, self).__init__()
        self.stft = STFT()
        self.encoder = nn.Sequential(Encoder(in_channels=129, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     Encoder(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     Encoder(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     Encoder(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.decoder = Decoder()

    def reset_states(self, batch_size=1):
        pass

    def forward(self, data, state):
        # x = torch.cat([context, data], dim=1)

        x = self.stft.forward_export(data)
        # print(f"stft.size() = {x.size()}")
        x = self.encoder(x)
        x, next_state = self.decoder.forward_export(x, state)

        return x, next_state

    
class SileroVADModel(nn.Module):
    def __init__(self):
        super(SileroVADModel, self).__init__()
        self._model = SileroVAD()

    def forward(self, x, sample_rate=16000):
        x = self._model(x, sample_rate)
        return x.squeeze(-1).mean()

    def reset_states(self, batch_size=1):
        self._model.reset_states(batch_size)


class SileroVADModelforExport(nn.Module):
    def __init__(self):
        super(SileroVADModelforExport, self).__init__()
        self._model = SileroVADforExport()

    def forward(self, x, state):
        x, next_state = self._model(x, state)
        return torch.mean(x.squeeze(-1), (0,1), keepdim=True), next_state

    def reset_states(self, batch_size=1):
        self._model.reset_states(batch_size)


if __name__ == '__main__':
    jit_model = torch.jit.load("./silero_vad.jit")
    jit_model.eval()
    state_dict = jit_model.state_dict()
    state_dict['_model.stft.forward_basis_buffer.weight'] = state_dict['_model.stft.forward_basis_buffer']

    batch_size = 1
    sr = 16000
    hidden_size = 128
    context_size = 64 if sr == 16000 else 32
    context = torch.zeros(batch_size, context_size)
    state = torch.zeros(2, batch_size, hidden_size)
    num_samples = 512 if sr == 16000 else 256

    model = SileroVADModelforExport()
    model.eval()
    model.load_state_dict(state_dict, strict=False)
    model.reset_states()

    pth_model = SileroVADModel()
    pth_model.eval()
    pth_model.load_state_dict(state_dict, strict=False)
    pth_model.reset_states()

    for i in range(10):
        # Perform forward pass
        input_tensor = torch.randn(1, num_samples)  # Sample input (batch_size=10, feature_dim=256)

        # output, state = model(input_tensor, state, context)
        # context = input_tensor[..., -context_size:]

        jit_output = jit_model(input_tensor, sr)

        pth_model(input_tensor)

        # print(torch.allclose(output, jit_output))