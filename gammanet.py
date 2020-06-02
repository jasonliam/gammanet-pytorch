import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

import numpy as np
from collections import OrderedDict

from fgru import fConvGRUCell

# torch.manual_seed(42)

"""
config entries

- in_channels: input channels
- input_timeseries
- return_sequences
- num_filters (list): number of filters at each layer of the network

- conv_kernel_size (list)
- conv_blocksize (list)
- conv_normtype: 'instancenorm' or 'batchnorm'
- conv_dropout_p: None if no dropout
- conv_residual

- fgru_hidden_size (list) 
- fgru_kernel_size (list): one entry for every block (instead of every depth)
- fgru_timesteps: number of timesteps to run on input; ignored if input is timeseries
- fgru_normtype
- fgru_channel_sym
- fgru_attention_args

- upsample_mode
- upsample_all2all: whether to enable top-down connections during upsampling

"""


class GammaNet(nn.Module):
    """ 
    NOTE: no readout block, the user should add it. 
    NOTE: Linsley et al's implementation also has a few ff encoding blocks before gammanet
    """

    @staticmethod
    def _get_default_config():
        v2_big_working = {
            'in_channels': 1,
            'input_timeseries': False,
            'return_sequences': False,
            'num_filters': [24, 28, 36, 48, 64],
            'conv_kernel_size': [3, 3, 3, 3, 3],
            'conv_blocksize': [1, 1, 1, 1, 1],
            'conv_normtype': 'instancenorm',
            'conv_dropout_p': 0.2,  # 0.2
            'conv_residual': False,
            'fgru_hidden_size': [24, 28, 36, 48, 64],
            'fgru_kernel_size': [9, 7, 5, 3, 1, 1, 1, 1, 1],
            'fgru_timesteps': 2,
            'fgru_normtype': 'instancenorm',
            'fgru_channel_sym': True,
            'fgru_attention_args': {
                "type": "gala",
                "filters": 5,
                "layers": 1
            },
            'upsample_mode': 'bilinear',
            'upsample_all2all': True,
        }
        return v2_big_working

    def __init__(self, config=None):
        super().__init__()

        config = config if config is not None else self._get_default_config()
        self.config = config
        self.network_height = len(config['num_filters'])

        # downsampling blocks
        self.conv_down = nn.ModuleDict()
        self.fgru_down = nn.ModuleDict()
        self.pool = nn.ModuleDict()
        for i in range(self.network_height - 1):
            in_c = config['in_channels'] if i == 0 else config['num_filters'][i - 1]
            blk = self._conv_block(in_c, config['num_filters'][i],
                                   kernel_size=config['conv_kernel_size'][i],
                                   blocksize=config['conv_blocksize'][i],
                                   normtype=config['conv_normtype'],
                                   dropout_p=config['conv_dropout_p'],
                                   name='')
            self.conv_down[str(i)] = blk
            fgru_cell = fConvGRUCell(config['num_filters'][i],
                                     config['fgru_hidden_size'][i],
                                     config['fgru_kernel_size'][i],
                                     config['fgru_timesteps'],
                                     config['fgru_normtype'],
                                     config['fgru_channel_sym'],
                                     config['fgru_attention_args'])
            self.fgru_down[str(i)] = fgru_cell
            self.pool[str(i)] = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.conv_bottleneck = self._conv_block(config['num_filters'][-2],
                                                config['num_filters'][-1],
                                                kernel_size=config['conv_kernel_size'][-1],
                                                blocksize=config['conv_blocksize'][-1],
                                                normtype=config['conv_normtype'],
                                                dropout_p=config['conv_dropout_p'])
        self.fgru_bottleneck = fConvGRUCell(config['num_filters'][-1],
                                            config['fgru_hidden_size'][-1],
                                            config['fgru_kernel_size'][self.network_height - 1],
                                            config['fgru_timesteps'],
                                            config['fgru_normtype'],
                                            config['fgru_channel_sym'],
                                            config['fgru_attention_args'])

        # upsampling blocks
        self.upsample = nn.ModuleDict()
        self.ups_conv = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.fgru_up = nn.ModuleDict()
        for i in range(self.network_height - 2, -1, -1):  # 2nd-to-deepest to first level

            # upsampling operations
            if config['upsample_mode'] == 'transpose':
                if config['upsample_all2all']:
                    raise NotImplementedError(
                        'Transpose mode does not support all-to-all')
                self.upsample[str(i)] = nn.ConvTranspose2d(config['num_filters'][i + 1],
                                                           config['num_filters'][i],
                                                           kernel_size=2,
                                                           stride=2)
            else:
                # ups_out_dims = tuple(
                #     np.array(config['in_dims'][:2]) // (2 ** i))
                if config['upsample_all2all']:  # will concat fgru act from all layers below
                    ups_in_channels = [config['num_filters'][i + 1]]
                    for j in range(i + 1, self.network_height):
                        ups = nn.Upsample(scale_factor=2 ** (j - i),
                                          mode=config['upsample_mode'],
                                          align_corners=False)
                        self.upsample["{}-{}".format(j, i)] = ups
                        ups_in_channels += [config['num_filters'][j]]
                    ups_in_channels = sum(ups_in_channels)
                else:
                    ups = nn.Upsample(scale_factor=2,
                                      mode=config['upsample_mode'],
                                      align_corners=False)
                    self.upsample["{}-{}".format(i + 1, i)] = ups
                    ups_in_channels = config['num_filters'][i + 1]
                self.ups_conv[str(i)] = nn.Conv2d(ups_in_channels,
                                                  config['num_filters'][i],
                                                  kernel_size=1)

            # conv block
            blk = self._conv_block(config['num_filters'][i] * 2,  # concat'd skip activity
                                   config['num_filters'][i],
                                   kernel_size=config['conv_kernel_size'][i],
                                   blocksize=config['conv_blocksize'][i],
                                   normtype=config['conv_normtype'],
                                   dropout_p=config['conv_dropout_p'],
                                   name='')
            self.conv_up[str(i)] = blk

            # fgru
            fgru_cell = fConvGRUCell(config['num_filters'][i],
                                     config['fgru_hidden_size'][i],
                                     config['fgru_kernel_size'][(
                                                                        self.network_height * 2 - 2) - i],
                                     config['fgru_timesteps'],
                                     config['fgru_normtype'],
                                     config['fgru_channel_sym'],
                                     config['fgru_attention_args'])
            self.fgru_up[str(i)] = fgru_cell

    def forward(self, x):

        if self.config['input_timeseries']:
            assert len(x.shape) == 5, "Expected x in (N, T, C, H, W)"
            num_timesteps = x.shape[1]
        else:
            assert len(x.shape) == 4, "Expected x in (N, C, H, W)"
            num_timesteps = self.config['fgru_timesteps']

        # init fgru hidden states
        fgru_act = {}
        for i in range(self.network_height):
            # NOTE: init h2 as (batch, channel, height, width)
            fgru_act[i] = torch.empty((x.shape[0], self.config['num_filters'][i],
                                       x.shape[-2] // (2 ** i), x.shape[-1] // (2 ** i)))
            init.xavier_normal_(fgru_act[i])
            if torch.cuda.is_available():
                fgru_act[i] = fgru_act[i].cuda().float()
            else:
                fgru_act[i] = fgru_act[i].double()

        # downsampling activities for skip connections
        down_act = {}

        # iterate over timesteps
        act_arr = []  # activity per timestep
        for timestep in range(num_timesteps):

            if self.config['input_timeseries']:
                act = x[:, timestep]
            else:
                act = x  # fix forward drive

            # downsampling path (excludes bottleneck)
            for i in range(self.network_height - 1):
                # conv block
                act = self.conv_down[str(i)](act)
                down_act[i] = act  # skip

                # fgru
                fgru_act[i] = self.fgru_down[str(i)](
                    act, fgru_act[i], timestep)
                act += fgru_act[i]  # fgru learns residual

                # pool
                act = self.pool[str(i)](act)

            # bottleneck
            act = self.conv_bottleneck(act)
            depth = self.network_height - 1
            fgru_act[depth] = self.fgru_bottleneck(
                act, fgru_act[depth], timestep)
            act += fgru_act[depth]

            # upsampling path
            for i in range(self.network_height - 2, -1, -1):

                # upsampling
                if self.config['upsample_mode'] == 'transpose':
                    act = self.upsample["{}-{}".format(i + 1, i)](act)
                else:
                    if self.config['upsample_all2all']:
                        act = [self.upsample["{}-{}".format(i + 1, i)](act)]
                        for j in range(self.network_height - 1, i, -1):
                            ups = self.upsample["{}-{}".format(j, i)]
                            act_extra = ups(fgru_act[j])
                            act += [act_extra]
                        act = torch.cat(act, dim=1)
                    else:
                        act = self.upsample[str(i)][act]
                    act = self.ups_conv[str(i)](act)

                # concat skip connection
                act = torch.cat([act, down_act[i]], dim=1)

                # conv block
                act = self.conv_up[str(i)](act)

                # fgru
                fgru_act[i] = self.fgru_up[str(i)](act, fgru_act[i], timestep)
                act = fgru_act[i]

            # record output for current timestep
            act_arr += [act.detach()]

        if self.config['return_sequences']:
            return act, act_arr
        else:
            return act

    def _conv_block(self, in_channels, out_channels, kernel_size, blocksize,
                    normtype='batchnorm', dropout_p=None, name='conv'):
        """ 
        Helper for building a simple conv block
        - blocksize: number of conv layers in the block
        """
        block = OrderedDict()
        for i in range(blocksize):
            block["{}_conv{}".format(name, i)] = \
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size, padding=kernel_size // 2)
            if normtype == 'batchnorm':
                block["{}_bn{}".format(name, i)] = nn.BatchNorm2d(out_channels)
            elif normtype == 'instancenorm':
                block["{}_in{}".format(name, i)] = nn.InstanceNorm2d(
                    out_channels)
            if dropout_p is not None and i > 0 and i < blocksize - 1:
                block["{}_do{}".format(name, i)] = nn.Dropout(dropout_p)
            block["{}_relu{}".format(name, i)] = nn.ReLU()
        return nn.Sequential(block)


class EnsembleGammaNet(nn.Module):
    def __init__(self, ed, es):
        super(EnsembleGammaNet, self).__init__()
        self.ed = ed
        self.es = es

    def forward(self, x, frame_types):
        ed_count, es_count = sum(frame_types).int().item(), sum(~frame_types).int().item()

        if ed_count == 0:
            return self.es(x)
        elif es_count == 0:
            return self.ed(x)

        shape = x.shape
        x_ed, x_es = self.__split(x, frame_types)
        y_ed = self.ed(x_ed)
        y_es = self.es(x_es)

        return self.__merge(y_ed, y_es, frame_types, shape)

    @staticmethod
    def __split(x, frame_types):
        return x[frame_types], x[~frame_types]

    @staticmethod
    def __merge(y_ed, y_es, frame_types, shape):
        out = torch.empty(shape).cuda()
        out[frame_types] = y_ed
        out[~frame_types] = y_es
        return out


def build_gamma_net(config_data):
    gammanet_config = GammaNet._get_default_config()
    gammanet_config["input_timeseries"] = config_data['model']['timeseries']
    gammanet_config["fgru_timesteps"] = config_data['model']['fgru_timesteps']
    model = nn.Sequential(
        GammaNet(gammanet_config),
        nn.ReLU(),
        nn.BatchNorm2d(24, eps=1e-3),
        nn.Conv2d(24, 1, 5, padding=2),  # Change the expected number of output classes!
    )
    return model


def get_model(config_data):
    ensemble = config_data['model']['ensemble']
    if not ensemble:
        return build_gamma_net(config_data)
    else:
        ed_model = build_gamma_net(config_data)
        es_model = build_gamma_net(config_data)
        return EnsembleGammaNet(ed_model, es_model)
