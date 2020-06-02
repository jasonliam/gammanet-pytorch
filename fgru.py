import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from collections import OrderedDict


# torch.manual_seed(42)


class fConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, timesteps=8,
                 normtype='batchnorm', channel_sym=True, attention_args=None):
        """
        Parameters
        - input_size (tuple): shape of one input sample
        - hidden_size (int): number of hidden/output channels in layer
        - kernel_size (int): size of W kernels
        - timesteps (int): number of expected timesteps
        - normtype (bool): if set, use ReLU+normtype for each timestep; 
                           if None, use tanh+timestep weights
        - channel_sym (bool): if True, apply channel symmetric constraint for W conv (Hebbian)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.timesteps = timesteps
        self.normtype = normtype
        self.channel_sym = channel_sym

        # u1 or attention
        if attention_args is not None:
            if attention_args["type"] == 'gala':
                self.attention = GALA_Attention(self.hidden_size,
                                                self.hidden_size,
                                                attention_args["filters"],
                                                attention_args["layers"])
        else:
            self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
            init.orthogonal_(self.u1_gate.weight)
            init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
            self.u1_gate.bias.data.log()

        # u2
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        init.orthogonal_(self.u2_gate.weight)
        if attention_args is not None:
            init.uniform_(self.u2_gate.bias.data, 1, 8.0 - 1)
            self.u2_gate.bias.data.log()
            self.u2_gate.bias.data = -self.u2_gate.bias.data
        else:
            self.u2_gate.bias.data = -self.u1_gate.bias.data

        # horizontal kernels
        self.w_gate_inh = nn.Parameter(torch.empty(
            hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(
            hidden_size, hidden_size, kernel_size, kernel_size))
        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        if self.channel_sym:
            self.w_gate_inh.register_hook(lambda grad: (
                                                               grad + torch.transpose(grad, 1, 0)) * 0.5)
            self.w_gate_exc.register_hook(lambda grad: (
                                                               grad + torch.transpose(grad, 1, 0)) * 0.5)

        # scalars
        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.omega = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.omega, 0.5)
        init.constant_(self.mu, 1)
        self.params = nn.ParameterDict()
        self.params['w_inh'] = self.w_gate_inh
        self.params['w_exc'] = self.w_gate_exc
        self.params['alpha'] = self.alpha
        self.params['gamma'] = self.gamma
        self.params['kappa'] = self.kappa
        self.params['omega'] = self.omega

        # norm layers
        if normtype == 'instancenorm':
            self.norm = nn.ModuleList(
                [nn.InstanceNorm2d(hidden_size, eps=1e-03) for i in range(4 * timesteps)])
        elif normtype == 'batchnorm':
            self.norm = nn.ModuleList(
                [nn.BatchNorm2d(hidden_size, eps=1e-03) for i in range(4 * timesteps)])
            for norm in self.norm:
                init.constant_(norm.weight, 0.1)
        else:
            self.eta = nn.Parameter(torch.randn(self.timesteps, 1, 1))
            self.params['eta'] = self.eta

    def forward(self, input_, prev_state2, timestep):

        # NOTE: do NOT init h2; this is handled externally

        # import pdb; pdb.set_trace()
        i = timestep
        if self.normtype is not None:
            if "attention" in dir(self):
                g1_t = self.attention(prev_state2)
            else:
                g1_t = torch.sigmoid(
                    self.norm[i * 4 + 0](self.u1_gate(prev_state2)))
            c1_t = self.norm[i * 4 + 1](F.conv2d(prev_state2 *
                                                 g1_t, self.w_gate_inh, padding=self.padding))
            next_state1 = F.relu(
                input_ - F.relu(c1_t * (self.alpha * prev_state2 + self.mu)))
            g2_t = torch.sigmoid(self.norm[i * 4 + 2](self.u2_gate(next_state1)))
            c2_t = self.norm[i * 4 + 3](F.conv2d(next_state1,
                                                 self.w_gate_exc, padding=self.padding))
            h2_t = F.relu(self.kappa * next_state1 + self.gamma *
                          c2_t + self.omega * next_state1 * c2_t)
            prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        else:
            g1_t = torch.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t,
                            self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(
                input_ - c1_t * (self.alpha * prev_state2 + self.mu))
            g2_t = torch.sigmoid(self.norm[i * 4 + 2](self.u2_gate(next_state1)))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = torch.tanh(self.kappa * (next_state1 + self.gamma *
                                            c2_t) + (self.omega * (next_state1 * (self.gamma * c2_t))))
            prev_state2 = self.eta[timestep] * \
                          ((1 - g2_t) * prev_state2 + g2_t * h2_t)

        return prev_state2


# ===== Attention layers =====
# Source: https://github.com/serre-lab/gammanet_pytorch/blob/master/layers/fgru_base.py

class SE_Attention(nn.Module):
    """ if layers > 1  downsample -> upsample """

    def __init__(self,
                 input_size,
                 output_size,
                 filter_size,
                 layers,
                 normalization=True,
                 normalization_type='InstanceNorm2d',  # 'BatchNorm2D'
                 normalization_params={'affine': True},
                 non_linearity='ReLU',
                 norm_pre_nl=False):
        super().__init__()

        if normalization_params is None:
            normalization_params = {}

        curr_feat = input_size
        self.module_list = []

        for i in range(layers):
            if i == layers - 1:
                next_feat = output_size
            elif i < layers // 2:
                next_feat = curr_feat // 2
            else:
                next_feat = curr_feat * 2

            conv = nn.Conv2d(curr_feat, next_feat,
                             filter_size, padding=filter_size // 2)
            init.orthogonal_(conv.weight)  # xavier_normal_
            init.constant_(conv.bias, 0)
            self.module_list.append(conv)

            if non_linearity is not None:
                nl = get_nl(non_linearity)

            if normalization is not None:
                norm = get_norm(normalization)(
                    next_feat, **normalization_params)
                init.constant_(norm.weight, 0.1)
                init.constant_(norm.bias, 0)

            if norm_pre_nl:
                if normalization is not None:
                    self.module_list.append(norm)
                if non_linearity is not None:
                    self.module_list.append(nl)
            else:
                if non_linearity is not None:
                    self.module_list.append(nl)
                if normalization is not None:
                    self.module_list.append(norm)

            curr_feat = next_feat
        self.attention = nn.Sequential(*self.module_list)

    def forward(self, input_):
        return self.attention(input_)


class SA_Attention(nn.Module):
    """ if layers > 1  downsample til 1 """

    def __init__(self,
                 input_size,
                 output_size,
                 filter_size,
                 layers,
                 normalization='InstanceNorm2d',  # 'BatchNorm2D'
                 normalization_params={'affine': True},
                 non_linearity='ReLU',
                 norm_pre_nl=False):
        super().__init__()

        if normalization_params is None:
            normalization_params = {}

        curr_feat = input_size
        self.module_list = []
        for i in range(layers):
            if i == layers - 1:
                next_feat = output_size
            else:
                next_feat = curr_feat // 2

            conv = nn.Conv2d(curr_feat, next_feat,
                             filter_size, padding=filter_size // 2)
            init.orthogonal_(conv.weight)  # xavier_normal_
            init.constant_(conv.bias, 0)
            self.module_list.append(conv)

            if non_linearity is not None:
                nl = get_nl(non_linearity)

            if normalization is not None:
                norm = get_norm(normalization)(
                    next_feat, **normalization_params)
                init.constant_(norm.weight, 0.1)
                init.constant_(norm.bias, 0)

            if norm_pre_nl:
                if normalization is not None:
                    self.module_list.append(norm)
                if non_linearity is not None:
                    self.module_list.append(nl)
            else:
                if non_linearity is not None:
                    self.module_list.append(nl)
                if normalization is not None:
                    self.module_list.append(norm)

            curr_feat = next_feat
        self.attention = nn.Sequential(*self.module_list)

    def forward(self, input_):
        return self.attention(input_)


class GALA_Attention(nn.Module):
    """ if layers > 1  downsample til spatial saliency is 1 """

    def __init__(self,
                 input_size,
                 output_size,
                 saliency_filter_size,
                 layers,
                 normalization='InstanceNorm2d',  # 'BatchNorm2D'
                 normalization_params={'affine': True},
                 non_linearity='ReLU',
                 norm_pre_nl=False):
        super().__init__()

        self.se = SE_Attention(input_size, output_size, 1, layers,
                               normalization=normalization,  # 'BatchNorm2D'
                               normalization_params=normalization_params,
                               non_linearity=non_linearity,
                               norm_pre_nl=norm_pre_nl)
        self.sa = SA_Attention(input_size, 1, saliency_filter_size, layers,
                               normalization=normalization,  # 'BatchNorm2D'
                               normalization_params=normalization_params,
                               non_linearity=non_linearity,
                               norm_pre_nl=norm_pre_nl)

    def forward(self, input_):
        return self.sa(input_) * self.se(input_)


def get_nl(name, fun=False, **kwargs):
    if hasattr(F, name) and fun:
        return getattr(F, name)
    elif hasattr(nn, name):
        return getattr(nn, name)(**kwargs)
    else:
        raise Exception("non-linearity doesn't exist")


def get_norm(name, **kwargs):
    if hasattr(nn, name):
        return getattr(nn, name)
    else:
        raise Exception("normalization doesn't exist")
