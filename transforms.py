import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F


class AssertWidthMajor(object):
    """ make sure that input width > height (specifically for BSDS500) """

    def __call__(self, x):
        assert isinstance(x, torch.Tensor)
        if x.shape[-2] > x.shape[-1]:
            x = torch.transpose(x, -2, -1)
        return x


class PadTo2Power(object):
    """ 
    Pad input axes such that it's a multiple of 2^k
    Use to ensure that inputs upsample to original sizes in autoencoder-style networks
    """

    def __init__(self, axes, k, **kwargs):
        self.axes = axes  # axes to pad
        self.k = k  # 2 power to pad to
        self.kwargs = kwargs

    def __call__(self, x):

        padding = np.zeros((len(x.shape), 2))
        for axis in self.axes:
            x_dim = x.shape[axis]
            diff = 0 if x_dim % 2**self.k == 0 \
                else ((x_dim // 2**self.k + 1) * 2**self.k) - x_dim
            if diff == 0:
                continue
            padding[axis] = [diff // 2, diff // 2 + diff % 2]

        if isinstance(x, np.ndarray):
            padding = padding.astype(int).tolist()
            x = np.pad(x, padding, **self.kwargs)
        elif isinstance(x, torch.Tensor):
            padding = tuple(np.flip(padding, axis=0).flatten().astype(int))
            x = F.pad(x, padding, **self.kwargs)
        else:
            raise NotImplementedError

        return x


class PadToSquare(object):

    def __init__(self, axes, pad_label=False, **kwargs):
        self.axes = axes      # axes to make square
        self.pad_label = pad_label
        self.kwargs = kwargs  # pass to np.pad()

    def __call__(self, x):

        diff = x.shape[self.axes[0]] - x.shape[self.axes[1]]
        if diff == 0:
            return x
        pad_axis = self.axes[0] if diff < 0 else self.axes[1]
        padding = np.zeros((len(x.shape), 2))
        padding[pad_axis] = [abs(diff) // 2, abs(diff) // 2 + abs(diff) % 2]

        if isinstance(x, np.ndarray):
            padding = padding.astype(int).tolist()
            x = np.pad(x, padding, **self.kwargs)
        elif isinstance(x, torch.Tensor):
            padding = tuple(np.flip(padding, axis=0).flatten().astype(int))
            x = F.pad(x, padding, **self.kwargs)
        else:
            raise NotImplementedError

        return x


class Resize(object):

    def __init__(self, size, anti_aliasing=True):
        self.size = size
        self.anti_aliasing = anti_aliasing

    def __call__(self, x):
        return skimage.transform.resize(x, self.size, self.anti_aliasing)


class ToTensor(object):

    def __init__(self, make_channel_first=True, float_out=True, div=True):
        self.make_channel_first = make_channel_first
        self.float_out = float_out
        self.div = div

    def __call__(self, x):
        assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
        assert len(x.shape) >= 2, "Only valid for arrays with 2+ dimensions"
        if len(x.shape) == 2:
            np.expand_dims(np.array(x), axis=0)
        elif self.make_channel_first:
            x = np.transpose(x, axes=tuple(
                np.roll(np.arange(len(x.shape)), 1)))
        if self.float_out:
            x = x.astype(float)/255 if self.div else x.astype(float)
        else:
            x = x.astype(int)
        x = torch.from_numpy(x).contiguous()
        return x
