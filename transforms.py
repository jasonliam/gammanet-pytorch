import numpy as np
import skimage.transform
import scipy
import cv2 as cv
import torch
import torch.nn.functional as F


# ==================================================
# Padding, cropping
# ==================================================

def pad_base(x, padding, **kwargs):
    if isinstance(x, np.ndarray):
        padding = padding.astype(int).tolist()
        x = np.pad(x, padding, **kwargs)
    elif isinstance(x, torch.Tensor):
        padding = tuple(np.flip(padding, axis=0).flatten().astype(int))
        x = F.pad(x, padding, **kwargs)
    else:
        raise NotImplementedError
    return x


class PadTo2Power(object):
    """
    Pad input axes such that their sizes are multiples of 2^k
    Use to ensure that inputs upsample to original sizes in autoencoder-style networks
    """

    def __init__(self, k, axes=(0, 1), **kwargs):
        self.axes = axes  # axes to pad
        self.k = k  # 2 power to pad to
        self.kwargs = kwargs

    def __call__(self, x):

        padding = np.zeros((len(x.shape), 2))
        for axis in self.axes:
            x_dim = x.shape[axis]
            diff = 0 if x_dim % 2 ** self.k == 0 \
                else ((x_dim // 2 ** self.k + 1) * 2 ** self.k) - x_dim
            if diff == 0:
                continue
            padding[axis] = [diff // 2, diff // 2 + diff % 2]

        return pad_base(x, padding, **self.kwargs)


class PadToSquare(object):

    def __init__(self, axes=(0, 1), **kwargs):
        self.axes = axes  # axes to make square
        self.kwargs = kwargs  # pass to np.pad()

    def __call__(self, x):
        diff = x.shape[self.axes[0]] - x.shape[self.axes[1]]
        if diff == 0:
            return x
        pad_axis = self.axes[0] if diff < 0 else self.axes[1]
        padding = np.zeros((len(x.shape), 2))
        padding[pad_axis] = [abs(diff) // 2, abs(diff) // 2 + abs(diff) % 2]

        return pad_base(x, padding, **self.kwargs)


class PadToSize(object):

    def __init__(self, size, axes=(0, 1), **kwargs):
        self.axes = axes
        self.size = size
        self.kwargs = kwargs

    def __call__(self, x):
        h_diff = x.shape[self.axes[0]] - self.size[0]
        w_diff = x.shape[self.axes[1]] - self.size[1]
        assert h_diff <= 0 and w_diff <= 0
        if h_diff == 0 and w_diff == 0:
            return x
        padding = np.zeros((len(x.shape), 2))
        padding[0] = [abs(h_diff) // 2, abs(h_diff) // 2 + abs(h_diff) % 2]
        padding[1] = [abs(w_diff) // 2, abs(w_diff) // 2 + abs(w_diff) % 2]
        return pad_base(x, padding, **self.kwargs)


class CenterCrop(object):
    """ Center crop ndarray image (h,w,...)  """

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        h_diff = x.shape[0] - self.size[0]
        w_diff = x.shape[1] - self.size[1]
        assert h_diff >= 0 and w_diff >= 0
        if h_diff == 0 and w_diff == 0:
            return x
        elif h_diff == 0:
            return x[:, w_diff // 2:-(w_diff // 2 + w_diff % 2)]
        elif w_diff == 0:
            return x[h_diff // 2:-(h_diff // 2 + h_diff % 2), :]
        else:
            return x[h_diff // 2:-(h_diff // 2 + h_diff % 2), w_diff // 2:-(w_diff // 2 + w_diff % 2)]


class PadOrCenterCrop(object):
    """ Pad or center crop to given size (h,w,...) """

    def __init__(self, size, **kwargs):
        self.size = size
        self.kwargs = kwargs

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        h_diff = x.shape[0] - self.size[0]
        w_diff = x.shape[1] - self.size[1]
        if h_diff == 0 and w_diff == 0:
            return x
        elif h_diff < 0 and w_diff < 0:  # pad
            padding = np.zeros((len(x.shape), 2))
            padding[0] = [abs(h_diff) // 2, abs(h_diff) // 2 + abs(h_diff) % 2]
            padding[1] = [abs(w_diff) // 2, abs(w_diff) // 2 + abs(w_diff) % 2]
            return pad_base(x, padding, **self.kwargs)
        elif h_diff >= 0 and w_diff >= 0:  # crop
            return CenterCrop(self.size)(x)
        else:  # pad to square then crop
            h_pad_size = np.max([x.shape[0], self.size[0]])
            w_pad_size = np.max([x.shape[1], self.size[1]])
            x = PadToSize((h_pad_size, w_pad_size), (0, 1), **self.kwargs)(x)
            return CenterCrop(self.size)(x)


# ==================================================
# Resizing
# ==================================================

class MultiChannelTransform(object):
    """ 
    Base class for wrapping transforms which only support single-channel 
    - Supports up to 1 channel dimension, assumes HWC format
    - Implement __init__() and _transform() for child classes
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, x):
        if len(x.shape) == 2:
            return self._transform(x)
        elif len(x.shape) == 3:
            arr = []
            for i in range(x.shape[2]):
                arr += [self._transform(x[:, :, i])]
            return np.stack(arr, axis=2)
        else:
            raise NotImplementedError

    def _transform(self, x):
        raise NotImplementedError


class DownsampleShortAxis(MultiChannelTransform):
    """ Downsample to match shorter axis of the image to the given size """

    def __init__(self, size, **kwargs):
        self.size = size
        self.kwargs = kwargs

    def _transform(self, x):
        # don't do anything if any axis is smaller than size
        if np.min(x.shape) < self.size:
            return x
        ds_ratio = np.min(x.shape) / self.size
        new_shape = (int(x.shape[0] / ds_ratio), int(x.shape[1] / ds_ratio))
        return skimage.transform.resize(x, new_shape, **self.kwargs)


class Resize(MultiChannelTransform):
    """ Resize ndarray image """

    def __init__(self, size, **kwargs):
        self.size = size
        self.kwargs = kwargs

    def _transform(self, x):
        assert isinstance(x, np.ndarray)
        return skimage.transform.resize(x, self.size, **self.kwargs)


# ==================================================
# Filtering
# ==================================================

class GaussianSmooth(MultiChannelTransform):

    def __init__(self, size, sigma):
        assert size % 2 == 1
        self.size = size
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        self.kernel = g / g.sum()[None]

    def _transform(self, x):
        assert isinstance(x, np.ndarray)
        return scipy.signal.convolve2d(x, self.kernel, mode='same')


class CLAHE(MultiChannelTransform):

    def __init__(self, clipLimit, tileGridSize):
        self.clahe = cv.createCLAHE(
            clipLimit=clipLimit, tileGridSize=tileGridSize)

    def _transform(self, x):
        assert isinstance(x, np.ndarray)
        return self.clahe.apply(x.astype(np.uint16))


# ==================================================
# Scaling
# ==================================================

class MinMax(object):

    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def __call__(self, x):
        return ((x - x.min(axis=self.axes)) /
                (x.max(axis=self.axes) - x.min(axis=self.axes)))


class ZScore(object):
    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def __call__(self, x):
        return (x - x.mean(axis=self.axes)) / x.std(axis=self.axes)


# ==================================================
# Misc
# ==================================================


class SelectChannel(object):
    """ Select only one channel of the input """

    def __init__(self, label_id, x_format="CHW"):
        self.label_id = label_id
        self.x_format = x_format

    def __call__(self, x):
        if self.x_format == "CHW":
            return x[self.label_id]
        elif self.x_format == "HWC":
            return x[:, :, self.label_id]
        else:
            raise NotImplementedError


class SelectClass(object):
    """ Select only one class in a multi-class label """

    def __init__(self, class_id):
        self.class_id = class_id

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return (x == self.class_id).astype(int)
        elif isinstance(x, torch.Tensor):
            return (x == self.class_id).int()
        else:
            raise NotImplementedError


class AssertWidthMajor(object):
    """ make sure that input width > height (specifically for BSDS500) """

    def __call__(self, x):
        assert isinstance(x, torch.Tensor)
        if x.shape[-2] > x.shape[-1]:
            x = torch.transpose(x, -2, -1)
        return x


class ExpandDims(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return np.expand_dims(x, self.dim)
        elif isinstance(x, torch.Tensor):
            return torch.unsqueeze(x, self.dim)
        else:
            raise NotImplementedError


class ToTensor(object):

    def __init__(self, make_CHW=True, input_format="HW", out_type=float):
        self.make_CHW = make_CHW
        self.input_format = input_format
        self.out_type = out_type

    def __call__(self, x):
        assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
        if self.make_CHW:
            if self.input_format == "HW":
                x = np.expand_dims(x, axis=0)
            elif self.input_format == "HWC":
                x = np.transpose(x, axes=(2, 0, 1))
            else:
                raise NotImplementedError
        x = x.astype(self.out_type)
        x = torch.from_numpy(x).contiguous()
        return x


class TimeSeriesToTensor(object):

    def __init__(self, make_TCHW=True, input_format="HWT", out_type=float):
        self.make_TCHW = make_TCHW
        self.input_format = input_format
        self.out_type = out_type

    def __call__(self, x):
        assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
        if self.make_TCHW:
            if self.input_format == "HWT":
                x = np.transpose(x, axes=(2, 0, 1))
                x = np.expand_dims(x, axis=1)
            elif self.input_format == "THWC":
                x = np.transpose(x, axes=(0, 3, 1, 2))
            elif self.input_format == "HWCT":
                x = np.transpose(x, axes=(3, 2, 0, 1))
            else:
                raise NotImplementedError
        else:
            if self.input_format == "HWT":
                x = np.transpose(x, axes=(2, 0, 1))
            else:
                raise NotImplementedError
        x = x.astype(self.out_type)
        x = torch.from_numpy(x).contiguous()
        return x
