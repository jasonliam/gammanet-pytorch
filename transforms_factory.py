import torchvision.transforms as tf

from transforms import *


def get_transforms(config_data):
    x_out_train = []
    x_out_val = []
    y_out = []
    y_out_val = []

    for transform in config_data['dataset']['transforms']:
        if transform == 'GaussianSmooth':
            x_out_train.append(GaussianSmooth(3, 1))
            x_out_val.append(GaussianSmooth(3, 1))
        elif transform == 'CLAHE':
            x_out_train.append(CLAHE(clipLimit=2.0, tileGridSize=(8, 8)))
            x_out_val.append(CLAHE(clipLimit=2.0, tileGridSize=(8, 8)))
        elif transform == 'MinMax':
            x_out_train.append(MinMax())
            x_out_val.append(MinMax())
        elif transform == 'PadOrCenterCrop':
            x_out_train.append(PadOrCenterCrop(size=(176, 176)))
            y_out.append(PadOrCenterCrop(size=(176, 176)))
            x_out_val.append(PadOrCenterCrop(size=(176, 176)))
            y_out_val.append(PadOrCenterCrop(size=(176, 176)))

    if config_data['model']['timeseries']:
        x_out_train.append(TimeSeriesToTensor(make_TCHW=True, out_type=float))
        x_out_val.append(TimeSeriesToTensor(make_TCHW=True, out_type=float))
    else:
        x_out_train.append(ToTensor(make_CHW=True, out_type=float))
        x_out_val.append(ToTensor(make_CHW=True, out_type=float))

    y_out.append(ToTensor(make_CHW=False, out_type=int))
    y_out.append(SelectClass(3))
    y_out_val.append(ToTensor(make_CHW=False, out_type=int))
    y_out_val.append(SelectClass(3))

    return tf.Compose(x_out_train), tf.Compose(x_out_val), tf.Compose(y_out), tf.Compose(y_out_val)


def get_test_transforms(config_data):
    x_out = []

    for transform in config_data['dataset']['transforms']:
        if transform == 'GaussianSmooth':
            x_out.append(GaussianSmooth(3, 1))
        elif transform == 'CLAHE':
            x_out.append(CLAHE(clipLimit=2.0, tileGridSize=(8, 8)))
        elif transform == 'MinMax':
            x_out.append(MinMax())

    if config_data['model']['timeseries']:
        x_out.append(TimeSeriesToTensor(make_TCHW=True, out_type=float))
    else:
        x_out.append(ToTensor(make_CHW=True, out_type=float))

    return tf.transforms.Compose(x_out)
