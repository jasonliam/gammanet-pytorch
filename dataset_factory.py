from dataset import SimpleDataset
from dataset_maker import make_acdc_dataset, build_dataset_files, make_acdc_test_dataset
from transforms_factory import get_transforms, get_test_transforms


def get_datasets(config_data):
    x_transform_train, x_transform_val, y_transform, y_transform_val = get_transforms(config_data)
    data_dir = config_data['dataset']['path']
    train_fraction = config_data['dataset']['training_fraction']
    timeseries = config_data['dataset']['timeseries']
    timesteps = config_data['model']['fgru_timesteps']

    # Build Processed Dataset if it doesn't exist
    make_acdc_dataset(data_dir, timeseries, timesteps)
    x_train_file, y_train_file, x_val_file, y_val_file = build_dataset_files(data_dir, train_fraction)

    ds_train = SimpleDataset(x_train_file, y_train_file,
                             x_transform=x_transform_train, y_transform=y_transform, use_cache=True)
    ds_val = SimpleDataset(x_val_file, y_val_file,
                           x_transform=x_transform_val, y_transform=y_transform_val, use_cache=True)

    return ds_train, ds_val


def get_test_dataset(config_data):
    x_transform = get_test_transforms(config_data)
    data_dir = config_data['dataset']['path']
    timeseries = config_data['dataset']['timeseries']
    timesteps = config_data['model']['fgru_timesteps']

    # Build Processed Dataset if it doesn't exist
    test_file = make_acdc_test_dataset(data_dir, timeseries, timesteps)
    ds_test = SimpleDataset(test_file,
                            x_transform=x_transform, use_cache=True)
    return ds_test
