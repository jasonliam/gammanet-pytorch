import numpy as np
import os
import nibabel as nib
import scipy.io
import random

from constants import ORIGINAL_ACDC_DATASET_ROOT, ORIGINAL_ACDC_TEST_DATASET_ROOT


def txt2npy(in_file, out_file):
    """ read a list of filenames of npy files, and combine them into a single npy file """
    with open(in_file, 'r') as f:
        fnames = f.readlines()
    fnames = [fname.strip() for fname in fnames]
    arr = np.stack([np.load(fname) for fname in fnames], axis=0)
    np.save(out_file, arr)


def npy2txt(in_file, out_path, txt_fname="arr.txt", sample_prefix="data"):
    """ read a dataset in npy format, and split it to one file for each sample """
    arr = np.load(in_file)
    num_length = np.ceil(np.log10(arr.shape[0] + 1))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fnames = []
    for i in range(arr.shape[0]):
        fname = sample_prefix + "_{:0{n}d}.npy".format(i, n=num_length)
        fnames += [fname + '\n']
        np.save(os.path.join(out_path, fname), arr[i])
    with open(np.path.join(out_path, txt_fname), 'w') as f:
        f.writelines(fnames)


def make_acdc_dataset(out_root, timeseries, timesteps=None):
    os.makedirs(out_root, exist_ok=True)
    if not os.path.exists(os.path.join(out_root, "x_arr.txt")):
        data_root = ORIGINAL_ACDC_DATASET_ROOT
        if timeseries:
            convert_acdc_dataset_timeseries(data_root, out_root, num_frames=timesteps)
        else:
            convert_acdc_dataset(data_root, out_root)


def make_acdc_test_dataset(out_root, timeseries, timesteps=None):
    out_root = '{}_test'.format(out_root)
    os.makedirs(out_root, exist_ok=True)
    test_file = os.path.join(out_root, "x_arr.txt")
    if not os.path.exists(test_file):
        data_root = ORIGINAL_ACDC_TEST_DATASET_ROOT
        if timeseries:
            convert_acdc_dataset_timeseries(data_root, out_root, num_frames=timesteps, load_labels=False)
        else:
            convert_acdc_dataset(data_root, out_root, load_labels=False)
    return test_file


def convert_acdc_dataset(data_root, out_root, load_labels=True, val_ratio=None):
    """ Convert NII-formatted ACDC dataset to organized npy """
    print("Building ACDC dataset")
    x_arr = []
    if load_labels:
        y_arr = []

    patients = [p for p in os.listdir(data_root) if "patient" in p]
    for patient in patients:
        print(patient)
        x_arr_p = []
        if load_labels:
            y_arr_p = []

        patient_path = os.path.join(data_root, patient)
        frame_files = [f for f in os.listdir(patient_path)
                       if "frame" in f and "gt" not in f and "slice" not in f]

        patient_path_out = os.path.join(out_root, patient)
        if not os.path.exists(patient_path_out):
            os.makedirs(patient_path_out)

        # iterate over frames
        for frame_file in frame_files:

            # save each slice to npy
            frame = nib.load(os.path.join(
                patient_path, frame_file)).get_fdata()
            for i in range(frame.shape[2]):
                image_file = frame_file.split(
                    '.')[0] + "_slice{}.npy".format(i + 1)
                image_path = os.path.join(patient_path_out, image_file)
                x_arr_p += [image_path + '\n']
                np.save(image_path, frame[:, :, i])

            # save each slice's label to npy
            if load_labels:
                truth_file = frame_file.split('.')[0] + "_gt.nii.gz"
                truth = nib.load(os.path.join(
                    patient_path, truth_file)).get_fdata()
                for i in range(truth.shape[2]):
                    label_file = truth_file.split(
                        '.')[0] + "_slice{}.npy".format(i + 1)
                    label_path = os.path.join(patient_path_out, label_file)
                    y_arr_p += [label_path + '\n']
                    np.save(label_path, truth[:, :, i])

        x_arr += [x_arr_p]
        if load_labels:
            y_arr += [y_arr_p]

    if load_labels and val_ratio is not None:
        num_train = int((1.0 - val_ratio) * len(x_arr))
        x_train = [f for p in x_arr[:num_train] for f in p]
        y_train = [f for p in y_arr[:num_train] for f in p]
        x_val = [f for p in x_arr[num_train:] for f in p]
        y_val = [f for p in y_arr[num_train:] for f in p]
        with open(os.path.join(out_root, "x_train.txt"), 'w') as f:
            f.writelines(x_train)
        with open(os.path.join(out_root, "y_train.txt"), 'w') as f:
            f.writelines(y_train)
        with open(os.path.join(out_root, "x_val.txt"), 'w') as f:
            f.writelines(x_val)
        with open(os.path.join(out_root, "y_val.txt"), 'w') as f:
            f.writelines(y_val)
    else:
        with open(os.path.join(out_root, "x_arr.txt"), 'w') as f:
            f.writelines([f for p in x_arr for f in p])
        if load_labels:
            with open(os.path.join(out_root, "y_arr.txt"), 'w') as f:
                f.writelines([f for p in y_arr for f in p])
    print("Successfully Built ACDC dataset")


def convert_acdc_dataset_timeseries(data_root, out_root, num_frames=2, load_labels=True, val_ratio=None):
    """ Convert NII-formatted ACDC dataset to organized npy """

    x_arr = []
    if load_labels:
        y_arr = []

    patients = [p for p in os.listdir(data_root) if "patient" in p]
    for patient in patients:
        print(patient)
        x_arr_p = []
        if load_labels:
            y_arr_p = []

        patient_path_out = os.path.join(out_root, patient)
        if not os.path.exists(patient_path_out):
            os.makedirs(patient_path_out)

        patient_path = os.path.join(data_root, patient)
        patient_x_path = os.path.join(
            patient_path, "{}_4d.nii.gz".format(patient))
        x_data = nib.load(patient_x_path).get_fdata()

        with open(os.path.join(patient_path, "Info.cfg"), 'r') as meta_file:
            metadata = [l.strip().split(' ') for l in meta_file.readlines()]
        ed_frame = int(metadata[0][1])
        es_frame = int(metadata[1][1])

        if ed_frame >= es_frame:
            raise NotImplementedError

        # ED: note that frame order is reversed
        # save each slice to npy
        for i in range(x_data.shape[2]):
            ts_file = "{}_frame{:02d}-{:02d}_slice{:02d}.npy".format(
                patient, ed_frame + num_frames - 1, ed_frame, i + 1)
            ts_path = os.path.join(patient_path_out, ts_file)
            x_arr_p += [ts_path + '\n']
            ed_flip_idx = x_data.shape[3] - (ed_frame - 1)
            np.save(ts_path, np.flip(x_data, axis=-1)
            [:, :, i, ed_flip_idx - num_frames:ed_flip_idx])
        # save each slice's label to npy
        if load_labels:
            truth_file = "{}_frame{:02d}_gt.nii.gz".format(patient, ed_frame)
            truth = nib.load(os.path.join(
                patient_path, truth_file)).get_fdata()
            for i in range(truth.shape[2]):
                label_file = truth_file.split(
                    '.')[0] + "_slice{:02d}.npy".format(i + 1)
                label_path = os.path.join(patient_path_out, label_file)
                y_arr_p += [label_path + '\n']
                np.save(label_path, truth[:, :, i])

        # ES
        # save each slice to npy
        for i in range(x_data.shape[2]):
            ts_file = "{}_frame{:02d}-{:02d}_slice{:02d}.npy".format(
                patient, es_frame - num_frames + 1, es_frame, i + 1)
            ts_path = os.path.join(patient_path_out, ts_file)
            x_arr_p += [ts_path + '\n']
            np.save(ts_path, x_data[:, :, i, es_frame - num_frames:es_frame])
        # save each slice's label to npy
        if load_labels:
            truth_file = "{}_frame{:02d}_gt.nii.gz".format(patient, es_frame)
            truth = nib.load(os.path.join(
                patient_path, truth_file)).get_fdata()
            for i in range(truth.shape[2]):
                label_file = truth_file.split(
                    '.')[0] + "_slice{:02d}.npy".format(i + 1)
                label_path = os.path.join(patient_path_out, label_file)
                y_arr_p += [label_path + '\n']
                np.save(label_path, truth[:, :, i])

        x_arr += [x_arr_p]
        if load_labels:
            y_arr += [y_arr_p]

    if load_labels and val_ratio is not None:
        num_train = int((1.0 - val_ratio) * len(x_arr))
        x_train = [f for p in x_arr[:num_train] for f in p]
        y_train = [f for p in y_arr[:num_train] for f in p]
        x_val = [f for p in x_arr[num_train:] for f in p]
        y_val = [f for p in y_arr[num_train:] for f in p]
        with open(os.path.join(out_root, "x_train.txt"), 'w') as f:
            f.writelines(x_train)
        with open(os.path.join(out_root, "y_train.txt"), 'w') as f:
            f.writelines(y_train)
        with open(os.path.join(out_root, "x_val.txt"), 'w') as f:
            f.writelines(x_val)
        with open(os.path.join(out_root, "y_val.txt"), 'w') as f:
            f.writelines(y_val)
    else:
        with open(os.path.join(out_root, "x_arr.txt"), 'w') as f:
            f.writelines([f for p in x_arr for f in p])
        if load_labels:
            with open(os.path.join(out_root, "y_arr.txt"), 'w') as f:
                f.writelines([f for p in y_arr for f in p])


def convert_bsds500_dataset(data_root='data', save_root='data_p'):
    x_names = {}
    x_names['train'] = [f[:-4] for f in os.listdir(os.path.join(data_root, "images/train"))
                        if f.endswith(".jpg")]
    x_names['val'] = [f[:-4] for f in os.listdir(os.path.join(data_root, "images/val"))
                      if f.endswith(".jpg")]
    x_names['test'] = [f[:-4] for f in os.listdir(os.path.join(data_root, "images/test"))
                       if f.endswith(".jpg")]

    for s in ['train', 'val', 'test']:

        x_fnames = []
        y_contour_fnames = []
        y_segment_fnames = []

        for x_name in x_names[s]:

            # load MATLAB file for all labels for current image
            y = scipy.io.loadmat(os.path.join(
                data_root, "groundTruth/{}".format(s), "{}.mat".format(x_name)))

            y_dir = os.path.join(
                save_root, "groundTruth/{}/{}".format(s, x_name))
            os.makedirs(y_dir)

            # save each label to individual file
            for i in range(y['groundTruth'].shape[1]):
                # duplicate image file entries
                x_fname = os.path.join(
                    save_root, "images/{}".format(s), "{}.jpg".format(x_name))
                x_fnames += [x_fname + '\n']

                # save contours
                y_contour_i = y['groundTruth'][0, i][0, 0][1]
                y_contour_fname = os.path.join(y_dir, "c{}.npy".format(i))
                np.save(y_contour_fname, y_contour_i)
                y_contour_fnames += [y_contour_fname + '\n']

                # save segmentations
                y_segment_i = y['groundTruth'][0, i][0, 0][0]
                y_segment_fname = os.path.join(y_dir, "s{}.npy".format(i))
                np.save(y_segment_fname, y_segment_i)
                y_segment_fnames += [y_segment_fname + '\n']

        # write files
        with open(os.path.join(save_root, "x_{}.txt".format(s)), 'w') as f:
            f.writelines(x_fnames)
        with open(os.path.join(save_root, "y_c_{}.txt".format(s)), 'w') as f:
            f.writelines(y_contour_fnames)
        with open(os.path.join(save_root, "y_s_{}.txt".format(s)), 'w') as f:
            f.writelines(y_segment_fnames)


def build_dataset_files(processed_data_dir, train_fraction):
    file_list = ["x_train_{}.txt", "y_train_{}.txt", "x_val_{}.txt", "y_val_{}.txt"]
    file_list = [os.path.join(processed_data_dir, f.format(train_fraction)) for f in file_list]

    if os.path.exists(file_list[0]):
        print("Dataset Files Exist!!")
        return file_list

    print("Building dataset files for training fractions: ", train_fraction)
    x_file = os.path.join(processed_data_dir, "x_arr.txt")
    y_file = os.path.join(processed_data_dir, "y_arr.txt")
    with open(y_file, 'r') as f:
        y_arr = np.array(f.readlines())
    with open(x_file, 'r') as f:
        x_arr = np.array(f.readlines())

    patients = [p for p in os.listdir(processed_data_dir) if "patient" in p]
    num_train_patients = int(train_fraction * len(patients))
    train_patients = patients[:num_train_patients]

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for i in range(len(x_arr)):
        patient = (x_arr[i][x_arr[i].find('patient'):x_arr[i].find('patient') + 10])
        if patient in train_patients:
            x_train.append(x_arr[i])
            y_train.append(y_arr[i])
        else:
            x_val.append(x_arr[i])
            y_val.append(y_arr[i])

    with open(file_list[0], 'w') as f:
        f.writelines(x_train)
    with open(file_list[1], 'w') as f:
        f.writelines(y_train)
    with open(file_list[2], 'w') as f:
        f.writelines(x_val)
    with open(file_list[3], 'w') as f:
        f.writelines(y_val)

    return file_list
