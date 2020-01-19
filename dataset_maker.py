import torch

import numpy as np
import os
import nibabel as nib
import pickle
import imageio
import scipy.io
import gzip
import random


# NOTE: don't transfer data to CUDA in dataset (if we need to, use memory pinning)


def convert_acdc_dataset(data_root, load_labels=True, val_ratio=None):
    """ Convert NII-formatted ACDC dataset to organized pkls """

    x_arr = []
    if load_labels:
        y_arr = []

    patients = [p for p in os.listdir(data_root) if "patient" in p]
    for patient in patients:

        x_arr_p = []
        if load_labels:
            y_arr_p = []

        patient_path = os.path.join(data_root, patient)
        frame_files = [f for f in os.listdir(patient_path)
                       if "frame" in f and "gt" not in f and "slice" not in f]

        # iterate over frames
        for frame_file in frame_files:

            # save each slice to pkl
            frame = nib.load(os.path.join(
                patient_path, frame_file)).get_fdata()
            for i in range(frame.shape[2]):
                image_file = frame_file.split(
                    '.')[0] + "_slice{}.pkl".format(i+1)
                image_path = os.path.join(patient_path, image_file)
                x_arr_p += [image_path + '\n']
                pickle.dump(frame[:, :, i], open(image_path, 'wb'))

            # save each slice's label to pkl
            if load_labels:
                truth_file = frame_file.split('.')[0] + "_gt.nii.gz"
                truth = nib.load(os.path.join(
                    patient_path, truth_file)).get_fdata()
                for i in range(truth.shape[2]):
                    label_file = truth_file.split(
                        '.')[0] + "_slice{}.pkl".format(i+1)
                    label_path = os.path.join(patient_path, label_file)
                    y_arr_p += [label_path + '\n']
                    pickle.dump(truth[:, :, i], open(label_path, 'wb'))

        x_arr += [x_arr_p]
        if load_labels:
            y_arr += [y_arr_p]

    if load_labels and val_ratio is not None:
        num_train = int((1.0 - val_ratio) * len(x_arr))
        x_train = [f for p in x_arr[:num_train] for f in p]
        y_train = [f for p in y_arr[:num_train] for f in p]
        x_val = [f for p in x_arr[num_train:] for f in p]
        y_val = [f for p in y_arr[num_train:] for f in p]
        with open(os.path.join(data_root, "x_train.txt"), 'w') as f:
            f.writelines(x_train)
        with open(os.path.join(data_root, "y_train.txt"), 'w') as f:
            f.writelines(y_train)
        with open(os.path.join(data_root, "x_val.txt"), 'w') as f:
            f.writelines(x_val)
        with open(os.path.join(data_root, "y_val.txt"), 'w') as f:
            f.writelines(y_val)
    else:
        with open(os.path.join(data_root, "x_arr.txt"), 'w') as f:
            f.writelines([f for p in x_arr for f in p])
        with open(os.path.join(data_root, "y_arr.txt"), 'w') as f:
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
