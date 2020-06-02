import numpy as np
import nibabel as nib
import os
from skimage.measure import label


def get_largest_CC(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(int)


def get_details_from_path(path):
    path = path.strip()
    s = path.split('/')
    t = s[-1].split('_')
    patient = t[0]
    frame = t[1][-2:]
    slice_num = t[2][5:t[2].find('.')]
    return patient, int(frame), int(slice_num)


def get_metadata(file_list):
    out = {}
    for file in file_list:
        p, f, s = get_details_from_path(file)
        f, s = int(f), int(s)
        p_dict = out.get(p)
        if p_dict is None:
            p_dict = {}

        p_dict[f] = max(p_dict.get(f, -1), s)

        out[p] = p_dict

    for p in out:
        d = out[p].keys()
        ed = min(d)
        es = max(d)
        out[p][ed] = {'type': 'ED', 'slices': out[p][ed]}
        out[p][es] = {'type': 'ES', 'slices': out[p][es]}

    return out


def unpad(data, shape):
    padded_shape = data.shape
    if padded_shape[0] == shape[0]:
        diff = padded_shape[1] - shape[1]
        return data[:, diff // 2: diff // 2 + shape[1]]
    else:
        diff = padded_shape[0] - shape[0]
        return data[diff // 2: diff // 2 + shape[0], :]


def save_test_results(root_dir, results):
    test_dir = os.path.join(root_dir, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    file_list = results.keys()
    patients_metadata = get_metadata(file_list)
    final_output = {}
    for file in file_list:
        p, f, s = get_details_from_path(file)
        data = results[file]
        output = unpad(data['data'], data['shape'])
        output = get_largest_CC(output)
        output[output == 1] = 3
        metadata = patients_metadata[p]

        file_name = '{}_{}.nii.gz'.format(p, metadata[f]['type'])
        file_name = os.path.join(test_dir, file_name)

        r = final_output.get(file_name, np.zeros(output.shape + (metadata[f]['slices'],)))
        r[:, :, s - 1] = output
        final_output[file_name] = r

    affine = np.diag([-1, -1, 1, 1])
    for file in final_output:
        print('Saving ', file)
        data = final_output[file]
        nimg = nib.Nifti1Image(data, affine=affine)
        nimg.to_filename(file)
