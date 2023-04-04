import os
import numpy as np
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd

EXPE = 'dar_es_salam'

ENV_ROOT = '/media/jarry/Seagate Expansion Drive/' + 'Optical_images/' + EXPE + '/'

PATH_TO_SITS = ENV_ROOT
SITS_NAME = EXPE + '_%s.tif'

PATH_TO_DMSP = ENV_ROOT + 'DMSP/'
DMSP_NAME = 'ntl_%s.tif'

PATH_TO_VIIRS = ENV_ROOT + 'VIIRS/'
VIIRS_NAME = 'ntl_%s.tif'

BEGIN = 2000
END = 2021

EX_ZONE = ('zanzibar', [1900, 2800, 4900, 5900])
PS = 32

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def patch(X, ps):
    if len(X.shape) == 4:
        n_ts, h, w, n_c = X.shape
    elif len(X.shape) == 3:
        n_ts, h, w = X.shape
    else:
        h, w = None, None
        print('Warning : X shape is not valid')

    X = np.array([X[:, ps * j:ps * (j + 1), ps * k:ps * (k + 1)]
                  for j in range(h // ps)
                  for k in range(w // ps)])
    return X

def unpatch(X, n_row, n_col):
    if len(X.shape) == 5:
        _, n_ts, ps, _, n_c = X.shape
        img_patch_shape = (n_row, n_col, ps, ps, n_c, n_ts)
        transpose_axes = [0, 2, 1, 3, 4, 5]
        fin_shape = (n_row * ps, n_col * ps, n_c, n_ts)
    elif len(X.shape) == 4:
        _, n_ts, ps, _ = X.shape
        img_patch_shape = (n_row, n_col, ps, ps, n_ts)
        transpose_axes = [0, 2, 1, 3, 4]
        fin_shape = (n_row * ps, n_col * ps, n_ts)

    else:
        n_ts, ps, n_c, w = None, None, None, None
        img_patch_shape = None
        transpose_axes = None
        fin_shape = None
        print('Warning : X shape is not valid')

    X = np.moveaxis(X, 1, -1)
    X = X.reshape(img_patch_shape)
    X = np.transpose(X, axes=transpose_axes)
    X = X.reshape(fin_shape)
    X = np.moveaxis(X, -1, 0)

    return X

def save_raw():

    # 1 - Load SI TS and NTL TS

    writing_path = EXPE + '/raw_data/'
    os.makedirs(writing_path, exist_ok=True)

    bands = [1, 2, 3, 4, 5, 6]

    sits = []
    ntls_DMSP = []
    ntls_VIIRS = []

    for year in tqdm(range(BEGIN, END)):

        s_fname = PATH_TO_SITS + SITS_NAME % year

        with rasterio.open(s_fname) as tif:
            img = tif.read(bands)
            img = np.moveaxis(img, 0, -1)
        sits.append(img)

        ntls_dmsp = PATH_TO_DMSP + DMSP_NAME % year

        with rasterio.open(ntls_dmsp) as tif:
            img = tif.read()
            img = np.squeeze(img, 0)
        ntls_DMSP.append(img)

        ntls_viirs = PATH_TO_VIIRS + VIIRS_NAME % year

        with rasterio.open(ntls_viirs) as tif:
            img = tif.read()
            img = np.squeeze(img, 0)
        ntls_VIIRS.append(img)

    # 2 Save raw data

    sits = np.array(sits)
    ntls_DMSP = np.array(ntls_DMSP)
    ntls_VIIRS = np.array(ntls_VIIRS)



    np.save(writing_path + 'sits', sits)
    np.save(writing_path + 'dmsp', ntls_DMSP)
    np.save(writing_path + 'viirs', ntls_VIIRS)

    vis_sits = sits[:, :, :, [2, 1, 0]].astype('uint8')

    B2, B4 = sits[0, :, :, 1].astype('float32'), sits[0, :, :, 3].astype('float32')
    sits = (B2 - B4) / (B2 + B4)

    mask_water = np.zeros_like(sits)

    mask_water[np.logical_and(sits >= 0.1, sits <= 0.7)] = 1

    np.save(writing_path + '/ndwi', mask_water)

    fig, axes = plt.subplots(1, 4)


    axes[0].imshow(vis_sits[0])
    axes[0].set_title('Optical 2000')
    axes[1].imshow(sits)
    axes[1].set_title('NDWI 2000')
    axes[2].imshow(ntls_DMSP[0])
    axes[2].set_title('DMSP 2000')
    axes[3].imshow(ntls_VIIRS[0])
    axes[3].set_title('VIIRS 2000')

    fig.savefig(writing_path + 'raw_data.png')

    plt.show()

def load_and_patch(ntl_type, PS):

    X = np.load(EXPE + '/raw_data/sits.npy')
    y = np.load(EXPE + f'/raw_data/{ntl_type}.npy')

    n_row, n_col = X.shape[1] // PS, X.shape[2] // PS

    X = patch(X, PS)
    y = patch(y, PS).mean(axis=(-1, -2)).astype('float32')

    return X, y, n_row, n_col

def train_test_split(ntl_type):

    def exclude_zone(n_ex, h, w, PS):
        # ------3. 1 Exclude image zone

        all_index = np.reshape(range(0, n_ex), (h // PS, w // PS))

        from_row, to_row, from_col, to_col = EX_ZONE[1]

        ex_index = all_index[from_row // PS:to_row // PS, from_col // PS:to_col // PS]
        ex_index = ex_index.ravel()

        all_index = set(all_index.ravel())
        ex_index = set(ex_index)
        kept_index = np.array(list(all_index - ex_index))

        return kept_index, np.array(list(ex_index))

    def k_fold(index, n_folds=6):

        np.random.shuffle(index)
        n_ex_per_folds = index.shape[0] // n_folds + 1
        folds = {f'fold_{i}': index[i * n_ex_per_folds:(i + 1) * n_ex_per_folds] for i in range(n_folds)}

        return folds

    # 1 - Patch the data

    dmsp = np.load(EXPE + '/raw_data/dmsp.npy')
    n_ts, h, w = dmsp.shape
    dmsp = patch(dmsp, PS)
    n_ex = dmsp.shape[0]

    # 3 - Train Test split

    #Get indexes of excluded zone and kept zone
    kept_index, ex_index = exclude_zone(n_ex, h, w, PS)


    fold_saving_path = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/'
    os.makedirs(fold_saving_path, exist_ok=True)

    np.save(fold_saving_path + 'ex_index', ex_index)

    # Don't make K-fold if it already exists
    if os.path.exists(fold_saving_path + 'folds.json'):
        print('folds already exists, pass')
        with open(fold_saving_path + 'folds.json', 'r') as file:
            folds = json.load(file)

    else:
        folds = k_fold(kept_index, n_folds=6)
        with open(fold_saving_path + 'folds.json', 'w') as file:
            json.dump(folds, file, cls=NpEncoder)

    assert np.sum([len(fold) for _, fold in folds.items()]) == len(kept_index)

    X, y, h, w = load_and_patch(ntl_type, PS)

    for fold_name, fold in folds.items():
        current_fold = fold_saving_path + f'data/{fold_name}/'
        os.makedirs(current_fold, exist_ok=True)
        np.save(current_fold + 'sits.npy', X[fold])
        np.save(current_fold + f'{ntl_type}.npy', y[fold])

    current_fold = fold_saving_path + f'data/ex_zone/'
    os.makedirs(current_fold, exist_ok=True)
    np.save(current_fold + 'sits.npy', X[ex_index])
    np.save(current_fold + f'{ntl_type}.npy', y[ex_index])


    return folds

def load_train_val_test_ex(train, val, test, ntl_type):

    def filter_train(X_train, y_train, zeros_kept=500):

        non_zeros = np.where(y_train.mean(axis=-1) > 0)[0].tolist()
        zeros = np.where(y_train.mean(axis=-1) == 0)[0].tolist()

        np.random.shuffle(zeros)
        zeros = zeros[:500]

        X_train = X_train[non_zeros + zeros]
        y_train = y_train[non_zeros + zeros]

        return X_train, y_train

    X = []
    y = []

    data_path = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/data/'

    for t in train:
        X.extend(np.load(data_path + f'fold_{t}/sits.npy'))
        y.extend(np.load(data_path + f'fold_{t}/{ntl_type}.npy'))

    X_train = np.array(X)
    y_train = np.array(y)

    del X
    del y

    X_train, y_train = filter_train(X_train, y_train)

    X_val = np.load(data_path + f'fold_{val}/sits.npy')
    y_val = np.load(data_path + f'fold_{val}/{ntl_type}.npy')

    X_test = np.load(data_path + f'fold_{test}/sits.npy')
    y_test = np.load(data_path + f'fold_{test}/{ntl_type}.npy')

    X_ex = np.load(data_path + f'ex_zone/sits.npy')
    y_ex = np.load(data_path + f'ex_zone/{ntl_type}.npy')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-save_raw", action='store_true')
    parser.add_argument("-write_folds", action='store_true')
    parser.add_argument("-ntl_type")

    args = parser.parse_args()

    if args.save_raw:
        save_raw()
    if args.write_folds:
        train_test_split(args.ntl_type)

