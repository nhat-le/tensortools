import numpy as np
import joblib
import glob
from scipy.ndimage.morphology import binary_fill_holes


def load_pkl_data(matpath, ensembles_path):
    matdata = joblib.load(matpath)
    ensemble = joblib.load(ensembles_path)

    feedback = matdata['feedback']
    responses = matdata['responses']
    targets = matdata['target']

    # Try to recover the mask from matdata
    maxmask = np.max(matdata['ens_map_all'] ** 2, axis=2)
    maskbinary = maxmask > 0

    if matdata['W'].shape[0] != np.sum(maskbinary > 0):
        maskbinary = binary_fill_holes(maxmask > 0)

    try:
        assert (matdata['W'].shape[0] == np.sum(maskbinary > 0))
    except AssertionError:
        print(f'Warning: file mask size does not match dimension of W')

    return ensemble, matdata, dict(feedback=feedback, responses=responses, targets=targets), maskbinary

def prepare_data(datamat, mask):
    '''
    Normalize the data and binarize the mask
    :return: (data, mask)
    '''
    maskbinary = mask.copy()
    maskbinary[mask != 0] = 1
    mask_unroll = maskbinary.flatten().astype('int')
    N1,N2, T, K = datamat.shape

    datamat_unroll = np.reshape(datamat, (N1*N2, T, -1))
    datamat_unroll = datamat_unroll[mask_unroll == 1,:,:]
    minval = np.nanmin(datamat_unroll)
    maxval = np.nanmax(datamat_unroll)

    # Normalize the data
    normdatamat = (datamat_unroll - minval) / (maxval - minval)

    # Return the 2d normdatamat
    normdatamat2d = convert_to_2d(normdatamat, maskbinary)

    return normdatamat, normdatamat2d, maskbinary

def convert_to_2d(datamat, mask):
    '''
    Given a flattened data, return the data to a 2-d representation
    :param datamat: np array of size N x T x K where N < N1 * N2 is a subset of pixels
    defined by the mask
    :param mask: np array of size N1 x N2, where N pixels are 1, others are zero
    :return: N1 x N2 x T x K np array
    '''
    N1, N2 = mask.shape

    if datamat.ndim == 3:
        _, T, K = datamat.shape
        datamat2d = np.zeros((N1 * N2, T, K)) * np.nan
        mask_unroll = mask.flatten().astype('int')
        datamat2d[mask_unroll == 1, :, :] = datamat
        datamat2d = datamat2d.reshape((N1, N2, T, K))
    elif datamat.ndim == 2:
        T, K = datamat.shape
        datamat2d = np.zeros((N1 * N2, K))
        mask_unroll = mask.flatten().astype('int')
        datamat2d[mask_unroll != 0] = datamat
        datamat2d[mask_unroll == 0] = np.nan
        datamat2d = np.reshape(datamat2d, (N1, N2, K))

    return datamat2d

def apply_mask(datamat, mask):
    '''
    Given a datamat, apply and flatten based on mask
    :param datamat: np array of size N1 x N2 x T x K
    :param mask: np array of size N1 x N2, where
    :param mask: np array of size N1 x N2, where N pixels are 1, others are zero
    :return: N x T x K np array
    '''
    N1, N2, T, K = datamat.shape
    assert mask.shape[0] == N1 and mask.shape[1] == N2
    mask_unroll = mask.flatten().astype('int')
    res = datamat.reshape((N1 * N2, T, K))
    return res[mask_unroll == 1, :, :]






