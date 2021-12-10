import numpy as np
import pickle
import scipy.io
import joblib
import tensortools as tt
import smartload.smartload as smart
import glob
from scipy.ndimage.morphology import binary_fill_holes
import mat73
from os.path import exists


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


def fit_tca_on_path(filepath, maskpath, savepath, dotca=0, ranks=10, replicates=3):
    '''
    Load and perform TCA on the file specified
    :param filepath: path to the file
    :return: TCA ensemble, raw X data and binary mask
    '''
    print(f'Processing file {filepath}')
    # Load file and preprocess
    data = mat73.loadmat(filepath)
    Xdata = data['allData']['data']
    animal = filepath.split('_')[-2]
    datestr = filepath.split('_')[-1][:6]

    # Try to recover the mask from matdata
    matdata = joblib.load(maskpath)
    maxmask = np.max(matdata['ens_map_all'] ** 2, axis=2)
    maskbinary = binary_fill_holes(maxmask > 0)

    # Baseline subtraction
    baseline = np.mean(Xdata[:, :, :10, :], axis=(2, 3))
    Xdata = Xdata - baseline[:, :, np.newaxis, np.newaxis]
    Xdata[Xdata < 0] = 0
    Xdata = Xdata * maskbinary[:,:,np.newaxis,np.newaxis]
    Xdata = Xdata.reshape((-1, Xdata.shape[2], Xdata.shape[3]))
    print(Xdata.shape)

    # templatefile = f'/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/templateData_{animal}_{datestr}pix.mat'
    # if not exists(templatefile):
    #     raise IOError('Template file does not exist')
    # templateData = scipy.io.loadmat(templatefile)
    # maskbinary =


    # TCA fitting
    print('Fitting TCA...')
    if dotca:
        ensemble = tt.Ensemble(nonneg=True, fit_method="ncp_hals")
        ensemble.fit(Xdata, ranks=ranks, replicates=replicates)
    else:
        ensemble = np.nan

    print('Saving results')
    # Save result
    savefilename = f'{savepath}/{animal}_{datestr}_baseline_corrected_121021b.pkl'
    print(f'Save path: {savefilename}')
    if exists(savefilename):
        raise IOError('File exists')

    with open(savefilename, 'wb') as f:
        pickle.dump(dict(ensemble=ensemble, filename=filepath, maskpath=maskpath, animal=animal, datestr=datestr,
                         maskbinary=maskbinary), f)
    print('Results saved')

    return ensemble, Xdata, maskbinary

if __name__ == '__main__':
    file = '/Volumes/KEJI_DATA_1/nhat/processed-WF/allData_extracted_f04_030321pix.mat'
    templatefile = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-mat/f04_030321pix_tca_nonneg_10comp-blueonly.mat'
    savepath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/ensembles_121021'
    ensemble, Xdata, maskbinary = fit_tca_on_path(file, templatefile, savepath, dotca=1)






