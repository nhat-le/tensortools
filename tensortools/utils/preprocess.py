import numpy as np

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
    datamat_unroll_masked = datamat_unroll[mask_unroll == 1,:,:]

    minval = np.nanmin(datamat_unroll_masked)
    maxval = np.nanmax(datamat_unroll_masked)

    datamat_unroll = datamat_unroll[mask_unroll == 1,:,:]

    # Normalize the data
    normdatamat = (datamat_unroll - minval) / (maxval - minval)

    # TODO: Return the 2d normdatamat
    normdatamat2d = np.zeros((N1 * N2, T, K)) * np.nan
    normdatamat2d[mask_unroll == 1, :, :] = normdatamat
    normdatamat2d = normdatamat2d.reshape((N1, N2, T, K))

    # datamat_unroll_copy = datamat_unroll.copy()
    # datamat_unroll_copy[mask_unroll != 1] = np.nan
    # datamat_unroll_reshape = np.reshape(datamat_unroll_copy, (N1, N2, T, -1))
    # datamat_unroll_reshape = np.transpose(datamat_unroll_reshape, axes=[2, 3, 0, 1])
    # datamat_unroll_reshape = (datamat_unroll_reshape - minval) / (maxval - minval)
    # datamat_unroll_reshape = np.transpose(datamat_unroll_reshape, axes=[2, 3, 0, 1])

    # Other ideas for normalizing the data
    # normdatamat = np.zeros_like(datamat_unroll_masked)
    # # for i in range(datamat_unroll.shape[0]):
    #     cellmat = datamat_unroll[i, :, :]
    #     normcellmat = (cellmat - minval) / (maxval - minval)
    #     normdatamat[i, :, :] = normcellmat

    return normdatamat, normdatamat2d, maskbinary