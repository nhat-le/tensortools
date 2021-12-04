import numpy as np


def reconstruction(ensemble, num_components, replicate, maskbinary=None, clustsubset=None):
    '''
    Perform a reconstruction of the raw data based on a TCA factor set
    :param ensemble: TCA ensemble as returned by tensortools
    :param num_components: number of components
    :param replicate: replicate in the component set
    :param maskbinary: binary mask (0/1) which indicate where the pixels of interest are
    :param clustsubset: subset of TCA modes to perform the reconstruction
    :return: Xhat, the reconstructed data, if maskbinary is not None, will expand to the full fov
    and pad with nan's
    '''
    assert clustsubset is None or type(clustsubset) is list
    W, B, A = ensemble.factors(num_components)[replicate].factors
    (npix, _), (T, _), (K, P) = W.shape, B.shape, A.shape

    Xhat = np.zeros((npix, T, K))

    if clustsubset is None:
        clustsubset = np.arange(P)
    Wsub = W[:, clustsubset]
    Bsub = B[:, clustsubset]
    Asub = A[:, clustsubset]

    for i in range(K):
        WA = np.dot(Wsub, np.diag(Asub[i, :]))
        Xk = np.dot(WA, Bsub.T)
        Xhat[:, :, i] = Xk

    # Expand Xhat to the full set
    N1, N2 = maskbinary.shape
    mask_unroll = maskbinary.flatten()
    if maskbinary is not None:
        Xhatsquare = np.zeros((N1 * N2, T, K)) * np.nan
        Xhatsquare[mask_unroll == 1] = Xhat
        Xhatsquare = Xhatsquare.reshape((N1, N2, T, -1))

    return Xhatsquare

def find_cluster_contributions(ensemble, num_components, replicate, maskbinary=None):
    '''
    Determine the contribution of each cluster over time
    :param ensemble: TCA ensemble as returned by tensortools
    :param num_components: number of components
    :param replicate: replicate in the component set
    :param maskbinary: binary mask (0/1) which indicate where the pixels of interest are
    :param clustsubset: subset of TCA modes to perform the reconstruction
    '''
    W, B, A = ensemble.factors(num_components)[replicate].factors
    (npix, _), (T, _), (K, P) = W.shape, B.shape, A.shape

    recons_lst = [reconstruction(ensemble, num_components, replicate, maskbinary, clustsubset=[i]) for i in range(P)]

    # take trial average
    recons_means = [np.mean(elem, axis=3) for elem in recons_lst]
    recons_sum = np.sum(np.array(recons_means), axis=0)
    contributions = [elem / recons_sum for elem in recons_means]


    return contributions






