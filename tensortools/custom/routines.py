# A collection of useful routines for TCA analysis (batch)
import numpy as np
import scipy.optimize
import scipy.stats
from tensortools.custom import imaging_data

def load_and_run(rootpath, animal, expdate, fit_method, ranks, kwargs):
    '''
    Load, run and save TCA ensemble information
    :param animal: str, animal name
    :param expdate: str, date of data, like 030121
    :return: ensemble information
    '''

    kwargs = parse_kwargs(kwargs)

    animaldata = imaging_data.ImagingData(rootpath, animal, expdate, trialsubset=kwargs['trialsubset'])
    animaldata.normalize(type=kwargs['normtype'], baseline_frames=kwargs['baseline_frames'],
                         keep_positive=kwargs['keep_positive'])
    animaldata.tca_fit(nonneg=kwargs['nonneg'], fit_method=fit_method, ranks=ranks,
                       replicates=kwargs['replicates'])
    return animaldata.ensemble, animaldata.mask


def log_llh(p: [float, float], samples: np.ndarray) -> float:
    mu, sigma = p
    # assumes truncated at zero, need to normalize the distribution
    norm_factor = 1 - scipy.stats.norm.cdf(0, mu, sigma)

    llh = np.exp(-(samples - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2) / norm_factor
    L = np.sum(np.log(llh))

    return -L


def get_norm_lick_rate(licks, zstates, feedback, choices, criterion, nbins=120):
    '''
    Get normalized lick rates
    '''
    lick_array = [licks[i] for i in range(len(zstates)) if zstates[i] in criterion['zstates'] \
                  and feedback[i] in criterion['feedback'] and choices[i] in criterion['choice']]

    lick_array = [np.array(elem) for elem in lick_array]
    # print(lick_array)

    if len(lick_array) == 0:
        licks_concat = []
        bins = []
        n = []
    else:
        licks_concat = np.hstack(lick_array)
        n, bins = np.histogram(licks_concat, bins=nbins)
        bin_width = (max(licks_concat) - min(licks_concat)) / nbins  # TODO: generalize to arbitrary lengths
        n = n / len(lick_array) / (bin_width)

    return bins[1:], n


def infer_truncated_normal(samples: np.ndarray) -> (float, float):
    '''
    Infer the mean and std of samples of a truncated normal at zero
    :param samples: the samples
    :return: mu, sigma
    '''
    # remove zeros and fit
    samples_pos = samples[samples > 0]
    results = scipy.optimize.minimize(log_llh, [np.mean(samples_pos), np.std(samples_pos)], (samples_pos))
    if results.success:
        return results.x[0], results.x[1]
    else:
        print('Warning: failed to infer mean and std, using default values')
        return np.mean(samples), np.std(samples)



def parse_kwargs(kwargs):
    '''
    Function for parsing default arguments
    :param kwargs: arguments passed to load_and_run
    :return: a dict of filled in params with default values
    '''
    if 'trialsubset' not in kwargs:
        kwargs['trialsubset'] = None

    if 'normtype' not in kwargs:
        kwargs['normtype'] = 'baseline'

    if 'baseline_frames' not in kwargs:
        kwargs['baseline_frames'] = np.arange(10)

    if 'keep_positive' not in kwargs:
        kwargs['keep_positive'] = 0

    if 'nonneg' not in kwargs:
        kwargs['nonneg'] = False

    if 'replicates' not in kwargs:
        kwargs['replicates'] = 3

    return kwargs


if __name__ == '__main__':
    # rootpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw'
    # animal = 'f01'
    # expdate = '030521'
    # fit_method = 'ncp_hals'
    # ranks = [1]
    # kwargs = {}
    # ensemble = load_and_run(rootpath, animal, expdate, fit_method, ranks, kwargs)
    # print(ensemble)
    samples = scipy.stats.truncnorm.rvs(-3 / 2.4, 100000, 3, 2.4, size=(10000))
    results = scipy.optimize.minimize(log_llh, [np.mean(samples), np.std(samples)], (samples))
    print(results.x)

