# A collection of useful routines for TCA analysis (batch)
import numpy as np
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
    return animaldata.ensemble


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
    rootpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw'
    animal = 'f01'
    expdate = '030521'
    fit_method = 'ncp_hals'
    ranks = [1]
    kwargs = {}
    ensemble = load_and_run(rootpath, animal, expdate, fit_method, ranks, kwargs)
    print(ensemble)
