# A class for ensemble data with trial and mask information
import glob

import numpy as np
import smartload.smartload as smart
import time
import matplotlib.pyplot as plt

class EnsembleData(object):
    def __init__(self, filepath, **kwargs):
        '''
        :param filepath: path to the ensemble pkl data
        :param loadbehav: whether to load the behavior data too
        '''

        if 'load_behavior' in kwargs:
            load_behavior = kwargs['load_behavior']
        else:
            load_behavior = True

        self.filepath = filepath

        self.animal, self.expdate, _, _, method = filepath.split('/')[-1].split('_')
        self.method = method[:-4]

        # Load the ensemble
        data = smart.load_pickle(filepath)
        self.ensemble = data['ensemble']

        ranks = sorted(self.ensemble.results)
        self.nreps = len(self.ensemble.results[ranks[0]])

        self.rootpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw'
        datapath = f'{self.rootpath}/extracted/{self.animal}/allData_extracted_{self.animal}_{self.expdate}pix.mat'
        # print('Loading the behavioral data...')
        behavdata = smart.loadmat(datapath, vars=['trialInfo'])
        self.feedback = behavdata['trialInfo']['feedback']
        self.choices = behavdata['trialInfo']['responses']
        self.targets = behavdata['trialInfo']['target']

        if load_behavior:
            self.load_state_information()
            self.load_wheel_lick_information()

        self.load_mask(data)


    def load_wheel_lick_information(self) -> None:
        '''
        Load the wheel and lick information from the processed/raw/behavior folder
        :return: None
        '''
        # Load the lick/wheel information
        metricspath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw/behavior'
        metrics_filepath = f'{metricspath}/20{self.expdate[4:]}-{self.expdate[:2]}-{self.expdate[2:4]}_{self.sessnum}_{self.animal.upper()}/behavior_metrics.mat'
        metricdata = smart.loadmat(metrics_filepath)
        self.licks = metricdata['lickAligned']
        self.wheel = metricdata['wheelSpeed']
        self.wheelTimes = metricdata['wheelSpeedTime']

        # check number of blocks
        nblocks = np.sum(np.diff(self.targets) != 0) + 1
        assert (len(self.zstates) == nblocks)

        # Verify number of trials
        ranks = sorted(self.ensemble.results)
        Ntrials = self.ensemble.factors(ranks[0])[0].factors[2].shape[0]
        assert (len(self.feedback) == Ntrials)
        assert (len(self.choices) == Ntrials)
        assert (len(self.targets) == Ntrials)



    def load_state_information(self):
        '''
        Load the state information
        '''
        # Load the state information
        hmm_filepath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/PaperFigures/code/blockhmm/opto_analysis/optodata/061522defaultK6/wf_hmm_info.mat'
        hmminfo = smart.loadmat(hmm_filepath)
        animals = np.array([item['animal'] for item in hmminfo['animalinfo']])
        animal_idx = np.where(animals == self.animal)[0]
        assert len(animal_idx) == 1

        # find the corresponding session
        sessdate_lst = np.char.lower(hmminfo['animalinfo'][animal_idx[0]]['sessnames'])
        filename = f'20{self.expdate[4:]}-{self.expdate[:2]}-{self.expdate[2:4]}_1_{self.animal.lower()}_block.mat'
        # print('filename:', filename)
        fileidx = np.where(sessdate_lst == filename)[0]
        assert len(fileidx) == 1

        self.zstates = hmminfo['animalinfo'][animal_idx[0]]['zclassified'][fileidx[0]]

        # check how many behavioral sessions were done on that day
        rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox'
        behavfolder = f'{rigboxpath}/{self.animal}/20{self.expdate[4:]}-{self.expdate[:2]}-{self.expdate[2:4]}/*/*Block.mat'
        files = glob.glob(behavfolder)
        # print('--')
        # print(files)

        # handle special cases with multiple sessions. i.e. len(files) > 1
        if self.animal.lower() == 'e57' and self.expdate == '021721':
            self.zstates = self.zstates[-5:]
            self.sessnum = 3
        elif self.animal.lower() == 'e57' and self.expdate == '030221':
            self.zstates = self.zstates[-9:]
            self.sessnum = 3
        elif self.animal.lower() == 'e57' and self.expdate == '022321':
            self.zstates = self.zstates[-7:]
            self.sessnum = 2
        elif self.animal.lower() == 'f02' and self.expdate == '030121':
            self.zstates = self.zstates[-7:]
            self.sessnum = 2
        elif self.animal.lower() == 'f03' and self.expdate == '032221':
            self.zstates = self.zstates[:10]
            self.sessnum = 1
        elif self.animal.lower() == 'f25' and self.expdate == '112321':
            self.zstates = self.zstates[:]
            self.sessnum = 2
        elif self.animal.lower() == 'f25' and self.expdate == '112621':
            self.zstates = self.zstates[:]
            self.sessnum = 2
        else:
            self.sessnum = 1
            assert (len(files) == 1)  # TODO: take care of this case


    def load_mask(self, data):
        if 'mask' not in data:
            # TODO: Manually load the mask information
            templatepath = f'{self.rootpath}/templateData/{self.animal}/templateData_{self.animal}_{self.expdate}pix.mat'
            # print('Mask not found, loading from template file...')

            templatedata = smart.loadmat(templatepath)
            self.mask = (np.abs(templatedata['template']['atlas']) < 300) & (templatedata['template']['atlas'] != 0)

            # Check that the dimensions are consistent with the data
            ranks = sorted(self.ensemble.results)
            W, _, _ = self.ensemble.factors(ranks[0])[0].factors
            assert W.shape[0] == np.sum(self.mask)
            # print('Mask loaded')

        else:
            self.mask = data['mask']


    def visualize_trials(self) -> None:
        '''
        Plot the choices and outcomes of the session
        :return: None
        '''
        tpoints = np.arange(len(self.choices))
        plt.figure(figsize=(9, 5))
        plt.plot(tpoints[self.feedback == 1], self.targets[self.feedback == 1], 'bo')
        plt.plot(tpoints[self.feedback == 0], self.targets[self.feedback == 0], 'rx')


    def reconstruct(self, rank, Nreps):
        '''
        Reconstruct the raw data from the factors
        :param rank: int, rank of model
        :param Nreps: int, the n of replicate
        :return: array of size Npix x T x Ntrials
        '''
        return self.ensemble.results[rank][Nreps].factors.full()


    def make_square_matrix(self, input):
        '''
        Make a square matrix using the stored mask information
        :param input: array of size Ninputs x T x Ntrials
        :return: output of size N1 x N2 x T x Ntrials
        '''
        output = np.zeros((self.N1 * self.N2, self.T, len(self.trialsubset)))
        output[self.mask_unroll,:,:] = input
        return np.reshape(output, (self.N1, self.N2, self.T, len(self.trialsubset)))


    def get_objectives(self):
        return self.ensemble.get_objectives()

    def get_similarities(self):
        return self.ensemble.get_similarities()

    def get_factors(self, rank):
        '''
        Return the corresponding factors
        :param rank: num factors
        :param nreps: num replicates
        :return: a list of (W, B, A) factors, one tuple
        for each rep he factors W, B, A as a tuple
        W has shape Nneurons x Nfactors
        B has shape T x Nfactors
        A has shape Ntrials x Nfactors
        '''
        self.ensemble._check_rank(rank)
        return [self.ensemble.factors(rank)[i].factors for i in range(self.nreps)]


    def get_temporal_all_reps(self, rank):
        '''
        Get temporal factors for all reps
        :param rank: int
        :return: an np array of size Nreps x T x rank of
        all temporal factors (including all reps in the specified rank
        '''
        all_factors = self.get_factors(rank)
        return np.array([elem[1] for elem in all_factors])


    def get_trial_all_reps(self, rank):
        '''
        Get the trial factors for all reps
        :param rank: int
        :return: an np array of size Nreps x Ntrials x rank
        of all trial factors (including all reps in the specified rank)
        '''
        all_factors = self.get_factors(rank)
        arr = np.array([elem[2] for elem in all_factors])
        return np.transpose(arr, (1,2,0))

    def sort_trial_factors(self, rank):
        '''
        Sort trial factors into positions in the block
        :param rank: int, rank of the ensemble of interest
        :return: TF_blockPos, a list T of lists, where T[0] is a list
        of all the factors occurring at position 0
        '''
        # build the 'position' array indicating the position of the trial in a block
        position = []
        count = 0
        for i in range(len(self.targets) - 1):
            position.append(count)
            if self.targets[i] != self.targets[i + 1]:
                count = 0
            else:
                count += 1

        position.append(count)
        position = np.array(position)

        assert(len(position) == len(self.targets))

        trial_factor_arr = self.get_trial_all_reps(rank)

        TF_blockPos = []
        for i in range(max(position)):
            trial_factor_in_pos = trial_factor_arr[position == i, :, :]
            TF_blockPos.append(trial_factor_in_pos)

        return TF_blockPos, position









    def _find_variance_explained(self, rank, nrep):
        '''
        For the replicate of a given rank, find the variance explained
        by each TCA component
        :param rank: int
        :param nrep: int
        :return: array of dimension rank x 1
        '''
        all_factors = self.get_factors(rank)
        W, B, A = all_factors[nrep]
        lambdas = []
        for i in range(W.shape[1]):
            Wvec = W[:,i]
            Bvec = B[:,i]
            Avec = A[:,i]
            lambdas.append(np.linalg.norm(Wvec) * np.linalg.norm(Bvec) * np.linalg.norm(Avec))

        return lambdas


    def get_spatial_all_reps(self, rank):
        '''
        Get all the spatial factors with given rank, all reps included
        :param rank:
        :return: list of all factors
        '''
        return np.array([self._get_spatial_unmasked(rank, i) for i in range(self.nreps)])


    def _get_spatial_unmasked(self, rank, nrep):
        '''
        Unmask and visualize spatial factor
        :param rank: int
        :param nrep: int
        :return: a list of N1 x N2 arrays
        '''
        assert self.mask is not None
        N1, N2 = self.mask.shape
        W, _, _ = self.ensemble.factors(rank)[nrep].factors
        mask_unroll = self.mask.ravel()
        unmasked = np.zeros((N1 * N2, W.shape[1]))
        unmasked[mask_unroll == 1, :] = W
        unmasked = np.reshape(unmasked, (N1, N2, -1))
        return unmasked








