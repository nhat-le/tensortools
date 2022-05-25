# For defining object class for imaging data
# that can be conveniently manipulated

import numpy as np
import smartload.smartload as smart
import tensortools as tt
import time

class ImagingData(object):
    '''
    Class for imaging data
    '''
    def __init__(self, rootpath, animal, expdate, trialsubset=None):
        '''
        Initalize the object
        :param rootpath: path to the processed folder
        :param animal: lowercase string, animal name
        :param expdate: expdate string, such as '030421'
        :param trialsubset: subset of trials to fit the TCA
        '''
        self.animal = animal
        self.expdate = expdate

        datapath = f'{rootpath}/extracted/{animal}/allData_extracted_{animal}_{expdate}pix.mat'
        templatepath = f'{rootpath}/templateData/{animal}/templateData_{animal}_{expdate}pix.mat'

        start = time.time()
        data = smart.loadmat(datapath)
        templatedata = smart.loadmat(templatepath)
        end = time.time()
        print(f'Finished loading in {end - start} secs')

        self.data = data['allData']['data']
        self.feedback_full = data['trialInfo']['feedback']
        self.N1, self.N2, self.T, self.Ntrials = self.data.shape
        assert(len(self.feedback_full) == self.Ntrials)

        # if specified, only pick a subset of the data to fit
        if trialsubset is not None:
            self.trialsubset = trialsubset
        else:
            self.trialsubset = np.arange(self.Ntrials)

        self.masktemp = (np.abs(templatedata['template']['atlas']) < 300) & (templatedata['template']['atlas'] != 0)

        if self.masktemp.shape[0] != self.N1 or self.masktemp.shape[1] != self.N2:
            print(f'Dimension mismatch: mask shape: {self.masktemp.shape[0]} x {self.masktemp.shape[1]}')
            print(f'Data shape: {self.N1} x {self.N2}')

        # pads the mask if there is a dimension mismatch
        self.mask = np.zeros((self.N1, self.N2))
        self.mask[:min(self.N1, self.masktemp.shape[0]), :min(self.N2, self.masktemp.shape[1])] = self.masktemp[:min(self.N1, self.masktemp.shape[0]),
                                                                     :min(self.N2, self.masktemp.shape[1])]
        self.mask_unroll = self.mask.ravel()

        self.datamat_unroll = np.reshape(self.data[:,:,:,self.trialsubset], (self.N1 * self.N2, self.T,
                                len(self.trialsubset)))[self.mask_unroll == 1, :, :]
        self.feedback = self.feedback_full[self.trialsubset]

        self.ensemble = None
        self.recons = None
        self.tca_ranks = None
        self.conditional_pred = None

    def change_trial_subset(self, subset):
        '''
        Change the trial subset and the 'active' data matrix to be fitted
        :param subset: the subset of trials of interest
        :return: nothing
        '''
        if subset is None:
            subset = np.arange(self.Ntrials)
        self.trialsubset = subset
        self.datamat_unroll = np.reshape(self.data[:,:,:,self.trialsubset], (self.N1 * self.N2, self.T, len(self.trialsubset)))[self.mask_unroll, :, :]
        self.feedback = self.feedback_full[self.trialsubset]

        # Clear all models
        self.ensemble = None
        self.recons = None
        self.tca_ranks = None
        self.conditional_pred = None



    def normalize(self, type='range', **kwargs):
        '''
        Normalize the date
        :param type: range means normalize to range 0 to 1,
        'baseline' means normalize to the baseline
        :param kwargs: arguments to be passed
        :return: nothing
        '''
        if type == 'range':
            # if 'min_prctile' in kwargs:
            #     min_prctile = kwargs['min_prctile']
            # else:
            #     min_prctile = 0
            min_prctile = kwargs['min_prctile'] if 'min_prctile' in kwargs else 0
            max_prctile = kwargs['max_prctile'] if 'max_prctile' in kwargs else 100

            minval = np.percentile(self.datamat_unroll, min_prctile, axis=(1, 2))[:, np.newaxis, np.newaxis]
            maxval = np.percentile(self.datamat_unroll, max_prctile, axis=(1, 2))[:, np.newaxis, np.newaxis]

            self.datamat_norm = (self.datamat_unroll - minval) / (maxval - minval)


        if type == 'baseline':
            # Perform baseline subtraction
            if 'baseline_frames' in kwargs:
                baseline_frames = kwargs['baseline_frames']
            else:
                baseline_frames = np.arange(10)

            keep_positive = kwargs['keep_positive'] if 'keep_positive' in kwargs else 0

            baseline = np.mean(self.datamat_unroll[:,baseline_frames,:], axis=(1,2))
            self.datamat_norm = self.datamat_unroll - baseline[:,np.newaxis, np.newaxis]

            if keep_positive:
                self.datamat_norm[self.datamat_norm < 0] = 0

            # return self.datamat_norm

    def compute_errors(self):
        '''
        Compute the errors (recons - raw) based on all TCA ensembles
        If multiple replicates, only compute for the first one
        :return: (nfactors, errors, baseline_error)
            nfactors: numpy array, should be the same as self.tca_ranks
            errors: mean squared errors corresponding to each nfactor
            baseline_error: mse of condition-aware model
        '''
        assert(self.tca_ranks is not None)
        nfactors = self.tca_ranks
        errors = []
        Nreplicates = 0
        for n in nfactors:
            recons, raw, cond = self.reconstruct(n, Nreplicates)
            errors.append(np.sqrt(np.linalg.norm(recons - raw)))

        if self.conditional_pred is not None:
            baseline_error = np.sqrt(np.linalg.norm(cond - raw))
        else:
            baseline_error = None

        return nfactors, errors, baseline_error





    def make_conditional_predictions(self):
        '''
        Perform a conditional predictoin based on reward/errors
        :return: matrix of shape Npixels x T x Ntrials
        '''
        # We only have two conditions for now: correct vs incorrect
        corrarr = self.datamat_norm[:, :, self.feedback == 1]
        incorr_arr = self.datamat_norm[:, :, self.feedback == 0]

        meancorr = np.mean(corrarr, axis=2)
        mean_incorr = np.mean(incorr_arr, axis=2)

        self.conditional_pred = np.zeros_like(self.datamat_norm)
        self.conditional_pred[:, :, self.feedback == 1] = meancorr[:, :, np.newaxis]
        self.conditional_pred[:, :, self.feedback == 0] = mean_incorr[:, :, np.newaxis]

        # return self.conditional_pred



    def tca_fit(self, nonneg=False, fit_method="cp_als", ranks=[1], replicates=1):
        '''
        Fit an ensemble of models
        :return: the ensemble object
        '''
        ensemble = tt.Ensemble(nonneg=nonneg, fit_method=fit_method)
        ensemble.fit(self.datamat_norm, ranks=ranks, replicates=replicates)
        self.tca_ranks = ranks
        self.ensemble = ensemble
        return ensemble

    def tca_crossval(self, test_fraction, ranks=[1], method='ncp_hals'):
        # Create random mask to holdout ~10% of the data at random.
        Npix, T, Ntrials = self.datamat_norm.shape
        mask = np.random.rand(Npix, T, Ntrials) > test_fraction

        # Fit nonnegative tensor decomposition.
        X = self.datamat_norm
        train_errors = []
        test_errors = []
        for rank in ranks:
            print('Fitting model with rank', rank)
            start = time.time()
            if method == 'ncp_hals':
                U = tt.ncp_hals(X, rank=rank, mask=mask, verbose=False)
            elif method == 'mcp_als':
                U = tt.mcp_als(X, rank=rank, mask=mask, verbose=False)
            else:
                raise ValueError('Invalid fitting method')
            Xhat = U.factors.full()
            # Compute norm of residuals on training and test sets.
            train_error = np.linalg.norm(Xhat[mask] - X[mask])
            test_error = np.linalg.norm(Xhat[~mask] - X[~mask])
            train_errors.append(train_error)
            test_errors.append(test_error)
            end = time.time()
            print('Time elapsed = ', end - start, 'secs')
            print('Train error = ', train_error, '. Test error =', test_error)

        return train_errors, test_errors

    def make_square_matrix(self, input):
        '''
        Make a square matrix using the stored mask information
        :param input: array of size Ninputs x T x Ntrials
        :return: output of size N1 x N2 x T x Ntrials
        '''
        output = np.zeros((self.N1 * self.N2, self.T, len(self.trialsubset)))
        output[self.mask_unroll,:,:] = input
        return np.reshape(output, (self.N1, self.N2, self.T, len(self.trialsubset)))


    def reconstruct(self, Nfactors, Nreplicates):
        '''
        Make reconstructions (square) using: (1) TCA model, (2) raw data, (3) conditional averaged data
        :param Nfactors: number of factors to use in the TCA model
        :param Nreplicates: number of replicates to use in the TCA model
        :return: the three arrays each of size N1 x N2 x T x Ntrials
        '''
        self.recons = self.ensemble.factors(Nfactors)[Nreplicates].full()
        recons_sq = self.make_square_matrix(self.recons)

        raw_sq = self.make_square_matrix(self.datamat_norm)

        cond_sq = self.make_square_matrix(self.conditional_pred)

        return recons_sq, raw_sq, cond_sq


if __name__ == '__main__':
    animal = 'f01'
    expdate = '030421'
    obj = ImagingData(animal, expdate)


