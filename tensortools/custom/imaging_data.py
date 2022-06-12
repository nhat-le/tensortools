# For defining object class for imaging data
# that can be conveniently manipulated

import numpy as np
import smartload.smartload as smart
import tensortools as tt
import time

import tensortools.tensors
from tensortools.custom.ensemble_data import EnsembleData


class Template(object):
    '''
    Class to store the template atlas
    '''
    def __init__(self, templatepath):
        '''
        Initiate the template by loading from the template path
        :param templatepath: a string
        '''
        templatedata = smart.loadmat(templatepath)
        self.atlas = templatedata['template']['atlas']
        self.names = templatedata['template']['areaStrings']
        self.ids = templatedata['template']['areaid']



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
        self.template = Template(templatepath)

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

        self.masktemp = (np.abs(self.template.atlas) < 300) & (self.template.atlas != 0)
        # self.masktemp = (np.abs(templatedata['template']['atlas']) < 300) & (templatedata['template']['atlas'] != 0)

        if self.masktemp.shape[0] != self.N1 or self.masktemp.shape[1] != self.N2:
            print(f'Dimension mismatch: mask shape: {self.masktemp.shape[0]} x {self.masktemp.shape[1]}')
            print(f'Data shape: {self.N1} x {self.N2}')

        # pads the mask and template atlas if there is a dimension mismatch
        self.mask = np.zeros((self.N1, self.N2))
        self.mask[:min(self.N1, self.masktemp.shape[0]), :min(self.N2, self.masktemp.shape[1])] = self.masktemp[:min(self.N1, self.masktemp.shape[0]),
                                                                     :min(self.N2, self.masktemp.shape[1])]
        atlaspad = np.zeros((self.N1, self.N2))
        atlaspad[:min(self.N1, self.masktemp.shape[0]), :min(self.N2, self.masktemp.shape[1])] = self.template.atlas[:min(self.N1, self.masktemp.shape[0]),
                                                                     :min(self.N2, self.masktemp.shape[1])]
        self.template.atlas = atlaspad



        self.mask_unroll = self.mask.ravel().astype('int')

        self.datamat_unroll = np.reshape(self.data[:,:,:,self.trialsubset], (self.N1 * self.N2, self.T,
                                len(self.trialsubset)))[self.mask_unroll == 1, :, :]
        self.feedback = self.feedback_full[self.trialsubset]

        self.ensemble = None
        self.recons = None
        self.tca_ranks = None
        self.conditional_pred = None

        self.datamat_norm = None

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


    def compute_bilateral_correlations(self, region_name):
        '''
        Compute the correlations between the regions on left and right side
        :param region_name: str, a region name (without L, R)
        :return: a single float, correlation, or np.nan if regions don't exist
        '''
        left_name = 'L-' + region_name
        right_name = 'R-' + region_name

        if left_name not in self.template.names or right_name not in self.template.names:
            return np.nan

        raw_sq = self.make_square_matrix(self.datamat_norm)
        # print(raw_sq.shape)

        activityL = self.extract_region_activity(left_name, custom_mat=raw_sq)
        activityR = self.extract_region_activity(right_name, custom_mat=raw_sq)

        # compute the correlation
        mean_actL = np.mean(activityL[20:25, :], axis=0)
        mean_actR = np.mean(activityR[20:25, :], axis=0)

        return np.corrcoef(mean_actR, mean_actL)[0, 1]

    def color_atlas(self, regnames, vals):
        '''
        given values corresponding to individual regions in an atlas
        Use the template to create a 'colored' atlas with regions
        replaced with their corresponding values
        :param regnames: array, names of the regions, if L/R omitted, will
        color both sides
        :param vals: the values corresponding to the regions, requires
        len(vals) == len(regnames)
        :return: the colored atlas of size N1 x N2
        '''
        assert(len(regnames) == len(vals))
        arr = np.zeros_like((self.template.atlas))
        for name, val in zip(regnames, vals):
            if name[:2] != 'L-' or name[:2] != 'R-': #no side given, will color both sides
                nameL = 'L-' + name
                nameR = 'R-' + name
                idxL = np.where(self.template.names == nameL)
                idxR = np.where(self.template.names == nameR)

                arr[self.template.atlas == self.template.ids[idxL]] = val
                arr[self.template.atlas == self.template.ids[idxR]] = val

            else: # color one side only
                idx = np.where(self.template.names == name)
                arr[self.template.atlas == self.template.ids[idx]] = val

        return arr





    def extract_region_activity(self, name, custom_mat=None, custom_mask=None):
        '''
        Extract the region activity based on the template and region name
        If custom_mat is given, will extract the regional activity from the custom array instead
        else, if custom_mat is None (default), will extract the activity from the self.data
        If custom_mask is provided (size N1 x N2 array), will use the mask provided instead
        :param name: str, region name
        :return: an array of size T x Ntrials: average activity of all pixels in the region
        '''
        if custom_mask is None:
            idx = np.where(self.template.names == name)[0]
            assert (len(idx) == 1)
            regionID = self.template.ids[idx]
            region_mask = (self.template.atlas == regionID).ravel()
        else:
            region_mask = custom_mask.ravel()

        if custom_mat is None:
            if self.datamat_norm is not None:
                region_data_unroll = np.reshape(self.datamat_norm, (self.N1 * self.N2, self.T, -1))[region_mask == 1, :, :]
            else:
                region_data_unroll = np.reshape(self.data, (self.N1 * self.N2, self.T, -1))[region_mask == 1, :, :]

        else:
            assert(custom_mat.shape[0] == self.N1)
            assert(custom_mat.shape[1] == self.N2)
            assert(custom_mat.shape[2] == self.T)
            assert(custom_mat.shape[3] == self.Ntrials)
            region_data_unroll = np.reshape(custom_mat, (self.N1 * self.N2, self.T, -1))[region_mask == 1, :, :]


        return np.mean(region_data_unroll, axis=0)





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


    def tca_load(self, method):
        '''
        Load a tca ensemble that has already been saved in a pkl file
        :return: ensembleData object, also stored as a property
        '''
        rootpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors/052422-baseline-corrected'
        filepath = rootpath + f'/{self.animal}_{self.expdate}_ensemble_{method}.pkl'
        ensemble = EnsembleData(filepath)
        self.ensemble = ensemble
        self.ranks = sorted(ensemble.ensemble.results)
        self.is_loaded_data = True


    def tca_fit(self, nonneg=False, fit_method="cp_als", ranks=[1], replicates=1):
        '''
        Fit an ensemble of models
        :return: the ensemble object
        '''
        ensemble = tt.Ensemble(nonneg=nonneg, fit_method=fit_method)
        ensemble.fit(self.datamat_norm, ranks=ranks, replicates=replicates)
        self.tca_ranks = ranks
        self.ensemble = ensemble
        self.is_loaded_data = False
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
        output[self.mask_unroll == 1,:,:] = input
        return np.reshape(output, (self.N1, self.N2, self.T, len(self.trialsubset)))


    def reconstruct(self, Nfactors, Nreplicates, components=None):
        '''
        Make reconstructions (square) using: (1) TCA model, (2) raw data, (3) conditional averaged data
        :param Nfactors: number of factors to use in the TCA model
        :param Nreplicates: number of replicates to use in the TCA model
        :param components: list of ints, if specified, only construct with the indicated components
        :return: the three arrays each of size N1 x N2 x T x Ntrials
        '''
        if self.is_loaded_data:
            KTensor = self.ensemble.ensemble.factors(Nfactors)[Nreplicates]
            cond_sq = None
        else:
            KTensor = self.ensemble.factors(Nfactors)[Nreplicates]
            cond_sq = self.make_square_matrix(self.conditional_pred)

        # Select relevant components in the tensor
        if components is not None:
            NeuronsMat, Tmat, Bmat = KTensor.factors
            NeuronsMat = NeuronsMat[:, components]
            Tmat = Tmat[:, components]
            Bmat = Bmat[:, components]
            KTensor = tensortools.tensors.KTensor([NeuronsMat, Tmat, Bmat])


        self.recons = KTensor.full()
        recons_sq = self.make_square_matrix(self.recons)
        raw_sq = self.make_square_matrix(self.datamat_norm)

        return recons_sq, raw_sq, cond_sq


if __name__ == '__main__':
    animal = 'f01'
    expdate = '030421'
    print('done')
    # obj = ImagingData(animal, expdate)


