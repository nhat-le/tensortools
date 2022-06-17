# Structures and utilities for keeping track and organizing the TCA factors
# and characterizing their properties
import numpy as np
import tensortools.custom.ensemble_data as Ens
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tqdm.notebook
import glob
import pandas as pd






class AnimalCollection(object):
    '''
    A collection of sessions of the same animal
    '''
    def __init__(self, animal, N, filepath=None, verbose=False):
        '''
        :param filepath: str (optional), the path to the pkl files with the fitted TCA ensembles
        if None, will use the default filepath
        :param animal: animal name, will only consider sessions with the animal pattern
        :param N: number of modes
        '''
        # Loads all sessions for the animal, each as a ModeCollection object
        if filepath is None:
            if N <= 6:
                filepath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors/nodelays_061322'
            elif N <= 12:
                filepath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors/nodelays_K7_061322'
            else:
                raise ValueError('Only Nclusters 1 to 12 fitted so far...')

        self.animal = animal
        self.Nmodes = N
        self.sessions = []

        filenames = glob.glob(f'{filepath}/{animal}_*.pkl')

        for path in tqdm.notebook.tqdm(filenames):
            if verbose:
                print(f'Processing session {path}...')
            session = ModeCollection(path, N)
            self.sessions.append(session)
            if verbose:
                print(sorted(session.ens.ensemble.results))

        self.expdates = np.array([item.ens.expdate for item in self.sessions])

    def get_ensemble_obj(self, verbose=False):
        '''
        Get the fit performance of the ensemble
        as a function of the number of clusters
        :return: obj, Nclusters
        '''
        filepath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors/nodelays_061322'
        filepathK7 = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors/nodelays_K7_061322'
        filenames = glob.glob(f'{filepath}/{self.animal}_*.pkl')
        filenamesK7 = []

        objs_all = {}
        sims_all = {}

        # Get the corresponding list of K7 files
        for path in filenames:
            fname = path.split('/')[-1]
            animal = fname.split('_')[0]
            datestr = fname.split('_')[1]
            filenamesK7.append(f'{filepathK7}/{animal}_{datestr}_ensemble_ncp_halsK7to12.pkl')

        # Get objectives and similarities for N = 1 to 6
        for Nmodes in range(1, 7):
            objs_lst = []
            sims_lst = []
            for path in filenames:
                if verbose:
                    print(path)
                session = ModeCollection(path, Nmodes)
                objs = session.ens.ensemble.objectives(Nmodes)
                sims = session.ens.ensemble.similarities(Nmodes)[1:]
                objs_lst.extend(objs)
                sims_lst.extend(sims)
            objs_all[Nmodes] = objs_lst
            sims_all[Nmodes] = sims_lst

        # Get objectives for N = 7 to 12
        for Nmodes in range(7, 13):
            objs_lst = []
            sims_lst = []
            for path in filenamesK7:
                if verbose:
                    print(path)
                session = ModeCollection(path, Nmodes)
                objs = session.ens.ensemble.objectives(Nmodes)
                sims = session.ens.ensemble.similarities(Nmodes)[1:]

                objs_lst.extend(objs)
                sims_lst.extend(sims)

            objs_all[Nmodes] = objs_lst
            sims_all[Nmodes] = sims_lst


        return objs_all, sims_all



class ModeCollection(object):
    '''
    A collection of TCA modes that were classified
    '''
    def __init__(self, filepath, N):
        '''
        :param filepath: str, path to the ensemble file
        :param N: number of modes of TCA fit
        '''
        self.Nmodes = N
        self.ens = Ens.EnsembleData(filepath)
        self.animal = self.ens.animal
        self.method = self.ens.method

        self.spatials = self.ens.get_spatial_all_reps(N)
        self.temporals = self.ens.get_temporal_all_reps(N)
        self.trials = self.ens.get_trial_all_reps(N) # size: Ntrials x N x Nreps

        self.Nreps, self.N1, self.N2, _ = self.spatials.shape
        self.T = self.temporals.shape[1]
        self.Ntrials = self.trials.shape[1]

        self.spatials = np.transpose(self.spatials, (1, 2, 3, 0)) # size: N1 x N2 x N x Nreps
        self.temporals = np.transpose(self.temporals, (1, 2, 0)) # size: T x N x Nreps

        # Trial history regression
        self.regression_estimates = []
        self.regression_stderrs = []

        for mode_id in range(self.Nmodes):
            for rep_id in range(self.Nreps):
                params, stderr = self.do_trial_history_regression(mode_id, rep_id, Nback=5)
                self.regression_estimates.append(params)
                self.regression_stderrs.append(stderr)

        self.copy = self.regression_estimates.copy()

        self.regression_estimates = np.array(self.regression_estimates).reshape((self.Nmodes, self.Nreps, -1))
        self.regression_stderrs = np.array(self.regression_stderrs).reshape((self.Nmodes, self.Nreps, -1))


    # Try different constructor
    # @classmethod
    # def alternative_construct(cls):
    #     filepath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors/nodelays_061322/e57_022621_ensemble_ncp_hals.pkl'
    #     obj = cls(filepath, 6)
    #     return obj


    def get_modes(self, modeID: int, repID: int):
        '''
        Get a specific mode from the ensemble.
        Returns a tuple of spatial, temporal and trial factors
        :param repID: int, id of the rep, must be < Nreps
        :param modeID: int, id of the mode, must be < Nmodes
        '''
        # print(repID, modeID, self.Nmodes, self.Nreps)
        assert(repID < self.Nreps)
        assert(modeID < self.Nmodes)

        # Get spatial factor
        spatial = self.spatials[:,:, modeID, repID]
        temporal = self.temporals[:, modeID, repID]
        trial = self.trials[:, modeID, repID]

        return spatial, temporal, trial








    def do_trial_history_regression(self, mode_id, rep_id, Nback=5):
        '''
        :param mode_id: id of mode from 0 to Nmodes - 1
        :param rep_id: id of rep from 0 to Nreps - 1
        :param Nback: # of trials back for regression
        :return: the results of the regression, as two lists: the estimates and the standard error
        '''
        reward_arrs = []
        choice_arrs = []
        cr_arrs = []

        for i in range(Nback + 1):
            if i == 0:
                reward_ni = self.ens.feedback[Nback - i:].astype('float') * 2 - 1
                choice_ni = self.ens.choices[Nback - i:].astype('float')
            else:
                reward_ni = self.ens.feedback[Nback - i: -i].astype('float') * 2 - 1
                choice_ni = self.ens.choices[Nback - i: -i].astype('float')
            cr_ni = reward_ni * choice_ni
            reward_arrs.append(reward_ni)
            choice_arrs.append(choice_ni)
            cr_arrs.append(cr_ni)

        const = np.ones(len(reward_ni))

        Xmat = np.vstack([const] + reward_arrs + choice_arrs + cr_arrs).T

        mdl = sm.OLS(self.trials[Nback:, mode_id, rep_id], Xmat)
        res = mdl.fit()

        return res.params, res.bse

    def plot_spatial_temporal(self):
        '''
        Plot the spatial and temporal factors
        :return:
        '''
        fig, ax = plt.subplots(self.Nreps * 2, self.Nmodes, figsize=(8, 5))
        for i in range(self.Nreps):
            for j in range(self.Nmodes):
                ax[i * 2][j].imshow(self.spatials[:, :, j, i])
                ax[i * 2][j].set_xticks([])
                ax[i * 2][j].set_yticks([])
                ax[i * 2 + 1][j].plot(self.temporals[:, j, i])


    def plot_trials(self, rep_id):
        '''
        Plot the trial factors
        :return:
        '''
        trial_factor_arr = self.ens.get_trial_all_reps(self.Nmodes)
        trialidx = np.arange(len(self.ens.feedback))
        fig, ax = plt.subplots(self.Nmodes, 2, figsize=(10, 10))

        # color the points by what criterion
        criterion = self.ens.feedback == 1

        for i in range(self.Nmodes):
            ax[i][0].plot((self.ens.targets + 1) / 2)
            ax[i][0].plot(trialidx[criterion], trial_factor_arr[criterion, i, rep_id], 'b.')
            ax[i][0].plot(trialidx[~criterion], trial_factor_arr[~criterion, i, rep_id], 'r.')
            ax[i][1].violinplot([trial_factor_arr[~criterion, i, rep_id],
                                 trial_factor_arr[criterion, i, rep_id]], showmedians=True)

        plt.xlabel('Trials')
        plt.ylabel('Activity')

        plt.tight_layout()



def get_tca_variance_explained(animal, verbose=False):
    '''
    Get the variance explained as a function of number of TCA modes
    :param animal: str, animal name
    :return: two arrays representing the mean and std of variance explained
    '''
    if verbose:
        print(f'Processing animal {animal}...')
    col = AnimalCollection(animal, N=6)
    objs_all, sims_all = col.get_ensemble_obj()
    objs, sims = [], []
    for i in range(1, 13):
        objs.extend(objs_all[i])
        sims.extend(sims_all[i])

    objs_arr = np.reshape(objs, (12, -1, 3))
    sims_arr = np.reshape(sims, (12, -1, 2))

    obj_meanarr = 1 - np.mean(objs_arr ** 2, axis=(1, 2))
    obj_stdarr = np.std(objs_arr, axis=(1, 2))
    sim_meanarr = np.mean(sims_arr, axis=(1, 2))
    sim_stdarr = np.std(sims_arr, axis=(1, 2))

    return obj_meanarr, obj_stdarr, sim_meanarr, sim_stdarr


class TCAMode(object):
    '''
    A class to store the properties of a TCA mode
    '''
    def __init__(self, collection: AnimalCollection, session: str, modeID: int, repID: int):
        '''
        Extract a TCA mode from a collection
        :param collection: an AnimalCollection object
        :param repID: int, ID of the rep
        :param modeID: int, ID of the mode
        :param sessionID: int, ID of the session
        '''
        # Find the session ID from the session name
        sessionID = np.where(collection.expdates == session)[0]
        assert(len(sessionID) == 1)
        sessionID = sessionID[0]

        session_obj = collection.sessions[sessionID]
        self.spatial, self.temporal, self.trial = session_obj.get_modes(modeID, repID)



class GrandCollection(object):
    '''
    A collection of different animal collections representing
    all collected experimental data
    '''
    def __init__(self, collections: list[AnimalCollection]):
        self.collections = collections

    # alternative construction
    @classmethod
    def from_namelist(cls, Nmodes_lst, animals):
        collections = [AnimalCollection(animal, N=Nmodes) for animal, Nmodes in zip(animals, Nmodes_lst)]
        return cls(collections)

    def filter_modes(self, criterion: dict):
        '''
        Filter modes based on regional criterion
        :param criterion: a dict, consisting of region 0/1 filters
        :return: a list of TCAMode objects, representing the extracted modes from all animals
        that fulfill the criteiron
        '''

        extracted_animals = []
        extracted_dates = []
        allmodes = []
        for collection in self.collections:
            animal = collection.animal
            Nmodes = collection.Nmodes

            modetbl = pd.read_excel('tca_annotations_061422.xlsx', animal)
            modetbl = modetbl.fillna(0)

            assert (min(modetbl.Nmodes) == Nmodes)
            assert (max(modetbl.Nmodes) == Nmodes)

            # Gather all modes of the same type
            criterion_idx = np.isin(modetbl.Frontal, criterion['frontal']) & \
                        np.isin(modetbl.Visual, criterion['visual']) & \
                        np.isin(modetbl.Motor, criterion['motor']) & \
                        np.isin(modetbl.RSC, criterion['rsc']) & \
                        (modetbl.Flag == 0)

            subtbl = modetbl[criterion_idx].reset_index(drop=True)

            for row in range(len(subtbl)):
                session = subtbl.Session[row][1:-1]
                assert (len(session) == 6)
                modeID = subtbl.ModeID[row] - 1
                repID = subtbl.Nreps[row]

                mode_obj = TCAMode(collection, session, modeID=modeID, repID=repID)
                print(animal, session, modeID, repID)
                extracted_animals.append(animal)
                extracted_dates.append(session)

                allmodes.append(mode_obj)

        print(f'Total modes extracted = {len(allmodes)}')

        spatials = [item.spatial for item in allmodes]
        temporals = [item.temporal for item in allmodes]
        return spatials, temporals, extracted_animals, extracted_dates





