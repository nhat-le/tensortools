# Containing structures and routines to store multiple TCA modes
# together with aggregating and plotting functionalities
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


@dataclass
class TCASimple:
    '''
    A simple class of TCA mode that stores only the basic params
    '''
    animal: str
    session: str
    modeID: int
    repID: int
    spatial: np.ndarray
    temporal: np.ndarray
    dR2: np.ndarray


class GLMCollection(object):
    '''
    A class of GLM dR2 together with information about the session
    and type of mode
    '''
    def __init__(self, tcalst: list[TCASimple]):
        self.tca_modes = tcalst
        self.Nmodes = len(tcalst)

    @classmethod
    def from_animal_info(cls, animals, dates, reps, modes, spatials, temporals, glmdR2):
        '''
        :param animals: list of strings
        :param dates: list of strings
        :param reps: list[int]
        :param modes: list[int]
        :param spatials: list[np array]
        :param temporals: list[np array]
        :param glmdR2: list[np array]
        '''
        assert(len(animals) == len(dates) == len(reps) == len(modes) == len(spatials) == len(temporals))
        assert(len(animals) == glmdR2.shape[0])

        tca_modes = []
        for i in range(len(animals)):
            tca_mode = TCASimple(animals[i], dates[i], modes[i], reps[i], spatials[i], temporals[i], glmdR2[i])
            tca_modes.append(tca_mode)

        return cls(tca_modes)

    def get_glm_dR2means(self) -> np.ndarray:
        '''
        Compute the mean dr2 for each glm mode
        :return: dR2means array, size nmodes x nfeatures
        '''
        glmdR2_mean = []
        for mode in self.tca_modes:
            glmdR2_mean.append(np.mean(mode.dR2, axis=0))

        return np.array(glmdR2_mean)

    def cluster_dR2means(self, K: int, seed=126):
        '''
        Clustering of the mean dR2 vectors
        :return:
        '''
        glmdR2_mean = self.get_glm_dR2means()
        np.random.seed(seed)  # 126 is good
        mdl = KMeans(n_clusters=K)
        res = mdl.fit(glmdR2_mean)
        idxsort = np.argsort(res.labels_)
        return idxsort, res.labels_

    def filter_collection(self, idxlst: list[int]) -> GLMCollection:
        '''
        Returns a subcollection using the indices indicated
        :param idxlst: list of modes to filter out
        :return:
        '''
        filtered_modes = [self.tca_modes[i] for i in range(self.Nmodes) if i in idxlst]
        return GLMCollection(filtered_modes)

    def plot_spatial_temporal(self) -> None:
        '''
        Plot the spatial and temporal factors
        :return:
        '''
        fig, ax = plt.subplots(self.Nmodes, 3, figsize=(8, self.Nmodes))
        for i in range(self.Nmodes):
            ax[i][0].imshow(self.tca_modes[i].spatial)
            ax[i][1].plot(self.tca_modes[i].temporal)
            ax[i][2].plot(self.tca_modes[i].dR2.T, 'b')
            ax[i][1].set_ylabel(self.tca_modes[i].animal)
