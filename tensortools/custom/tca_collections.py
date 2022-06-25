# Containing structures and routines to store multiple TCA modes
# together with aggregating and plotting functionalities
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class SpatialAnnotation:
    '''
    A class for annotations (manual) of spatial TCA clusters
    '''
    motor: bool
    visual: bool
    frontal: bool
    rsc: bool
    midline: bool
    diffuse: bool
    side: int
    flag: bool

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
    annotation: SpatialAnnotation


class GLMCollection(object):
    '''
    A class of GLM dR2 together with information about the session
    and type of mode
    '''
    def __init__(self, tcalst: list[TCASimple]):
        self.tca_modes = tcalst
        self.Nmodes = len(tcalst)

    @classmethod
    def from_animal_info(cls, animals: list[str], dates: list[str], reps: list[int],
                         modes: list[int], spatials, temporals, glmdR2):
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
            # Read annotations from excel file
            modetbl = pd.read_excel('tca_annotations_062022.xlsx', animals[i])
            modetbl = modetbl.fillna(0)
            # print(modetbl.shape)

            datestr = f'"{dates[i]}"'
            # print(animals[i], reps[i], modes[i], dates[i])
            mode_row = modetbl[(modetbl.Session == datestr) & (modetbl.Nreps == reps[i]) & (modetbl.ModeID == modes[i] + 1)]
            mode_row = mode_row.reset_index()
            # print(mode_row.shape)
            assert(len(mode_row) == 1)
            annotation = SpatialAnnotation(motor=mode_row.Motor[0], visual=mode_row.Visual[0], frontal=mode_row.Frontal[0],
                                           rsc=mode_row.RSC[0], midline=mode_row.Midline[0], diffuse=mode_row.Diffuse[0],
                                           side=mode_row.Side[0], flag=mode_row.Flag[0])


            tca_mode = TCASimple(animals[i], dates[i], modes[i], reps[i], spatials[i], temporals[i], glmdR2[i], annotation)
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


    def plot_regional_distribution(self) -> dict:
        '''
        Plot the counts of the region annotations (motor, visual, rsc, frontal, midline)
        :return: the dictionary of counts
        '''
        motor_counts = np.sum([mode.annotation.motor for mode in self.tca_modes])
        visual_counts = np.sum([mode.annotation.visual for mode in self.tca_modes])
        frontal_counts = np.sum([mode.annotation.frontal for mode in self.tca_modes])
        rsc_counts = np.sum([mode.annotation.rsc for mode in self.tca_modes])
        midline_counts = np.sum([mode.annotation.midline for mode in self.tca_modes])

        # plt.figure()
        plt.plot([motor_counts, visual_counts, frontal_counts, rsc_counts, midline_counts], 'bo')
        plt.xticks(np.arange(5), ['Motor', 'Visual', 'Frontal', 'RSC', 'Midline'], rotation=45)

        return dict(motor=motor_counts, visual=visual_counts, rsc=rsc_counts, frontal=frontal_counts,
                    midline=midline_counts)

