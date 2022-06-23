# A script with utilities for fitting of GLMs and doing cross-validation
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from tensortools.custom.ensemble_properties import  TCASimple

class GLM(object):
    '''
    A class for GLM fitting
    '''
    def __init__(self, X, y, mdltype='ols', nsplits=5):
        '''
        :param X: np array, the X-variable, size: Nsamples x Nfeatures
        :param y: the y variable, size: Nsamples x 1
        :param mdltype: type of model to fit the data
        '''
        self.X = X
        self.y = y
        self.nsamples, self.nfeatures = X.shape

        # Make the partial models
        self.partials = []
        for i in range(self.nfeatures):
            idx_lst = np.array([elem for elem in range(self.nfeatures) if elem != i])
            self.partials.append(self.X.copy()[:, idx_lst])

        self.nsplits = nsplits
        self.mdltype = mdltype

        # Pick the cross-validation folds
        self.train_folds = []
        self.test_folds = []

        # Find R2 and dR2 for each fold
        R2_all = []
        dR2_all = []
        for idx_train, idx_test in KFold(n_splits=nsplits).split(X, y):
            self.train_folds.append(idx_train)
            self.test_folds.append(idx_test)
            R2, dR2 = self.fit_and_evaluate(idx_train, idx_test)
            R2_all.append(R2)
            dR2_all.append(dR2)

        self.R2 = np.array(R2_all)
        self.dR2 = np.array(dR2_all)


    def custom_model(self, ytrain, Xtrain, mdltype='ols'):
        if mdltype == 'ols':
            return sm.OLS(ytrain, Xtrain)
        elif mdltype == 'poisson':
            return sm.GLM(ytrain, Xtrain, family=sm.families.Poisson())

    def get_R2(self, Xtrain,ytrain, Xtest, ytest, mdltype='ols'):
        mdl = self.custom_model(ytrain, Xtrain, mdltype)
        # Fit and predict
        res = mdl.fit()
        preds = res.predict(Xtest)
        R2 = np.corrcoef(preds, ytest)[0, 1] ** 2
        return R2




    def fit_and_evaluate(self, idx_train: list, idx_test: list):
        '''
        :param idx_train: list of idx in the train set
        :param idx_test: list of idx in the test set
        :return: array of delta-R2 in this run
        '''

        Xtrain, ytrain, Xtest, ytest = self.X[idx_train], self.y[idx_train], self.X[idx_test], self.y[idx_test]

        # Fit the full model
        mdl = self.custom_model(ytrain, Xtrain, mdltype=self.mdltype)

        # Fit and predict
        R2 = self.get_R2(Xtrain, ytrain, Xtest, ytest)

        # Fit the partial models
        partialR2_lst = []
        for Xpartial in self.partials:
            Xtrain, Xtest = Xpartial[idx_train], Xpartial[idx_test]
            partialR2 = self.get_R2(Xtrain, ytrain, Xtest, ytest)
            partialR2_lst.append(R2 - partialR2)

        return R2, partialR2_lst


class GLMCollection(object):
    '''
    A class of GLM dR2 together with information about the session
    and type of mode
    '''
    def __init__(self, animals, dates, reps, modes, spatials, temporals, glmdR2):
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

        self.tca_modes = []
        for i in range(len(animals)):
            tca_mode = TCASimple(animals[i], dates[i], modes[i], reps[i], spatials[i], temporals[i], glmdR2[i])
            self.tca_modes.append(tca_mode)



    
        # self.animals = animals
        # self.dates = dates
        # self.reps = reps
        # self.modes = modes
        # self.spatials = spatials
        # self.temporals = temporals
        # self.glmdR2 = glmdR2








