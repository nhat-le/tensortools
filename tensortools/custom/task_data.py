# For processing task-related variables
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import statsmodels.api as sm


def get_switch_prob(sessdata, window=10):
    '''
    sessdata: EnsembleData object
    window: window for calculating the switch prob
    :return: the switch probability, of shape Ntrials x 1
    '''
    Ntrials = len(sessdata.choices)
    switchprob = []
    for i in range(Ntrials):
        lowerN = max(i - int(window / 2), 0)
        upperN = min(i + int(window / 2), Ntrials)

        windowForSwitches = sessdata.choices[lowerN:upperN]
        switchCounts = np.nansum(np.diff(windowForSwitches) != 0)
        switchprob.append(switchCounts / (upperN - lowerN))

    return np.array(switchprob)


def plot_performance(sessdata):
    '''
    Function for visualizing the session performance
    :param sessdata: an EnsembleData object
    :return:
    '''
    idtrials = np.arange(sessdata.choices.shape[0])
    choices = sessdata.choices
    targets = sessdata.targets
    outcomes = sessdata.feedback

    #plot the correct responses
    plt.plot(idtrials[outcomes == 1], targets[outcomes == 1], 'bo')
    plt.plot(idtrials[outcomes == 0], targets[outcomes == 0], 'ro')





def get_value_regressor(sessdata):
    '''
    Get the value regressor from the session data
    :param sessdata: EnsembleData object
    :return: the value regressor, of shape Ntrials x 1
    '''
    # first, fit the data and find the optimal parameter set
    pfit = fit_value_regressor(sessdata)
    # print(pfit.x)

    # simulate the values according to pfit
    prob, v1, v2 = simulate_values(pfit.x, sessdata)
    return prob, v1, v2

def simulate_values(p, sessdata):
    '''
    Simulate the array of values according to the behavior data and parameters
    :param p: tuple of (learning rate, bias)
    :param sessdata: EnsembleData object
    :return: tuple of
        - Value of left actions, v1
        - Value of right actions, v2
        - Prob of choosing left, prob1
    '''
    gamma, bias = p
    # bias = 0
    choices = sessdata.choices
    outcomes = sessdata.feedback

    # Fit a Q-learning model
    v1 = 0.5
    v2 = 0.5
    v1lst = [0.5]
    v2lst = [0.5]

    for idx, (choice, outcome) in enumerate(zip(choices, outcomes)):
        if choice == 1:
            v1 = v1 + gamma * (outcome - v1)
        else:
            v2 = v2 + gamma * (outcome - v2)

        v1lst.append(v1)
        v2lst.append(v2)

    # Find the likelihood of choosing each action
    v1pos = np.maximum(np.array(v1lst), 0.001)[:-1]
    v2pos = np.maximum(np.array(v2lst), 0.001)[:-1]

    # vdiff = v1pos - v2pos
    # prob1 = 1 / (1 + np.exp(-slope * vdiff + bias))
    prob1 = v1pos / (v1pos + v2pos) + bias

    return prob1, v1pos, v2pos


def fit_value_regressor(sessdata):
    '''
    Find parameters for a value regression and fit
    the values of the two actions to the model
    :param sessdata: the EnsembleData object
    :return: tuple of (learning rate, slope, bias)
    that best explains the data
    '''
    p0 = (0.5, 0.1) #initial guess
    p = scipy.optimize.minimize(likelihood, p0, (sessdata))
    return p


def likelihood(p, sessdata):
    '''
    Likelihood of the data given parameters p (calculated based on a RL model)
    :param p: tuple of (learning rate, bias)
    :param sessdata: EnsembleData object
    :return: the likelihood (a float)
    '''
    prob1, _, _ = simulate_values(p, sessdata)
    prob1 = np.minimum(prob1, 0.999)
    prob1 = np.maximum(prob1, 0.001)
    # print(prob1)
    choices = sessdata.choices
    choices = (choices + 1) / 2
    assert(max(choices) == 1 and min(choices) == 0)
    # print(choices)
    # print(min(prob1))
    Lsingle = choices * np.log(prob1) + (1 - choices) * np.log(1 - prob1)
    return -np.sum(Lsingle)

def get_regression_Xmat(sessdata):
    '''
    Gather the design matrix for the TCA regression
    this will consist of:  outcomes, choice x outcomes switch and values
    Previous choice x outcomes
    :param sessdata:
    :return:
    '''
    # TODO: include wheel speed regressors

    # Set up the count regressor
    targets = sessdata.targets
    count_regressor = []
    for i in range(len(targets)):
        if i == 0:
            currcount = 0
        elif targets[i] == targets[i-1]:
            currcount += 1
        else:
            currcount = 0
        count_regressor.append(currcount)

    _, v1, v2 = get_value_regressor(sessdata)
    switch_regressor = get_switch_prob(sessdata, window=10)
    outcome_regressor = sessdata.feedback.astype('float') * 2 - 1
    winstay_regressor = sessdata.choices * outcome_regressor
    time_regressor = np.arange(len(sessdata.choices)).astype('float')

    assert(len(v1) == len(v2) == len(switch_regressor) == len(outcome_regressor) == len(winstay_regressor))

    prev_outcome_regressor = outcome_regressor[:-1]
    prev_winstay_regressor = winstay_regressor[:-1]

    v1_regressor = v1[1:]
    v2_regressor = v2[1:]
    time_regressor = time_regressor[1:]
    time_regressor /= max(time_regressor)
    count_regressor = count_regressor[1:]
    count_regressor = np.array(count_regressor) / len(count_regressor)
    switch_regressor = switch_regressor[1:]
    outcome_regressor = outcome_regressor[1:]
    winstay_regressor = winstay_regressor[1:]

    Xmat = np.vstack((v1_regressor, v2_regressor, switch_regressor, outcome_regressor,
                     prev_outcome_regressor, time_regressor, count_regressor))

    return Xmat.T



def conduct_tca_regression(sessdata, **kwargs):
    '''
    Performs the following operations:
    (1) extract the y-data from the trial factor
    (2) extract the design matrix, X
    (3) Performs the regression using sm.OLS
    (4) Extract the pvalues of the regression coefficients and
    performs a Benjamin-Hochberg correction to determine which hypotheses
    to reject
    :param sessdata:
    :return:
    '''
    rank = kwargs['rank']
    repid = kwargs['rep']
    alpha = kwargs['alpha']
    # Get y-data
    ydata = sessdata.get_trial_all_reps(rank)

    # Get X-data
    X = get_regression_Xmat(sessdata)
    X = sm.add_constant(X)

    pvals_all = np.zeros((ydata.shape[1], X.shape[1]))

    # Performs the regression
    for factorid in range(ydata.shape[1]):
        ysig = ydata[1:, factorid, repid]

        mdl = sm.OLS(ysig, X)
        res = mdl.fit()
        pvals_all[factorid, :] = res.pvalues

    pvals_unroll = pvals_all.ravel()

    rejected, _, _, _ = sm.stats.multipletests(pvals_unroll, alpha=alpha, method='fdr_bh')
    rejected = np.reshape(rejected, pvals_all.shape)

    return rejected












