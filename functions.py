from __future__ import print_function
import os
# import sys
from glob import glob
import mne
# from mne import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
import pandas as pd
# from mne.decoding import UnsupervisedSpatialFilter
# from sklearn.decomposition import PCA, FastICA, FactorAnalysis
# from sklearn.externals import joblib
import statsmodels.formula.api as smf
from scipy import signal

million = 1000000.

import eegf


## Some preprocessing functions
def load_subject_csv(subject):
    dat_cols = [u'participant', u'cb', u'condition',
                u'block_nr', u'block_half', u'trial_nr', u'v_win', u'p_win', u'action',
                u'response', u'rt', u'outcome', u'score_delta', u'score', u'visible']
    fn = glob('data/csv/%i*.csv' % subject)
    if len(fn) != 1:
        print('Something wrong reading files.', fn)
        raise Exception('Something wrong reading files.')
    fn = fn[0]
    data = pd.read_csv(fn)[dat_cols]
    visible = data[data['visible'] == 1].copy()
    m = smf.logit('response ~ v_win * p_win', data=visible).fit()
    pred = m.predict(data)
    data['predicted_response'] = np.where(data['visible'] == 1, pred, np.nan)
    data['predicted_action'] = np.where(data['condition'] == 0,
                                        data['predicted_response'], 1 - data['predicted_response'])
    data['difficulty'] = .5 - np.abs(data['predicted_response'] - .5)
    data['difficult'] = data['difficulty'] > data['difficulty'].median()
    return data


def get_gfp_peaks(erp, lp=4):
    gfp = erp.data.var(0)
    # gfp = np.concatenate([[np.nan], np.diff(gfp)])
    gfp_smooth = eegf.butter_lowpass_filter(gfp, lp, 250)
    peaks = signal.find_peaks(gfp_smooth)[0]
    times = erp.times[peaks]
    return times


def varimax(components, method='varimax', eps=1e-6, itermax=100):
    """Return rotated components."""
    if method == 'varimax':
        gamma = 1.0
    elif method == 'quartimax':
        gamma = 0.0

    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _i in range(itermax):
        comp_rot = np.dot(components, rotation_matrix)
        tmp = np.diag((comp_rot ** 2).sum(axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - np.dot(comp_rot, tmp)))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and (var_new < var * (1 + eps)):
            break
        var = var_new
    return np.dot(components, rotation_matrix).T


def oblimin_objective(L=None, A=None, T=None, gamma=0,
                      rotation_method='orthogonal',
                      return_gradient=True):
    if L is None:
        assert (A is not None and T is not None)
        L = rotateA(A, T, rotation_method=rotation_method)
    p, k = L.shape
    L2 = L ** 2
    N = np.ones((k, k)) - np.eye(k)
    if np.isclose(gamma, 0):
        X = L2.dot(N)
    else:
        C = np.ones((p, p)) / p
        X = (np.eye(p) - gamma * C).dot(L2).dot(N)
    phi = np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = L * X
        return phi, Gphi
    else:
        return phi


def find_csv(subject, datapath):
    csvs = glob(os.path.join(datapath, 'csv/*'))
    csvs = sorted(csvs)
    for p in csvs:
        if p.find('csv/%i' % subject) > -1:
            return p


def find_all_csv(subject, datapath):
    csvs = glob(os.path.join(datapath, 'csv/*'))
    csvs = sorted(csvs)
    matches = []
    for p in csvs:
        if p.find('%i_' % subject) > -1:
            matches.append(p)
    return matches


def merge_behav_data(subject, datapath='data'):
    files = find_all_csv(subject, datapath)
    if len(files) == 0:
        raise Exception("csv files not in directory for subject %i" % subject)
    dfs = []
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        d['section'] = i
        d['fname'] = f
        dfs.append(d)
    df = pd.concat(dfs)
    return df


def get_order(participants, participant):
    order = participants[participants['Participant ID'] == participant][
        'Condition (1 = first 1, then 3; 2 = first 3, then 1)'].iloc[0]
    return order


def augment_metadata(data, order):
    """
    Code condition details for the Flowers task.
    Note that this differs from the original function.
    The first argument is the metadata itself (epochs.metadata)
    The second argument is the order (1 or 2)
    Usage:
    > order = get_order(participants, participant)
    > epochs.metadata = augment_metadata(epochs.metadata, order)
    """
    if not order in [1, 2]:
        raise Exception('Second argument to augment_metadata should be order (1 or 2).')
    is_new_block = np.diff(data['block_elapsed']) < 0
    is_new_block = np.concatenate([is_new_block, [False]])
    block = np.cumsum(is_new_block)
    phase = np.where(block < 2, 0, 1)
    if order == 1:
        condition = np.where(phase == 0, 'Predictable', 'Unpredictable')
    elif order == 2:
        condition = np.where(phase == 0, 'Unpredictable', 'Predictable')
    data['block'] = block
    data['phase'] = phase
    data['order'] = order
    data['condition'] = condition
    data['response'] = np.where(data['reward'] == 1, 'Skip', 'Harvest')
    data['rt'] = data['t_action'] - data['t_start'] - 1  # One second waiting period.
    data['rt'] = np.where(data['rt'] < 0, np.nan, data['rt'])
    return data


def get_epochs(raw, event_id, tmin=-.1, tmax=2., baseline=(-.1, 0.), rej=200., **kwargs):
    events = mne.find_events(raw, shortest_event=1)
    reject = {'eeg': rej / million}
    # _d = dict(events=events, event_id=event_id,
    #           tmin=tmin, tmax=tmax, baseline=baseline,
    #           reject=reject)
    return mne.Epochs(raw, events=events, event_id=event_id,
                      tmin=tmin, tmax=tmax, baseline=baseline,
                      reject=reject, **kwargs).load_data()


# reload(functions)


def add_annotations(raw, starts, ends):
    lengths = np.array(ends) - np.array(starts)
    orig_time = raw.info['meas_date']
    if not np.isscalar(orig_time):
        orig_time = orig_time[0]
    return mne.Annotations(starts, lengths, ['bad_segment'] * len(starts), orig_time=orig_time)


def concat_annotations(list_of_annots, raw):
    starts = []
    ends = []
    for A in list_of_annots:
        if A is not None:
            starts += list(A.onset)
            ends += list(A.onset + A.duration)
    print(starts)
    print(ends)
    return add_annotations(raw, starts, ends)


## Imports from Roulette project - UNTESTED!

def func_by_subject(epochs, func, exclude=[]):
    """func must take an axis argument."""
    data = epochs.metadata
    subjects = data['participant'].unique()
    subjects = [s for s in subjects if s not in exclude]
    res = []
    for s in subjects:
        X = epochs['participant == %i' % s].get_data()
        x = func(X, axis=0)
        res.append(x)
    return np.array(res)


def mean_by_subject(epochs, exclude=[]):
    return func_by_subject(epochs, np.mean, exclude=exclude)


def std_by_subject(epochs, exclude=[]):
    return func_by_subject(epochs, np.std, exclude=exclude)


def array_func_by_subject(X, func, subject_nr, exclude=[]):
    """
    func must take an axis argument.
    X is (times x trials)
    """
    subjects = subject_nr.unique()
    subjects = [s for s in subjects if s not in exclude]
    res = []
    for s in subjects:
        subjX = X[subject_nr == s]
        fX = func(subjX, axis=0)
        res.append(fX)
    return np.array(res)


def array_mean_by_subject(X, subject_nr, exclude=[]):
    return array_func_by_subject(X, np.mean, subject_nr, exclude=exclude)


def array_std_by_subject(X, subject_nr, exclude=[]):
    return array_func_by_subject(X, np.std, subject_nr, exclude=exclude)


def raw_by_subject(epochs, ch=9, yl=200, show_mean=True, show_raw=True):
    fig, axes = plt.subplots(figsize=(20, 20), ncols=4, nrows=5)
    axes = iter(np.concatenate(axes))
    times = epochs.times
    subjects = epochs.metadata['participant'].unique()
    for subject in subjects:
        ax = axes.next()
        X = epochs['participant == %i' % subject].get_data() * million
        rawplot(X, times=times, ch=ch, yl=yl, show_raw=show_raw, show_mean=show_mean, ax=ax)
        ax.set_title(subject)
    plt.tight_layout()
    plt.show()


def rawplot(X, times, ch=9, yl=200, show_raw=True, show_mean=True, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    if show_raw:
        for i in range(X.shape[0]):
            ax.plot(times, X[i, ch], alpha=.2, color='b')
    if show_mean:
        ax.plot(times, X[:, ch].mean(0), color='r')
    ax.set_ylim(-yl, yl)
    ax.invert_yaxis()


## PCA stuff
def topomap(w, info, axes=None, show=False, **kwargs):
    a = np.abs(w).max()
    return mne.viz.plot_topomap(w, info, axes=axes, show=show, **kwargs)


def rotate_eeg(X, L):
    return np.stack([X[i].T.dot(L) for i in range(X.shape[0])], axis=0).swapaxes(2, 1)


def correct_rotation_signs(rotmat, epochs, t0, t1):
    # t0, t1 = epochs.time_as_index([t0, t1])
    X = epochs.copy().crop(t0, t1).get_data()[:, :32]
    X_pca = rotate_eeg(X, rotmat)
    m_pca = X_pca.mean(0)
    comp_signs = np.sign(m_pca[:, -1] - m_pca[:, 0])
    return rotmat * comp_signs


def plot_weight_topomaps(weights, info, label='C'):
    """weights: n_comp x 32"""
    n = weights.shape[0]
    fig = plt.figure(figsize=(n * 1.5, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        topomap(weights[i], info, axes=ax, show=False)
        plt.title('%s%i' % (label, i + 1))
    return fig
