
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats, signal
import pandas as pd
from numba import jit
from glob import glob
import re
import os
import mne
from mne import io
import statsmodels.formula.api as smf
million = 1000000.

## File system stuff
def mkdir(p):
    sp = re.split('/|\\\\', p)
    bp = ''
    for pp in sp:
        bp = os.path.join(bp, pp)
        if not os.path.exists(bp):
            os.mkdir(bp)
            print( '%s created.' % bp)


## Data munging
def as_array(x):
    if type(x) == pd.Series:
        return x.values
    if type(x) == pd.DataFrame:
        if len(x.columns) == 1:
            return x.iloc[:,0].values
        else:
            raise Exception('Value passed is a DataFrame with multiple columns')
    else:
        return x

def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True

def interp_times(t=np.arange(-1., 5)):
    x, y = trial_epochs.times, np.arange(eeg.shape[2])
    return np.interp(t, x, y)

def times_as_index(times, target_times):
    res = []
    if not hasattr(target_times, '__iter__'):
        target_times = [target_times]
    for t in target_times:
        r = np.argmax(times > t)
        if r == 0 or t > times.max():
            r = np.nan
        res.append(r)
    return res

def get_peak2peak(eeg):
    def p2p(x):
        return np.max(x) - np.min(x)
    peak2peak = np.array([[p2p(eeg[i, j, :]) for i in range(eeg.shape[0])]
                          for j in range(eeg.shape[1])])
    trial_peak2peak = np.max(peak2peak, 0)
    return trial_peak2peak.flatten()


def get_peak2peak_threash(trial_peak2peak, percentile_cutoff=95, plot=False):
    x = trial_peak2peak*million
    c = np.percentile(x, percentile_cutoff)
    if plot:
        plt.hist(x, bins=40);
        yl = plt.ylim()
        plt.vlines(c, *yl)
        plt.text(c+10, np.mean(yl), '$%i^{th}percentile = %.0f\ \mu V$' % (percentile_cutoff, c))
    # plt.savefig('l1_maps/peak2peak_%i.png' % subject)
    return c / million

# def exclude_dropped_metadata(behaviour, epochs):
#     dl = [v[0]  if (len(v) > 0) else 'ok' for v in epochs.drop_log]
#     dl = np.array(dl)
#     dl = dl[dl != 'IGNORED']
#     # dl = dl[dl != 'NO_DATA']
#     data = behaviour.copy()
#     data['droplog'] = dl
#     data = data[data['droplog'] == 'ok']
#     data.index = range(len(data))
#     return data


def exclude_dropped_metadata(behaviour, epochs, exclude_no_data=False, verbose=False):
    dl = [v[0]  if (len(v) > 0) else 'ok' for v in epochs.drop_log]
    dl = np.array(dl)
    dl = dl[dl != 'IGNORED']
    if exclude_no_data:
        dl = dl[dl != 'NO_DATA']
    if verbose:
        print(pd.Series(dl).value_counts())
        print('len(behaviour): %i; len(epochs): %i, len(dl): %i' % (len(behaviour), len(epochs), len(dl)))
    data = behaviour.copy()
    data['droplog'] = dl
    data = data[data['droplog'] == 'ok']
    data.index = range(len(data))
    return data


def transpose_and_transform(X, m):
    n_epochs, n_channels, n_times = X.shape
    # trial as time samples
    X = np.transpose(X, [1, 0, 2])
    X = np.reshape(X, [n_channels, n_epochs * n_times]).T
    # apply method
    X = m.transform(X)
    # put it back to n_epochs, n_dimensions
    X = np.reshape(X.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
    return X

def z(x):
    return x - np.mean(x)


##############
## Plotting ##
##############

def plot_mean_sem(X, t, label=None, color=None, alpha=.5):
    m = np.mean(X, 0)
    std = np.std(X, 0)
    n = X.shape[0]
    sem = std / np.sqrt(n)
    if color is None:
        plt.plot(t, m, label=label)
        plt.fill_between(t, m-sem, m+sem, alpha=alpha)
    else:
        plt.plot(t, m, label=label, color=color)
        plt.fill_between(t, m-sem, m+sem, alpha=alpha, color=color)

def plot_sd(X, t, label=None, color=None, alpha=.5):
    std = np.std(X, 0)
    n = X.shape[0]
    sem = std / np.sqrt(n)
    if color is None:
        plt.plot(t, std, label=label)
    else:
        plt.plot(t, std, label=label, color=color)
        plt.fill_between(t, m-sem, m+sem, alpha=alpha, color=color)


def heatmap(X, aspect='auto'):
    """
    See https://github.com/matplotlib/matplotlib/issues/4282
    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    a = np.max(np.abs(X))
    if aspect=='auto':
        aspect = float(X.shape[1]) / X.shape[0]
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols = (1, 1), axes_pad=0.1, cbar_mode='single')
    imax = grid[0].imshow(X, vmin=-a, vmax=a, cmap='seismic', aspect=aspect * 3./4)
    cax = grid.cbar_axes[0]
    cb = fig.colorbar(imax, cax=cax)
    return fig

def my_heatmap(X, lim = None, cmap='seismic'):
    x = pd.DataFrame(X, columns=trial_epochs.times)
    if lim is None:
        a = np.max(np.abs(X))
        lim0, lim1 = -a, a
    else:
        if isinstance(lim, (int, float)):
            lim0, lim1 = -lim, lim
        else:
            if isinstance(lim, (list, np.ndarray)):
                lim0, lim1 = lim
    sns.heatmap(x, cmap=cmap, center=0, vmin=lim0, vmax=lim1)
    xt = plt.xticks()
    t = np.arange(-1., 3.1)
    plt.xticks(interp_times(t), t)

def center_cmap(cmap, vmin, vmax):
    """Center given colormap (ranging from vmin to vmax) at value 0.
    Note that eventually this could also be achieved by re-normalizing a given
    colormap by subclassing matplotlib.colors.Normalize as described here:
    https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    """  # noqa: E501
    from matplotlib.colors import LinearSegmentedColormap
    vzero = abs(vmin) / (vmax - vmin)
    index_old = np.linspace(0, 1, cmap.N)
    index_new = np.hstack([np.linspace(0, vzero, cmap.N // 2, endpoint=False),
                           np.linspace(vzero, 1, cmap.N // 2)])
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}
    for old, new in zip(index_old, index_new):
        r, g, b, a = cmap(old)
        cdict["red"].append((new, r, r))
        cdict["green"].append((new, g, g))
        cdict["blue"].append((new, b, b))
        cdict["alpha"].append((new, a, a))
    return LinearSegmentedColormap("erds", cdict)

def plot_evoked_at_time(evoked, time, center=False, figsize=(4,4), hz=256., **kwargs):
    i = evoked.time_as_index(time)[0]
    d = evoked.data[:, i] * 1000000
    if center:
        d = d - d.mean()
    fig, ax_topo = plt.subplots(1, 1, figsize=figsize)
    montage = mne.channels.read_montage('standard_1020')
    info = mne.create_info(electrodes, 256., 'eeg', montage)
    im, cn =  mne.viz.plot_topomap(d, evoked.info, axes=ax_topo, show=False, **kwargs)
    divider = make_axes_locatable(ax_topo)
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, ax_colorbar=cax)


def plot_components(pca_model, epochs, n_components=10):
    plt.figure(figsize=(20,20))
    for j in range(n_components):
        p = plt.subplot(8, 4, j+1)
        mp = pca_model.components_[j]
        mne.viz.plot_topomap(mp, epochs.info, show=False, axes=p)
        plt.title('Comp %i' % j)
    plt.tight_layout()
    plt.show()

def flipy():
    plt.gca().invert_yaxis()



def plot_raw_everything(raw, chans,
                        decim=256,
                        t0=None, t1=None,
                        events=[16],
                        markers=None,
                        markersize=30,
                       annot=False):
    t = raw.times
    eeg_array = raw.get_data()
    if t0 is None:
        t0 = raw.times[0]
        s0 = 0
    else:
        s0 = raw.time_as_index(t0)[0]
    if t1 is None:
        t1 = raw.times[-1]
        s1 = -1
    else:
        s1 = raw.time_as_index(t1)[0]
    X = eeg_array[:, s0:s1:decim]
    t = t[s0:s1:decim]
    fig = plt.figure(figsize=(24, 10))
    mpl.rcParams.update({'font.size': 20})
    for i in chans:
        plt.plot(t, X[i], alpha=.5, label=raw.ch_names[i])
    plt.legend(fontsize=15)
    ## Events
    if markers is None:
        markers = 'oxv^*++pP'
    event_ch = eeg_array[raw.ch_names.index( 'STI 014')]
    for i in range(len(events)):
        ev = events[i]
        mrk = markers[i]
        is_event = event_ch  == ev
        event_times = raw.times[is_event]
        mask = (event_times > t0) & (event_times < t1)
        event_times = event_times[mask]
        y = np.min(X) * np.ones_like(event_times) + i * .1 * np.min(X)
        event_times = event_times[event_times < t1]
        plt.scatter(event_times, y, marker=mrk, s=markersize)
    if annot and (raw.annotations is not None):
        yl = plt.ylim()
        for i in range(len(raw.annotations)):
            a0 = raw.annotations.onset[i]
            dur = raw.annotations.duration[i]
            if a0 > t0 and (a0+dur) < t1:
                plt.fill_between([a0, a0+dur], color='grey', alpha=.2, *yl)
    return fig


## Plotting dataframes
def plot_raw_df(dat, x='time', y='y'):
    for i, trial in dat.groupby('trial'):
        plt.plot(trial[x], trial[y], alpha=.2, color='b', label=None)

def plot_mean_df(dat, x='time', y='y', color='r', label=None):
    m = dat.groupby(x).mean()
    return m[y].plot(color=color, label=label, linewidth=2)

def plot_mean_sem_df(dat, x='time', y='y', color='r', label=None):
    g = dat.groupby(x)
    m = g.mean()
    sd = g.std()
    n = g.count()
    sem = sd[y] / np.sqrt(n[y])
    m[y].plot(color=color, label=label, linewidth=2)
    plt.fill_between(m.index, m[y]-sem, m[y]+sem, alpha=.2, color=color)

def plot_std_df(dat, x='time', y='y', color='r', label=None):
    sd = dat.groupby(x).std()
    sd[y].plot(color=color, label=label, linewidth=2)

## Others
def plot_similarity_matrix(mat, title, lab=True, scale=None):
    m = np.array(mat)
    if not scale:
        scale = np.max(np.abs(m))
    cax = plt.imshow(m, norm=colors.Normalize(-scale, scale),
                     aspect='auto',
                     cmap=plt.get_cmap('seismic'),
                     extent=[mat.columns[0], mat.columns[-1], 0, len(mat)])
    if lab:
        plt.colorbar(cax)
        plt.yticks([])
        plt.xlabel('Time')
        plt.ylabel('Trial')
        plt.title(title)
    return cax

def plot_spectrum(spec, t, f,
                  log=False, cmap='inferno', center=False, fmax=30,
                  interp='kaiser',
                  xl='Time', yl='Frequency (Hz)', legend=True):
    n_times = len(t)
    if spec.shape[1] != n_times:
        spec = np.transpose(spec)
    # plt.figure(figsize=(figw, figh))
    # spec = spec[:fmax,:]
    if center:
        a = np.max(np.abs(spec))
        norm = colors.Normalize(vmin=-a, vmax=a)
    else:
        norm = None
    if log:
        spec = 10*np.log10(spec)
    dt = np.diff(t)[0]
    extent = [t[0]-.5*dt, t[-1]+.5*dt, fmax, 0]
    cax = plt.imshow(spec,
                     norm=norm,
                     aspect='auto',
                     interpolation=interp,
                     cmap=plt.get_cmap(cmap),
                     extent=extent)
    plt.gca().invert_yaxis()
        #     plt.ylim(0, fmax)
    if legend:
        plt.colorbar(cax)
        plt.xlabel(xl)
        plt.ylabel(yl)

def plot_spectrum_tthesh(spec, t, f, tvals,
                         threshold=2,
                         fmax=30, interp='kaiser',
                         xl='Time', yl='Frequency (Hz)', legend=True):
    spec = spec.copy()
    n_times = len(t)
    if spec.shape[1] != n_times:
        spec = np.transpose(spec)
    if tvals.shape[1] != n_times:
        tvals = np.transpose(tvals)
    a = np.max(np.abs(spec))
    norm = colors.Normalize(vmin=-a, vmax=a)
    dt = np.diff(t)[0]
    extent = [t[0]-.5*dt, t[-1]+.5*dt, fmax, 0]
    # spec_not_sig = spec.copy()
    # spec_not_sig[np.abs(tvals)>ppthreshold] = -a
    spec_sig = spec.copy()
    spec_sig[np.abs(tvals) < threshold] = 0
    cax1 = plt.imshow(spec_sig,
                      norm=norm,
                      aspect='auto',
                      interpolation=interp,
                      cmap=plt.get_cmap('seismic'),
                      extent=extent)
    # cax0 = plt.imshow(spec_not_sig,
    #                   cmap=cm_seismic_bw_alpha,
    #                   norm=norm,
    #                   aspect='auto',
    #                   interpolation=interp,
    #                   extent=extent)
    plt.gca().invert_yaxis()
    plt.colorbar(cax1)
    plt.xlabel(xl)
    plt.ylabel(yl)

# def plot_mean_spectrum(data, dv, fmax=30,
#                        xl='Time', yl='Frequency', legend=True):
#     spect = get_mean_spectrum(data, dv)
#     spect_wide = pivot_spectrum(spect)
#     s = np.transpose(spect_wide.values)
#     plot_spectrum(s,
#                   t=spect_wide.index.values,
#                   f=spect_wide.columns.values,
#                   fmax=fmax, xl=xl, yl=yl, legend=legend)
#     return mean_spectrum, t, f

def plot_single_spectrum(x, hz, color='b', label='', log=False):
    f, t, Sxx = signal.spectrogram(x, fs=hz, nperseg=len(x))
    if log:
        plt.plot(f, 10*np.log10(Sxx), color=color, label=label)
        plt.xlim(0, 60)
    else:
        plt.plot(f, Sxx, color=color, label=label)
        plt.xlim(0, 10)

def plot_filter_response(cutoff, fs, order=5):
    # Plot the frequency response.
    b, a = butter_lowpass(cutoff, fs, order=order)
    w, h = signal.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 2*cutoff)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

def plot_image_vs_var(epochs, var, chan, overlay=False, negative=False, **kwargs):
    fig = plt.figure(figsize=(12, 6))
    df = epochs.metadata
    if df.index.max() != len(epochs):
        print('Warning - Metadata index doesn\'t match length of epochs')
        df.index = range(len(df))
    order = df.sort_values(var).index.values
    picks = epochs.ch_names.index(chan)
    if overlay:
        val = df[var].values
        if negative:
            val *= -1
    else:
        val = None
    epochs.plot_image(picks=picks, fig=fig,
                      order=order, overlay_times=val,
                      show=False, evoked=False,
                      **kwargs)


def gen_my_colormaps():
    cm_seismic_alpha = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    n = cm_seismic_alpha.shape[0]
    cm_seismic_alpha[(n/2)-5:(n/2+5),3] = 0.
    cm_seismic_alpha = colors.ListedColormap(cm_seismic_alpha)

    cm_seismic_bw = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    r, g, b = [cm_seismic_bw[:,i] for i in range(3)]
    m =  0.21*r + 0.72*g + 0.07*b
    for i in range(3):
        cm_seismic_bw[:,i] = m
    cm_seismic_bw = colors.ListedColormap(cm_seismic_bw)

    cm_seismic_bw_alpha = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    r, g, b = [cm_seismic_bw_alpha[:,i] for i in range(3)]
    m =  0.21*r + 0.72*g + 0.07*b
    # m = np.apply_along_axis(np.mean, 1, cm_seismic_bw_alpha[:,0:2])
    for i in range(3):
        cm_seismic_bw_alpha[:,i] = m + (1-m)*.5
        n = cm_seismic_bw_alpha.shape[0]
        cm_seismic_bw_alpha[0:2,3] = 0.
        cm_seismic_bw_alpha[n-2:n,3] = 0.0
    cm_seismic_bw_alpha = colors.ListedColormap(cm_seismic_bw_alpha)
    return cm_seismic_alpha, cm_seismic_bw, cm_seismic_bw_alpha

# cm_seismic_alpha, cm_seismic_bw, cm_seismic_bw_alpha = gen_my_colormaps()



########################
## Frequency analyses ##
########################

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    data = as_array(data)
    nans_at_start = np.argmin(np.isnan(data))
    nans_at_end = np.argmin(np.isnan(data[::-1])) # Prob. not needed
    if nans_at_start:
        data = data[nans_at_start:]
    if nans_at_end:
        data = data[:nans_at_end]
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    if nans_at_start:
        y = np.concatenate([np.repeat(np.nan, nans_at_start), y])
    if nans_at_end:
        y = np.concatenate([y, np.repeat(np.nan, nans_at_end)])
    return y

def manual_spectrogram(x, hz=256, nsegs=6, max_freq=30):
    x = as_array(x)
    window_size = len(x)/nsegs
    shortenedx = x[:window_size*nsegs]
    wdw = shortenedx.reshape(nsegs, window_size)
    o = [calculate_fft(wdw[i,:], hz=hz, max_freq=max_freq)
         for i in range(wdw.shape[0])]
    fs, powers = zip(*o)
    # print( all_equal(fs))
    f = fs[0]
    n = len(f)
    result = pd.DataFrame()
    result['window'] = np.repeat(range(nsegs), n)
    result['freq']   =  np.tile(f, nsegs)
    result['power']  = np.concatenate(powers)
    return result



def epochs_hilbert(epochs, frequencies, crop, baseline):
    """
    Hilbert transformation on epochs data
    From https://github.com/mne-tools/mne-python/issues/4869
    Parameters
    ----------
    frequencies : tuple (Names, freq_min, freq_max)
        Name and values of the frequency band to explore.
    crop : tuple (tmin, tmax)
        New crop to apply after the Hilbert transform.
    baseline: int
        Baseline to apply after the Hilbert transform.
    Returns
    -------
    epochs : instance of Epochs
        The epochs object with transformed data.
    """
    from scipy.signal import hilbert
    band, fmin, fmax = frequencies
    epochs.filter(fmin, fmax, n_jobs=1,
                  l_trans_bandwidth=1,
                  h_trans_bandwidth=1,
                  fir_design='firwin')

    # Hilbert transformation
    epochs._data = hilbert(epochs._data)
    # Crop the new epochs
    epochs.crop(tmin = crop[0], tmax = crop[1])
    # Aplly baseline
    epochs.apply_baseline(baseline=baseline)
    # remove evoked response and get analytic signal (envelope)
    epochs.subtract_evoked()
    epochs = mne.EpochsArray(data=np.abs(epochs.get_data()),
                             info=epochs.info,
                             tmin=epochs.tmin)
    return epochs


def plot_joint(erp, times, title='', width=12, height=8, invert=True, save=None):
    fig = erp.plot_joint(times=times,
                         show=False,
                         ts_args=dict(time_unit='s'),
                         topomap_args=dict(res=128, contours=4, time_unit='s'),
                         title=title)
    fig.set_figwidth(width)
    fig.set_figheight(height)
    axes = fig.get_axes()
    ax0 = axes[0]
    if invert:
        ax0.invert_yaxis()
    ch = ax0.get_children()
    for c in ch:
        if type(c) == plt.Annotation:
            c.remove()
        if type(c) == plt.Line2D:
            c.set_linewidth(2.5)
            c.set_alpha(.75)
    leg_ax = axes[-2]
    leg_ax.get_children()[0].set_sizes([50.])
    leg_ax.set_aspect('equal')
    if save is not None:
        fig.savefig(save)
    fig.show()



# def plot_joint(erp, times, title='', width=12, height=8, invert=True):
#     fig = erp.plot_joint(times=times,
#                          show=False,
#                          ts_args=dict(time_unit='ms'),
#                          topomap_args=dict(res=128, contours=4, time_unit='ms'),
#                          title=title)
#     fig.set_figwidth(width)
#     fig.set_figheight(height)
#     ax0 = fig.get_axes()[0]
#     if invert:
#         ax0.invert_yaxis()
#     ch = ax0.get_children()
#     for c in ch:
#         if type(c) == plt.Annotation:
#             c.remove()
#         if type(c) == plt.Line2D:
#             c.set_linewidth(2.5)
#     fig.show()




def varimax(components, method='varimax', eps=1e-6, itermax=100):
    """Return rotated components."""
    if method == 'varimax':
        gamma = 1.0
    elif (method == 'quartimax'):
        gamma = 0.0
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0
    for _ in range(itermax):
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

def cor(x, y):
    m = np.isnan(x) | np.isnan(y)
    r = np.corrcoef(x[m==False], y[m==False])
    return r[0,1]

def find_outlier_trials(epochs, thresh=120. / million):
    X = epochs.get_data()[:, :32]
    aX = np.abs(X).max(2).max(1)
    return aX > thresh

def surface_laplacian(epochs, leg_order=50, m=4, smoothing=1e-5, montage='standard_1020'):
    """
    https://github.com/alberto-ara/Surface-Laplacian/blob/master/surface%20laplacian.py
    This function attempts to compute the surface laplacian transform to an mne Epochs object. The
    algorithm follows the formulations of Perrin et al. (1989) and it consists for the most part in a
    nearly-literal translation of Mike X Cohen's 'Analyzing neural time series data' corresponding MATLAB
    code (2014).
    INPUTS are:
        - epochs: mne Epochs object
        - leg_order: maximum order of the Legendre polynomial
        - m: smothness parameter for G and H
        - smoothing: smothness parameter for the diagonal of G
        - montage: montage to reconstruct the transformed Epochs object (same as in raw data import)
    OUTPUTS are:
        - before: unaffected reconstruction of the original Epochs object
        - after: surface laplacian transform of the original Epochs object
    References:
        - Perrin, F., Pernier, J., Bertrand, O. & Echallier, J.F. (1989). Spherical splines for scalp
          potential and current density mapping. Electroencephalography and clinical Neurophysiology, 72,
          184-187.
        - Cohen, M.X. (2014). Surface Laplacian In Analyzing neural time series data: theory and practice
          (pp. 275-290). London, England: The MIT Press.
    """
    # import libraries
    import numpy as np
    from scipy import special
    import math
    import mne

    epochs = epochs.copy().pick_types(eeg=True)
    # get electrodes positions
    locs = epochs._get_channel_positions()

    x = locs[:,0]
    y = locs[:,1]
    z = locs[:,2]

    # arrange data
    # n_eeg = np.argmin([ch['kind'] == 2 for ch in epochs.info['chs']])
    # data = epochs.get_data()[:, :n_eeg] # data
    data = epochs.get_data() # data
    data = np.rollaxis(data, 0, 3)
    orig_data_size = np.squeeze(data.shape)

    numelectrodes = len(x)

    # normalize cartesian coordenates to sphere unit
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    junk1, junk2, spherical_radii = cart2sph(x,y,z)
    maxrad = np.max(spherical_radii)
    x = x/maxrad
    y = y/maxrad
    z = z/maxrad

    # compute cousine distance between all pairs of electrodes
    cosdist = np.zeros((numelectrodes, numelectrodes))
    for i in range(numelectrodes):
        for j in range(i+1,numelectrodes):
            cosdist[i,j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)/2)

    cosdist = cosdist + cosdist.T + np.identity(numelectrodes)

    # get legendre polynomials
    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))
    for ni in range(leg_order):
        for i in range(numelectrodes):
            for j in range(i+1, numelectrodes):
                #temp = special.lpn(8,cosdist[0,1])[0][8]
                legpoly[ni,i,j] = special.lpn(ni+1,cosdist[i,j])[0][ni+1]

    legpoly = legpoly + np.transpose(legpoly,(0,2,1))

    for i in range(leg_order):
        legpoly[i,:,:] = legpoly[i,:,:] + np.identity(numelectrodes)

    # compute G and H matrixes
    twoN1 = np.multiply(2, range(1, leg_order+1))+1
    gdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m, dtype=float)
    hdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m-1, dtype=float)

    G = np.zeros((numelectrodes, numelectrodes))
    H = np.zeros((numelectrodes, numelectrodes))

    for i in range(numelectrodes):
        for j in range(i, numelectrodes):
            g = 0
            h = 0
            for ni in range(leg_order):
                g = g + (twoN1[ni] * legpoly[ni,i,j]) / gdenom[ni]
                h = h - (twoN1[ni] * legpoly[ni,i,j]) / hdenom[ni]
            G[i,j] = g / (4*math.pi)
            H[i,j] = -h / (4*math.pi)
    G = G + G.T
    H = H + H.T
    G = G - np.identity(numelectrodes) * G[1,1] / 2
    H = H - np.identity(numelectrodes) * H[1,1] / 2
    if np.any(orig_data_size==1):
        data = data[:]
    else:
        data = np.reshape(data, (orig_data_size[0], np.prod(orig_data_size[1:3])))

    # compute C matrix
    Gs = G + np.identity(numelectrodes) * smoothing
    GsinvS = np.sum(np.linalg.inv(Gs), 0)
    dataGs = np.dot(data.T, np.linalg.inv(Gs))
    C = dataGs - np.dot(np.atleast_2d(np.sum(dataGs, 1)/np.sum(GsinvS)).T, np.atleast_2d(GsinvS))

    # apply transform
    # original = np.reshape(data, orig_data_size)
    surf_lap = np.reshape(np.transpose(np.dot(C,np.transpose(H))), orig_data_size)

    # re-arrange data into mne's Epochs object
    events = epochs.events
    event_id = epochs.event_id
    ch_names = epochs.ch_names
    sfreq = epochs.info['sfreq']
    tmin = epochs.tmin
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg', montage=montage)

    # original = np.rollaxis(original, 2, 0)
    surf_lap = np.rollaxis(surf_lap, 2, 0)

    # before = mne.EpochsArray(data=original, info=info, events=events, event_id=event_id, tmin=tmin, on_missing='ignore')
    after = mne.EpochsArray(data=surf_lap, info=info, events=events, event_id=event_id, tmin=tmin, on_missing='ignore')
    if epochs.metadata is not None:
        after.metadata = epochs.metadata.copy()
    return after
    # return before, after