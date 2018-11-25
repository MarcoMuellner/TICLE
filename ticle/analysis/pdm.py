import numpy as np
import logging
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

NCPUS = cpu_count()

def get_frequency_grid(times,
                       samplesperpeak=5,
                       nyquistfactor=5,
                       minfreq=None,
                       maxfreq=None,
                       returnf0dfnf=False):

    baseline = times.max() - times.min()
    nsamples = times.size

    df = 1. / baseline / samplesperpeak

    if minfreq is not None:
        f0 = minfreq
    else:
        f0 = 0.5 * df

    if maxfreq is not None:
        Nf = int(np.ceil((maxfreq - f0) / df))
    else:
        Nf = int(0.5 * samplesperpeak * nyquistfactor * nsamples)


    if returnf0dfnf:
        return f0, df, Nf, f0 + df * np.arange(Nf)
    else:
        return f0 + df * np.arange(Nf)

def phase_magseries(times, mags, period, epoch, wrap=True, sort=True):
    '''
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times)

    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
    )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))

    return outdict


def stellingwerf_pdm_theta(times, mags, errs, frequency,
                           binsize=0.05, minbin=9):
    '''
    This calculates the Stellingwerf PDM theta value at a test frequency.

    '''

    period = 1.0/frequency
    fold_time = times[0]

    phased = phase_magseries(times,
                             mags,
                             period,
                             fold_time,
                             wrap=False,
                             sort=True)

    phases = phased['phase']
    pmags = phased['mags']
    bins = np.arange(0.0, 1.0, binsize)
    nbins = bins.size

    binnedphaseinds = np.digitize(phases, bins)

    binvariances = []
    binndets = []
    goodbins = 0

    for x in np.unique(binnedphaseinds):

        thisbin_inds = binnedphaseinds == x
        thisbin_phases = phases[thisbin_inds]
        thisbin_mags = pmags[thisbin_inds]

        if thisbin_mags.size > minbin:
            thisbin_variance = np.var(thisbin_mags,ddof=1)
            binvariances.append(thisbin_variance)
            binndets.append(thisbin_mags.size)
            goodbins = goodbins + 1

    # now calculate theta
    binvariances = np.array(binvariances)
    binndets = np.array(binndets)

    theta_top = np.sum(binvariances*(binndets - 1)) / (np.sum(binndets) -
                                                      goodbins)
    theta_bot = np.var(pmags,ddof=1)
    theta = theta_top/theta_bot

    return theta

def stellingwerf_pdm_worker(task):
    '''
    This is a parallel worker for the function below.

    task[0] = times
    task[1] = mags
    task[2] = errs
    task[3] = frequency
    task[4] = binsize
    task[5] = minbin

    '''

    times, mags, errs, frequency, binsize, minbin = task

    try:

        theta = stellingwerf_pdm_theta(times, mags, errs, frequency,
                                       binsize=binsize, minbin=minbin)

        return theta

    except Exception as e:

        return np.nan

def stellingwerf_pdm(stimes,
                     smags,
                     serrs,
                     autofreq=True,
                     startp=None,
                     endp=None,
                     normalize=False,
                     stepsize=1.0e-4,
                     phasebinsize=0.05,
                     mindetperbin=9,
                     nbestpeaks=5,
                     periodepsilon=0.1,
                     sigclip=10.0,
                     nworkers=None,
                     verbose=True):
    '''This runs a parallel Stellingwerf PDM period search.

    '''
    # get the frequencies to use
    if startp:
        endf = 1.0/startp
    else:
        # default start period is 0.1 day
        endf = 1.0/0.1

    if endp:
        startf = 1.0/endp
    else:
        # default end period is length of time series
        startf = 1.0/(stimes.max() - stimes.min())

    # if we're not using autofreq, then use the provided frequencies
    if not autofreq:
        frequencies = np.arange(startf, endf, stepsize)
        if verbose:
            logger.info(
                'using %s frequency points, start P = %.3f, end P = %.3f' %
                (frequencies.size, 1.0/endf, 1.0/startf)
            )
    else:
        # this gets an automatic grid of frequencies to use
        frequencies = get_frequency_grid(stimes,
                                         minfreq=startf,
                                         maxfreq=endf)
        if verbose:
            logger.info(
                'using autofreq with %s frequency points, '
                'start P = %.3f, end P = %.3f' %
                (frequencies.size,
                 1.0/frequencies.max(),
                 1.0/frequencies.min())
            )

    # map to parallel workers
    if (not nworkers) or (nworkers > NCPUS):
        nworkers = NCPUS
        if verbose:
            logger.info('using %s workers...' % nworkers)

    pool = Pool(nworkers)

    # renormalize the working mags to zero and scale them so that the
    # variance = 1 for use with our LSP functions
    if normalize:
        nmags = (smags - np.median(smags))/np.std(smags)
    else:
        nmags = smags

    tasks = [(stimes, nmags, serrs, x, phasebinsize, mindetperbin)
             for x in frequencies]

    lsp = pool.map(stellingwerf_pdm_worker, tasks)

    pool.close()
    pool.join()
    del pool

    lsp = np.array(lsp)
    periods = 1.0/frequencies

    # find the nbestpeaks for the periodogram: 1. sort the lsp array by
    # lowest value first 2. go down the values until we find five values
    # that are separated by at least periodepsilon in period

    # make sure to filter out the non-finite values of lsp
    finitepeakind = np.isfinite(lsp)
    finlsp = lsp[finitepeakind]
    finperiods = periods[finitepeakind]

    # finlsp might not have any finite values if the period finding
    # failed. if so, argmin will return a ValueError.
    try:

        bestperiodind = np.argmin(finlsp)

    except ValueError:

        logger.error('no finite periodogram values for '
                 'this mag series, skipping...')
        return {'bestperiod':np.nan,
                'bestlspval':np.nan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'pdm',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}

    sortedlspind = np.argsort(finlsp)
    sortedlspperiods = finperiods[sortedlspind]
    sortedlspvals = finlsp[sortedlspind]

    prevbestlspval = sortedlspvals[0]

    # now get the nbestpeaks
    nbestperiods, nbestlspvals, peakcount = (
        [finperiods[bestperiodind]],
        [finlsp[bestperiodind]],
        1
    )
    prevperiod = sortedlspperiods[0]

    # find the best nbestpeaks in the lsp and their periods
    for period, lspval in zip(sortedlspperiods, sortedlspvals):

        if peakcount == nbestpeaks:
            break
        perioddiff = abs(period - prevperiod)
        bestperiodsdiff = [abs(period - x) for x in nbestperiods]

        # print('prevperiod = %s, thisperiod = %s, '
        #       'perioddiff = %s, peakcount = %s' %
        #       (prevperiod, period, perioddiff, peakcount))

        # this ensures that this period is different from the last
        # period and from all the other existing best periods by
        # periodepsilon to make sure we jump to an entire different peak
        # in the periodogram
        if (perioddiff > (periodepsilon*prevperiod) and
            all(x > (periodepsilon*prevperiod) for x in bestperiodsdiff)):
            nbestperiods.append(period)
            nbestlspvals.append(lspval)
            peakcount = peakcount + 1

        prevperiod = period


    return {'bestperiod':finperiods[bestperiodind],
            'bestlspval':finlsp[bestperiodind],
            'nbestpeaks':nbestpeaks,
            'nbestlspvals':nbestlspvals,
            'nbestperiods':nbestperiods,
            'lspvals':lsp,
            'periods':periods,
            'method':'pdm',
            'kwargs':{'startp':startp,
                      'endp':endp,
                      'stepsize':stepsize,
                      'normalize':normalize,
                      'phasebinsize':phasebinsize,
                      'mindetperbin':mindetperbin,
                      'autofreq':autofreq,
                      'periodepsilon':periodepsilon,
                      'nbestpeaks':nbestpeaks,
                      'sigclip':sigclip}}