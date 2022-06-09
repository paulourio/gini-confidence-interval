from itertools import product
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


def run():
    gini_list = [0.40, 0.65]  # Target gini statistic to generate
    nfolds_list = [10]  # Number of folds in your cross-validation
    p_list = [0.08, 0.15]     # P(y)
    nobs_list = [1_000, 10_000, 50_000, 100_000]  # len(y)
    trial_list = [i+1 for i in range(1000)]  # Experiment trial number
    params_comb = [gini_list, nfolds_list, p_list, nobs_list, trial_list]
    jobs = [
        delayed(evaluate)(
            gini=params[0],
            nfolds=params[1],
            p=params[2],
            nobs=params[3],
            trial=params[4],
        )
        for params in product(*params_comb)
    ]
    Parallel(n_jobs=-1)(jobs)


def evaluate(trial=0, nfolds=10, gini=0.65, nobs=10_000, p=0.08):
    result = dict(
        trial=trial,
        folds=nfolds,
        gini=gini,
        nobs=nobs,
        p=p,
    )

    h = hashlib.sha256()
    h.update(json.dumps(result).encode())
    fname = f'data/{h.hexdigest()}.json'

    if os.path.isfile(fname):
        print('skipping', fname)
        return

    y, y_hat = make_gini(gini=gini, n=nobs, p=p)
    folds, out = gini_folds(y, y_hat)

    result['gini_folds'] = folds
    result['gini_out'] = out

    with open(fname, 'wt') as out:
        out.write(json.dumps(result))
    print('Written', fname)


def gini_folds(y, y_hat, n=10):
    """Return gini scores for the first n folds.

    This function is fairly specific to a case where we simulate the
    entire data being split into as many as 80 partitions, such that
    only the first 10 is used for estimation (individually, the folds),
    the the remaining 70 for a single estimation (the "out-of-sample").

    I will give no more details on why I am using these numbers, but
    it fits my working case, and you should change to something that
    fits your work.
    """
    npartitions = 4*2
    folds = list()
    y_part = np.array_split(y, npartitions*n)
    y_hat_part = np.array_split(y_hat, npartitions*n)
    for yi, yi_hat in zip(y_part[:n], y_hat_part[:n]):
        folds.append(gini_score(yi, yi_hat))
    out = gini_score(y[sum(len(yi) for yi in y_part[:n]):],
                     y_hat[sum(len(yi) for yi in y_hat_part[:n]):])
    return folds, out


def make_gini(gini=0.64, p=0.08, n=10_000, perm=0.6, scale=0.1):
    """Return arrays with specified normalized gini index.

    This function generates two synthetic arrays so that the normalized
    gini statistic for the two samples approximates the requested gini.
    The algorithm starts with two samples with gini=1 and add noise
    to the second, continuous variable, until the gini drops down to
    the requested statistic.

    Parameters
    ----------
    gini : float
        The target normalized gini index, a value in domain (0, 1].
    p : float
        The probability of ones to exist in the first returning array.
    n : int
        The number of observations to generate.
    perm : float
        Ratio of the observations that can be permuted at once. This
        parameter controls how fast we will converge, but large values
        may result in a poor convergence (gini too below the target).
    scale : float
        Scale of the exponential noise to generate on permuted set.
        Larger scales leads to more re-arranging of data, hence faster
        convergence.

    Returns
    -------
    tuple of two ndarrays (y, y_hat).
        The first array has binary labels with p(y) approximate of
        param p.
        The second array is a continuous variable, such that the
        the normalized Gini index of (y, y_hat) is the requested gini.

    """
    y = np.random.binomial(1, p, n)
    y_hat = y.astype(float).copy()
    i = 0
    while True:
        f = gini_score(y, y_hat)
        size = min(n, int(n * perm * f))
        scale = scale + ((f - gini) * scale / 50)
        if i%100 == 0:
            print('f =', f, 'size =', size, 'scale =', scale)
        if f <= gini:
            break
        pos = np.random.choice(n, size=size, replace=False)
        noise = np.random.exponential(scale, size=size) * scale
        y_hat[pos] += noise
        i += 1
    f = gini_score(y, y_hat)
    return y, y_hat


def gini_score(expected: np.ndarray, predicted: np.ndarray) -> float:
    """Return the normalized gini index for two samples."""
    try:
        return (roc_auc_score(expected, predicted) - 0.5) / 0.5
    except ValueError as err:
        if 'Only one class' in str(err):
            return 0
        raise err


if __name__ == '__main__':
    run()
