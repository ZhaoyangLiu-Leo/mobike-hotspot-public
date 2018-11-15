# -*- coding: utf-8 -*-

import sys, os
import numpy as np
from math import sqrt
import getopt

__thesis__ = '''A script to compute the maximum mean discrepancy (mmd) for vectors of integers as defined
by Gretton 2007. Depends on Numpy.'''

'''
Implements the mean maximum discrepancy

See Arthur Gretton...
'''


def grbf(x1, x2, sigma):
    '''
    Calculates the Gaussian radial base function kernel
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h -= 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    return np.exp(-1 * h / (2 * pow(sigma, 2)))


def kernelwidth(x1, x2):
    '''Function to estimate the sigma parameter

       The RBF kernel width sigma is computed according to a rule of thumb:

       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h -= 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    mdist = np.median([i for i in h.flat if i])

    sigma = sqrt(mdist / 2.0)
    if not sigma: sigma = 1

    return sigma


def mmd(x1, x2, sigma=None, verbose=False):
    """
    Calculates the unbiased mmd from two arrays x1 and x2

    sigma: the parameter for grbf. If None sigma is estimated

    Returns (sigma, mmd)

    """
    if x1.size != x2.size:
        raise ValueError('Arrays should have an equal amount of instances')

    # Number of instances
    m, nfeatures = x1.shape

    # Calculate sigma
    if sigma is None:
        sigma = kernelwidth(x1, x2)
    if verbose:
        print('Got kernelwidth')

    # Calculate the kernels
    Kxx = grbf(x1, x1, sigma)
    if verbose:
        print('Got Kxx')
    Kyy = grbf(x2, x2, sigma)
    if verbose:
        print('Got Kyy')
    s = Kxx + Kyy
    del Kxx, Kyy

    Kxy = grbf(x1, x2, sigma)
    if verbose:
        print('Got Kxy')
    s = s - Kxy - Kxy
    del Kxy
    if verbose:
        print('Got sum')

    # For unbiased estimator: subtract diagonal
    s -= np.diag(s.diagonal())

    value = np.sum(s) / (m * (m - 1))

    return sigma, value


def read(fname):
    '''Reads in a file as an array'''
    f = open(os.path.abspath(os.path.expanduser(fname)), 'rU')
    x = []
    try:
        for l in f:
            line = l.strip()
            if line:
                x.append([int(i) for i in line.split()])
    finally:
        f.close()

    return np.array(x)


def getmmd(fname1, fname2, sigma=None):
    '''
    Takes two files with numeric instances without a class label
    and returns (sigma, MMD).

    sigma: the parameter for grbf. If None sigma is estimated
    '''
    x1 = read(fname1)
    x2 = read(fname2)

    sigma, value = mmd(x1, x2, sigma)
    return sigma, value


def _usage():
    print('Compute the maximum mean discrepancy USAGE '
          '$ python mmd.py [-s float] testfile trainingfile '
          'files: the files should lines of equal length. The elements should be '
          'integers and should be separated by whitespace. '
          '-s float : the sigma patameter for the grbf. If not given the value is estimated.')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        # print help information and exit:
        _usage()
        sys.exit(2)

    sigma = None
    for o, a in opts:
        if o in ('-h', '--help'):
            _usage()
            sys.exit()

    if len(args) != 2:
        _usage()
        sys.exit(1)

    fname1, fname2 = args
    sigma, value = getmmd(fname1, fname2, sigma=sigma)
    print('MMD with sigma=%.4f : %f' % (sigma, value))
