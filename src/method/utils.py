#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def _scaleSimMat(A):
    """Scale rows of similarity matrix.
    :param A: input similarity matrix
    :return: row-normalized matrix
    """
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A


def PPMI_matrix(M):
    """Compute Positive Pointwise Mutual Information Matrix.
    :param M: input similarity matrix
    :return: PPMI matrix of input matrix M
    """
    # normalize the matrix
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)
    # compute PPMI matrix
    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI
