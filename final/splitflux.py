# 2100011025, Yibo Wang, PKU CFD CLASS (2024)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def steger_split_x(U):
    """split flux in x using Steger-Warming"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

    e = gamma * p / (rho * (gamma-1))
    c = np.sqrt(gamma * p / rho)
    ek = (u**2 + v**2)/2

    X = np.zeros((4, 4, *rho.shape))

    X[0, 0] = 1
    X[0, 1] = 0
    X[0, 2] = 1
    X[0, 3] = 1

    X[1, 0] = u - c
    X[1, 1] = 0
    X[1, 2] = u
    X[1, 3] = u + c

    X[2, 0] = v
    X[2, 1] = 1
    X[2, 2] = v
    X[2, 3] = v

    X[3, 0] = e + ek - c * u
    X[3, 1] = v
    X[3, 2] = ek
    X[3, 3] = e + ek + c * u

    X_inv = np.zeros((4, 4, *rho.shape))

    X_inv[0, 0] = 0.5 * (u / c + ek / e)
    X_inv[0, 1] = -0.5 * (u / e + 1 / c)
    X_inv[0, 2] = -0.5 * v / e
    X_inv[0, 3] = 0.5 / e

    X_inv[1, 0] = -v
    X_inv[1, 1] = 0
    X_inv[1, 2] = 1
    X_inv[1, 3] = 0

    X_inv[2, 0] = 1 - ek / e
    X_inv[2, 1] = u / e
    X_inv[2, 2] = v / e
    X_inv[2, 3] = -1 / e

    X_inv[3, 0] = 0.5 * (-u / c + ek / e)
    X_inv[3, 1] = 0.5 * (-u / e + 1 / c)
    X_inv[3, 2] = -0.5 * v / e
    X_inv[3, 3] = 0.5 / e

    diag_p = np.zeros((4, 4, *rho.shape))

    diag_p[0, 0] = np.maximum(u - c, 0)
    diag_p[1, 1] = np.maximum(u, 0)
    diag_p[2, 2] = np.maximum(u, 0)
    diag_p[3, 3] = np.maximum(u + c, 0)

    X_inv_U = np.einsum('pqkr,qkr->pkr', X_inv, U)
    diag_X_inv_U_p = np.einsum('pqkr,qkr->pkr', diag_p, X_inv_U)
    F_p = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_p)

    diag_m = np.zeros((4, 4, *rho.shape))

    diag_m[0, 0] = np.minimum(u - c, 0)
    diag_m[1, 1] = np.minimum(u, 0)
    diag_m[2, 2] = np.minimum(u, 0)
    diag_m[3, 3] = np.minimum(u + c, 0)

    diag_X_inv_U_m = np.einsum('pqkr,qkr->pkr', diag_m, X_inv_U)
    F_m = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_m)

    return F_p, F_m


def steger_split_y(U):
    """split flux in y using Steger-Warming"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

    e = gamma * p / (rho * (gamma-1))
    c = np.sqrt(gamma * p / rho)
    ek = (u**2 + v**2)/2

    X = np.zeros((4, 4, *rho.shape))

    X[0, 0] = 1
    X[0, 1] = 0
    X[0, 2] = 1
    X[0, 3] = 1

    X[1, 0] = u
    X[1, 1] = 1
    X[1, 2] = u
    X[1, 3] = u

    X[2, 0] = v - c
    X[2, 1] = 0
    X[2, 2] = v
    X[2, 3] = v + c

    X[3, 0] = e + ek - c * v
    X[3, 1] = u
    X[3, 2] = ek
    X[3, 3] = e + ek + c * v

    X_inv = np.zeros((4, 4, *rho.shape))

    X_inv[0, 0] = 0.5 * (v / c + ek / e)
    X_inv[0, 1] = -0.5 * u / e
    X_inv[0, 2] = -0.5 * (v / e + 1 / c)
    X_inv[0, 3] = 0.5 / e

    X_inv[1, 0] = -u
    X_inv[1, 1] = 1
    X_inv[1, 2] = 0
    X_inv[1, 3] = 0

    X_inv[2, 0] = 1 - ek / e
    X_inv[2, 1] = u / e
    X_inv[2, 2] = v / e
    X_inv[2, 3] = -1 / e

    X_inv[3, 0] = 0.5 * (-v / c + ek / e)
    X_inv[3, 1] = -0.5 * u / e
    X_inv[3, 2] = 0.5 * (-v / e + 1 / c)
    X_inv[3, 3] = 0.5 / e

    diag_p = np.zeros((4, 4, *rho.shape))

    diag_p[0, 0] = np.maximum(v - c, 0)
    diag_p[1, 1] = np.maximum(v, 0)
    diag_p[2, 2] = np.maximum(v, 0)
    diag_p[3, 3] = np.maximum(v + c, 0)

    X_inv_U = np.einsum('pqkr,qkr->pkr', X_inv, U)
    diag_X_inv_U_p = np.einsum('pqkr,qkr->pkr', diag_p, X_inv_U)
    G_p = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_p)

    diag_m = np.zeros((4, 4, *rho.shape))

    diag_m[0, 0] = np.minimum(v - c, 0)
    diag_m[1, 1] = np.minimum(v, 0)
    diag_m[2, 2] = np.minimum(v, 0)
    diag_m[3, 3] = np.minimum(v + c, 0)

    diag_X_inv_U_m = np.einsum('pqkr,qkr->pkr', diag_m, X_inv_U)
    G_m = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_m)

    return G_p, G_m
