# 2100011025, Yibo Wang, PKU CFD CLASS (2024)
import numpy as np


def ftcs_split_x(U):
    """split flux in x using FTCS"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    F = np.zeros((4, *rho.shape))

    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = u * (E + p)

    F_mid = 0.5 * (F[:, :-1, :] + F[:, 1:, :])

    return F_mid


def ftcs_split_y(U):
    """split flux in y using FTCS"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    G = np.zeros((4, *rho.shape))

    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = v * (E + p)

    G_mid = 0.5 * (G[:, :, :-1] + G[:, :, 1:])

    return G_mid


def steger_split_x(U):
    """split flux in x using Steger-Warming"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    F = np.zeros((4, *rho.shape))

    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = u * (E + p)

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

    F_mid = 0.5 * (F[:, :-1, :] + F[:, 1:, :] + F_p[:, :-1, :] - F_m[:, 1:, :])

    return F_mid


def steger_split_y(U):
    """split flux in y using Steger-Warming"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    G = np.zeros((4, *rho.shape))

    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = v * (E + p)

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

    G_mid = 0.5 * (G[:, :, :-1] + G[:, :, 1:] + G_p[:, :, :-1] - G_m[:, :, 1:])

    return G_mid


def roe_split_x(U):
    """split flux in x using Roe"""
    gamma = 1.4
    rho0 = U[0]
    u0 = U[1]/U[0]
    v0 = U[2]/U[0]
    p0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)
    E0 = p0 / (gamma - 1) + 0.5 * rho0 * (u0**2 + v0**2)

    F = np.zeros((4, *rho0.shape))

    F[0] = rho0 * u0
    F[1] = rho0 * u0**2 + p0
    F[2] = rho0 * u0 * v0
    F[3] = u0 * (E0 + p0)

    UL = U[:, :-1, :]
    UR = U[:, 1:, :]
    rhoL = rho0[:-1, :]
    rhoL = np.tile(rhoL, (4, 1, 1))
    rhoR = rho0[1:, :]
    rhoR = np.tile(rhoR, (4, 1, 1))
    alphaL = np.sqrt(rhoL) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    alphaR = 1-alphaL

    U_ave = alphaL * UL + alphaR * UR

    rho = U_ave[0]
    u = U_ave[1]/U_ave[0]
    v = U_ave[2]/U_ave[0]
    p = (U_ave[3] - 0.5 * U_ave[0] * (U_ave[1]**2 +
         U_ave[2]**2) / U_ave[0]**2) * (gamma - 1)

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

    diag_p[0, 0] = np.abs(u - c)
    diag_p[1, 1] = np.abs(u)
    diag_p[2, 2] = np.abs(u)
    diag_p[3, 3] = np.abs(u + c)

    dU = U[:, :-1, :] - U[:, 1:, :]

    X_inv_U = np.einsum('pqkr,qkr->pkr', X_inv, dU)
    diag_X_inv_U_p = np.einsum('pqkr,qkr->pkr', diag_p, X_inv_U)
    F_p = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_p)

    F_mid = 0.5 * (F[:, :-1, :] + F[:, 1:, :]) - 0.5 * F_p

    return F_mid


def roe_split_y(U):
    """split flux in x using Roe"""
    gamma = 1.4
    rho0 = U[0]
    u0 = U[1]/U[0]
    v0 = U[2]/U[0]
    p0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)
    E0 = p0 / (gamma - 1) + 0.5 * rho0 * (u0**2 + v0**2)

    G = np.zeros((4, *rho0.shape))

    G[0] = rho0 * v0
    G[1] = rho0 * u0 * v0
    G[2] = rho0 * v0**2 + p0
    G[3] = v0 * (E0 + p0)

    UL = U[:, :, :-1]
    UR = U[:, :, 1:]
    rhoL = rho0[:, :-1]
    rhoL = np.tile(rhoL, (4, 1, 1))
    rhoR = rho0[:, 1:]
    rhoR = np.tile(rhoR, (4, 1, 1))
    alphaL = np.sqrt(rhoL) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    alphaR = 1-alphaL

    U_ave = alphaL * UL + alphaR * UR

    rho = U_ave[0]
    u = U_ave[1]/U_ave[0]
    v = U_ave[2]/U_ave[0]
    p = (U_ave[3] - 0.5 * U_ave[0] * (U_ave[1]**2 +
         U_ave[2]**2) / U_ave[0]**2) * (gamma - 1)

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

    diag_p[0, 0] = np.abs(v - c)
    diag_p[1, 1] = np.abs(v)
    diag_p[2, 2] = np.abs(v)
    diag_p[3, 3] = np.abs(v + c)

    dU = U[:, :, :-1] - U[:, :, 1:]

    X_inv_U = np.einsum('pqkr,qkr->pkr', X_inv, dU)
    diag_X_inv_U_p = np.einsum('pqkr,qkr->pkr', diag_p, X_inv_U)
    G_p = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_p)

    G_mid = 0.5 * (G[:, :, :-1] + G[:, :, 1:]) - 0.5 * G_p

    return G_mid


def lax_split_x(U):
    """split flux in x using Lax-Friedrichs"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    F = np.zeros((4, *rho.shape))

    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = u * (E + p)

    c = np.sqrt(gamma * p / rho)

    uL = u[:-1, :]
    uR = u[1:, :]
    cL = c[:-1, :]
    cR = c[1:, :]

    lamL = np.abs(uL) + cL
    lamR = np.abs(uR) + cR

    lamL = np.tile(lamL, (4, 1, 1))
    lamR = np.tile(lamR, (4, 1, 1))

    F_mid = 0.5 * (F[:, :-1, :] + F[:, 1:, :]) - \
        (lamR * U[:, 1:, :] - lamL * U[:, :-1, :])

    return F_mid


def lax_split_y(U):
    """split flux in x using Lax-Friedrichs"""
    gamma = 1.4
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    p = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    G = np.zeros((4, *rho.shape))

    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = v * (E + p)

    c = np.sqrt(gamma * p / rho)

    uL = u[:, :-1]
    uR = u[:, 1:]
    cL = c[:, :-1]
    cR = c[:, 1:]

    lamL = np.abs(uL) + cL
    lamR = np.abs(uR) + cR

    lamL = np.tile(lamL, (4, 1, 1))
    lamR = np.tile(lamR, (4, 1, 1))

    G_mid = 0.5 * (G[:, :, :-1] + G[:, :, 1:]) - \
        (lamR * U[:, :, 1:] - lamL * U[:, :, :-1])

    return G_mid
