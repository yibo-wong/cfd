# 2100011025, Yibo Wang, PKU CFD CLASS (2024)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def apply_boundary_condition(U_new, dx, dy, nx1, alpha):
    """update boundary condition - both Dirichlet and Newmann"""
    gamma = 1.4
    R = 287.14
    rho = U_new[0]
    u = U_new[1] / rho
    v = U_new[2] / rho
    p = (gamma - 1) * (U_new[3] - 0.5 * rho * (u**2 + v**2))
    T = p / (R * rho)

    beta = np.tan(np.deg2rad(alpha)) / (np.tan(np.deg2rad(alpha))**2 + 1)

    slope = 1 - 3 * np.tan(np.deg2rad(alpha)) / 2.4

    u[3:, -1] = u[3:, -2] = u[3:, -3] = u[3:, -4]
    u[-1, 3:] = u[-2, 3:] = u[-3, 3:] = u[-4, 3:]

    v[3:, -1] = v[3:, -2] = v[3:, -3] = v[3:, -4]
    v[-1, 3:] = v[-2, 3:] = v[-3, 3:] = v[-4, 3:]

    p[3:, -1] = p[3:, -2] = p[3:, -3] = p[3:, -4]
    # p[3:nx1, 0] = p[3:nx1, 1] = p[3:nx1, 2] = p[3:nx1, 3]  # wrong
    p[-1, :] = p[-2, :] = p[-3, :] = p[-4, :]

    T[3:, -1] = T[3:, -2] = T[3:, -3] = T[3:, -4]
    # T[3:nx1, 0] = T[3:nx1, 1] = T[3:nx1, 2] = T[3:nx1, 3]  # wrong
    T[-1, :] = T[-2, :] = T[-3, :] = T[-4, :]

    dpdx = np.zeros_like(p[:, 0])
    dpdx[1: -1] = (p[2:, 3] - p[:-2, 3]) / (2*dx)
    dpdx[0] = (p[1, 3] - p[0, 3]) / dx
    dpdx[-1] = (p[-1, 3] - p[-2, 3]) / dx

    dtdx = np.zeros_like(T[:, 0])
    dtdx[1: -1] = (T[2:, 3] - T[:-2, 3]) / (2*dx)
    dtdx[0] = (T[1, 3] - T[0, 3]) / dx
    dtdx[-1] = (T[-1, 3] - T[-2, 3]) / dx

    dpdy = beta * dpdx

    dtdy = beta * dtdx

    delta_y = np.zeros_like(dpdx)

    length = delta_y[nx1:].shape[0]

    delta_y[nx1:] = np.linspace(dy, dy*slope, length)

    p[nx1:, 2] = p[nx1:, 3] - 1 * delta_y[nx1:] * dpdy[nx1:]
    p[nx1:, 1] = p[nx1:, 3] - 2 * delta_y[nx1:] * dpdy[nx1:]
    p[nx1:, 0] = p[nx1:, 3] - 3 * delta_y[nx1:] * dpdy[nx1:]

    T[nx1:, 2] = T[nx1:, 3] - 1 * delta_y[nx1:] * dtdy[nx1:]
    T[nx1:, 1] = T[nx1:, 3] - 2 * delta_y[nx1:] * dtdy[nx1:]
    T[nx1:, 0] = T[nx1:, 3] - 3 * delta_y[nx1:] * dtdy[nx1:]

    U_new[0, 3:, -3:] = p[3:, -3:] / (R * T[3:, -3:])
    U_new[0, 3:, :3] = p[3:, :3] / (R * T[3:, :3])
    U_new[0, -3:, :] = p[-3:, :] / (R * T[-3:, :])

    U_new[1, 3:, -3:] = U_new[0, 3:, -3:] * u[3:, -3:]
    U_new[1, 3:, :3] = U_new[0, 3:, :3] * u[3:, :3]
    U_new[1, -3:, :] = U_new[0, -3:, :] * u[-3:, :]

    U_new[2, 3:, -3:] = U_new[0, 3:, -3:] * v[3:, -3:]
    U_new[2, 3:, :3] = U_new[0, 3:, :3] * v[3:, :3]
    U_new[2, -3:, :] = U_new[0, -3:, :] * v[-3:, :]

    U_new[3, 3:, -3:] = p[3:, -3:] / \
        (gamma - 1) + 0.5 * U_new[0, 3:, -3:] * (u[3:, -3:]**2 + v[3:, -3:]**2)
    U_new[3, 3:, :3] = p[3:, :3] / (gamma - 1) + \
        0.5 * U_new[0, 3:, :3] * (u[3:, :3]**2 + v[3:, :3]**2)
    U_new[3, -3:, :] = p[-3:, :] / \
        (gamma - 1) + 0.5 * U_new[0, -3:, :] * (u[-3:, :]**2 + v[-3:, :]**2)

    return U_new


def weno5_flux(fm2, fm1, f0, f1, f2, epsilon=1e-6):
    """compute 5th order weno flux"""
    f_0 = (1/3) * fm2 - (7/6) * fm1 + (11/6) * f0
    f_1 = (-1/6) * fm1 + (5/6) * f0 + (1/3) * f1
    f_2 = (1/3) * f0 + (5/6) * f1 - (1/6) * f2

    beta_0 = (13/12) * (fm2 - 2*fm1 + f0)**2 + (1/4) * (fm2 - 4*fm1 + 3*f0)**2
    beta_1 = (13/12) * (fm1 - 2*f0 + f1)**2 + (1/4) * (fm1 - f1)**2
    beta_2 = (13/12) * (f0 - 2*f1 + f2)**2 + (1/4) * (3*f0 - 4*f1 + f2)**2

    alpha_0 = 0.1 / (epsilon + beta_0)**2
    alpha_1 = 0.6 / (epsilon + beta_1)**2
    alpha_2 = 0.3 / (epsilon + beta_2)**2

    alpha_sum = alpha_0 + alpha_1 + alpha_2
    w_0 = alpha_0 / alpha_sum
    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum

    weno_flux = w_0 * f_0 + w_1 * f_1 + w_2 * f_2

    return weno_flux
