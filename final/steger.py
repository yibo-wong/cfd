# 2100011025, Yibo Wang, PKU CFD CLASS (2024)
import numpy as np
import matplotlib.pyplot as plt
from mesh import generate_mesh
from tqdm import tqdm
from splitflux import steger_split_x, steger_split_y
import time

AB = 1
AE = 2.4
DE = 4
angle = 10

nx = 160
ny = 80
dx = DE / nx
dy = AE / ny

R = 287.14
gamma = 1.4

nt = 4000
interval = 500

dt = 0.001

mesh_x, mesh_y, nx1 = generate_mesh(AB, AE, DE, angle, nx, ny)


def plot_mesh(mesh_x, mesh_y):
    """plot grid"""
    plt.figure()
    plt.plot(mesh_x, mesh_y, 'k', linewidth=0.5)
    plt.plot(np.transpose(mesh_x), np.transpose(mesh_y), 'k', linewidth=0.5)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./fig/steger/mesh.png", dpi=300)
    plt.show()


plot_mesh(mesh_x, mesh_y)


def compute_flux(U):
    """compute the flux vectors."""
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (gamma - 1) * (U[3] - 0.5 * rho * (u**2 + v**2))
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    F = np.zeros((4, *rho.shape))
    G = np.zeros((4, *rho.shape))

    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v + p
    F[3] = u * (E + p)

    G[0] = rho * v
    G[1] = rho * u * v + p
    G[2] = rho * v**2 + p
    G[3] = v * (E + p)

    return F, G


def initial_conditions(nx, ny):
    """initial conditions for density, velocity, and pressure"""
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.ones((nx, ny))

    u1 = 2
    p1 = 3

    u[:nx//4, :] = u1
    p[:nx//4, :] = p1
    u[:, ny-1] = u1
    u[:nx//4, 0] = u1

    return rho, u, v, p


def jacobi_coeff(nx, ny):
    """jacobi matrix coefficient for mesh transformation"""
    F_x = np.zeros((4, nx, ny))
    F_y = np.zeros((4, nx, ny))
    G_y = np.zeros((4, nx, ny))
    tan_angle = np.tan(np.deg2rad(angle))
    for i in range(nx):
        for j in range(ny):
            x = mesh_x[i, j]
            y = mesh_y[i, j]
            F_x[:, i, j] = 1.0
            F_y[:, i, j] = 0.0
            G_y[:, i, j] = 1.0
            if x >= AB:
                tan_alpha = tan_angle * (1 - y / AE)
                beta = 1 - (x - AB) / (AE - AB) * tan_angle
                F_y[:, i, j] = - tan_alpha / beta
                G_y[:, i, j] = 1 / beta
    return F_x, F_y, G_y


F_x, F_y, G_y = jacobi_coeff(nx, ny)


def steger_step(U, dt):
    """perform one Lax-Friedrichs time step."""
    F = steger_split_x(U)
    G = steger_split_y(U)
    F0, _ = compute_flux(U)

    U_new = U.copy()

    dFdx = (F[:, 1:, 1:-1] - F[:, :-1, 1:-1]) / (dx)
    dFdy = (F0[:, 1:-1, 2:] - F0[:, 1:-1, :-2]) / (2*dy)
    dGdy = (G[:, 1:-1, 1:] - G[:, 1:-1, :-1]) / (dy)

    dU_dt = -dFdx * F_x[:, 1:-1, 1:-1] - dFdy * \
        F_y[:, 1:-1, 1:-1] - dGdy * G_y[:, 1:-1, 1:-1]
    # U_new[:, 1:-1, 1:-1] = U[:, 1:-1, 1:-1] + dt * dU_dt
    U_ave = 0.25 * (U[:, 2:, 1:-1] + U[:, :-2, 1:-1] +
                    U[:, 1:-1, 2:] + U[:, 1:-1, :-2])
    U_in = U[:, 1:-1, 1:-1]

    eta = 0.6

    U_new[:, 1:-1, 1:-1] = eta*U_ave + (1-eta)*U_in + dt * dU_dt

    rho = U_new[0]
    u = U_new[1] / rho
    v = U_new[2] / rho
    p = (gamma - 1) * (U_new[3] - 0.5 * rho * (u**2 + v**2))
    T = p / (R * rho)

    u[1:, -1] = u[1:, -2]
    u[-1, 1:] = u[-2, 1:]

    v[1:, -1] = v[1:, -2]
    v[-1, 1:] = v[-2, 1:]

    p[1:, -1] = p[1:, -2]
    p[1:, 0] = p[1:, 1]
    p[-1, :] = p[-2, :]

    T[1:, -1] = T[1:, -2]
    T[1:, 0] = T[1:, 1]
    T[-1, :] = T[-2, :]

    U_new[0, 1:, -1] = p[1:, -1] / (R * T[1:, -1])
    U_new[0, 1:, 0] = p[1:, 0] / (R * T[1:, 0])
    U_new[0, -1, :] = p[-1, :] / (R * T[-1, :])

    U_new[1, 1:, -1] = U_new[0, 1:, -1] * u[1:, -1]
    U_new[1, 1:, 0] = U_new[0, 1:, 0] * u[1:, 0]
    U_new[1, -1, :] = U_new[0, -1, :] * u[-1, :]

    U_new[2, 1:, -1] = U_new[0, 1:, -1] * v[1:, -1]
    U_new[2, 1:, 0] = U_new[0, 1:, 0] * v[1:, 0]
    U_new[2, -1, :] = U_new[0, -1, :] * v[-1, :]

    U_new[3, 1:, -1] = p[1:, -1] / \
        (gamma - 1) + 0.5 * U_new[0, 1:, -1] * (u[1:, -1]**2 + v[1:, -1]**2)
    U_new[3, 1:, 0] = p[1:, 0] / (gamma - 1) + \
        0.5 * U_new[0, 1:, 0] * (u[1:, 0]**2 + v[1:, 0]**2)
    U_new[3, -1, :] = p[-1, :] / \
        (gamma - 1) + 0.5 * U_new[0, -1, :] * (u[-1, :]**2 + v[-1, :]**2)

    return U_new


def paint(U, name="shock_wave"):
    """plot the results"""
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (gamma - 1) * (U[3] - 0.5 * rho * (u**2 + v**2))
    T = p / (R * rho)

    plt.figure(figsize=(16, 12))

    plt.subplot(321)
    plt.pcolormesh(mesh_x, mesh_y, rho, shading='auto',
                   cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Density')

    plt.subplot(322)
    plt.pcolormesh(mesh_x, mesh_y, u, shading='auto',
                   cmap='viridis')
    plt.colorbar(label='Velocity u')
    plt.title('Velocity u')

    plt.subplot(323)
    plt.pcolormesh(mesh_x, mesh_y, v, shading='auto',
                   cmap='viridis')
    plt.colorbar(label='Velocity v')
    plt.title('Velocity v')

    plt.subplot(324)
    plt.pcolormesh(mesh_x, mesh_y, p, shading='auto',
                   cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title('Pressure')

    plt.subplot(325)
    plt.pcolormesh(mesh_x, mesh_y, np.sqrt(u**2+v**2),
                   shading='auto', cmap='viridis')
    plt.colorbar(label='Velocity')
    plt.title('Velocity')

    plt.subplot(326)
    plt.pcolormesh(mesh_x, mesh_y, T, shading='auto',
                   cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.title('Temperature')

    plt.tight_layout()
    plt.savefig('./fig/steger/gif/'+name+'.png')
    plt.show()
    plt.close()


rho, u, v, p = initial_conditions(nx, ny)
E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
U = np.zeros((4, nx, ny))
U[0] = rho
U[1] = rho * u
U[2] = rho * v
U[3] = E

for t in tqdm(range(nt), desc="time step"):
    U = steger_step(U, dt)
    if (t+1) % interval == 0:
        index = t//interval + 1
        paint(U, name=f"shock_wave_{index:03d}s")
