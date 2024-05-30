# 2100011025, Yibo Wang, PKU CFD CLASS (2024)
import numpy as np
import matplotlib.pyplot as plt
from mesh import generate_mesh
from tqdm import tqdm
import time

AB = 1
AE = 2.4
DE = 4
angle = 10

nx = 300
ny = 150
dx = DE / nx
dy = AE / ny

R = 287.14
gamma = 1.4

nt = 2000
dt = 0.001

mesh_x, mesh_y, nx1 = generate_mesh(AB, AE, DE, angle, nx, ny)

plt.figure()
plt.plot(mesh_x, mesh_y, 'k', linewidth=0.5)
plt.plot(np.transpose(mesh_x), np.transpose(mesh_y), 'k', linewidth=0.5)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("./fig/weno/mesh.png", dpi=300)
plt.show()


def compute_flux(rho, u, v, p):
    """compute the flux vectors."""
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
    """initial conditions for density, velocity, and pressure."""
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.ones((nx, ny))

    # Shock wave initial conditions
    u[:nx//4, :] = 3
    p[:nx//4, :] = 5
    u[:, ny-1] = 3
    u[:nx//4, 0] = 3

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


def apply_boundary_condition(U_new):
    """update boundary condition - both Dirichlet and Newmann"""
    rho = U_new[0]
    u = U_new[1] / rho
    v = U_new[2] / rho
    p = (gamma - 1) * (U_new[3] - 0.5 * rho * (u**2 + v**2))
    T = p / (R * rho)

    u[3:, -1] = u[3:, -2] = u[3:, -3] = u[3:, -4]
    u[-1, 3:] = u[-2, 3:] = u[-3, 3:] = u[-4, 3:]

    v[3:, -1] = v[3:, -2] = v[3:, -3] = v[3:, -4]
    v[-1, 3:] = v[-2, 3:] = v[-3, 3:] = v[-4, 3:]

    p[3:, -1] = p[3:, -2] = p[3:, -3] = p[3:, -4]
    p[3:, 0] = p[3:, 1] = p[3:, 2] = p[3:, 3]
    p[-1, :] = p[-2, :] = p[-3, :] = p[-4, :]

    T[3:, -1] = T[3:, -2] = T[3:, -3] = T[3:, -4]
    T[3:, 0] = T[3:, 1] = T[3:, 2] = T[3:, 3]
    T[-1, :] = T[-2, :] = T[-3, :] = T[-4, :]

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


def weno5_flux(fm2, fm1, f0, f1, f2, epsilon=1e-4):
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


def weno_step(U, dt):
    """perform one time weno step."""
    F, G = compute_flux(U[0], U[1] / U[0], U[2] / U[0], (U[3] -
                        0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1))
    Fx_weno = weno5_flux(F[:, :-4, :], F[:, 1:-3, :],
                         F[:, 2:-2, :], F[:, 3:-1, :], F[:, 4:, :])
    Fx_weno = Fx_weno[:, :, 2:-2]
    Fy_weno = weno5_flux(F[:, :, :-4], F[:, :, 1:-3],
                         F[:, :, 2:-2], F[:, :, 3:-1], F[:, :, 4:])
    Fy_weno = Fy_weno[:, 2:-2, :]
    Gy_weno = weno5_flux(G[:, :, :-4], G[:, :, 1:-3],
                         G[:, :, 2:-2], G[:, :, 3:-1], G[:, :, 4:])
    Gy_weno = Gy_weno[:, 2:-2, :]

    U_new = U.copy()
    dU_dt = -((Fx_weno[:, 2:, 1:-1] - Fx_weno[:, :-2, 1:-1]) / (2 * dx)) * F_x[:, 3:-3, 3:-3] \
        - ((Fy_weno[:, 1:-1, 2:] - Fy_weno[:, 1:-1, :-2]) / (2 * dy)) * F_y[:, 3:-3, 3:-3] \
        - ((Gy_weno[:, 1:-1, 2:] - Gy_weno[:, 1:-1, :-2]) /
           (2 * dy)) * G_y[:, 3:-3, 3:-3]
    U_new[:, 3:-3, 3:-3] = 0.25 * \
        (U_new[:, 2:-4, 3:-3] + U_new[:, 4:-2, 3:-3] +
         U_new[:, 3:-3, 2:-4] + U_new[:, 3:-3, 4:-2]) + dt * dU_dt

    U_new = apply_boundary_condition(U_new)

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
                   cmap='viridis', vmin=0, vmax=5)
    plt.colorbar(label='Density')
    plt.title('Density')

    plt.subplot(322)
    plt.pcolormesh(mesh_x, mesh_y, u, shading='auto',
                   cmap='viridis', vmin=0, vmax=5)
    plt.colorbar(label='Velocity u')
    plt.title('Velocity u')

    plt.subplot(323)
    plt.pcolormesh(mesh_x, mesh_y, v, shading='auto',
                   cmap='viridis', vmin=0, vmax=3)
    plt.colorbar(label='Velocity v')
    plt.title('Velocity v')

    plt.subplot(324)
    plt.pcolormesh(mesh_x, mesh_y, p, shading='auto',
                   cmap='viridis', vmin=0, vmax=20)
    plt.colorbar(label='Pressure')
    plt.title('Pressure')

    plt.subplot(325)
    plt.pcolormesh(mesh_x, mesh_y, np.sqrt(u**2+v**2),
                   shading='auto', cmap='viridis', vmin=0, vmax=5)
    plt.colorbar(label='Velocity')
    plt.title('Velocity')

    plt.subplot(326)
    plt.pcolormesh(mesh_x, mesh_y, T, shading='auto',
                   cmap='viridis', vmin=0, vmax=0.03)
    plt.colorbar(label='Temperature')
    plt.title('Temperature')

    plt.tight_layout()
    plt.savefig('./fig/weno/gif/'+name+'.png')
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
    U = weno_step(U, dt)
    interval = 20
    if (t+1) % interval == 0:
        index = t//interval + 1
        paint(U, name=f"shock_wave_{index:03d}s")
