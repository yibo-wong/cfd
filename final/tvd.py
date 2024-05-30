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

nx = 100
ny = 50
dx = DE / nx
dy = AE / ny

R = 287.14
gamma = 1.4

nt = 200
dt = 0.001

rho0 = 1
p0 = 100000
u0 = 500
Ma2 = rho0 * u0**2 / (p0 * gamma)
Ma = np.sqrt(Ma2)
Ma_g = rho0 * u0**2 / p0

mesh_x, mesh_y, nx1 = generate_mesh(AB, AE, DE, angle, nx, ny)


def plot_mesh(mesh_x, mesh_y):

    plt.figure()
    plt.plot(mesh_x, mesh_y, 'k', linewidth=0.5)
    plt.plot(np.transpose(mesh_x), np.transpose(mesh_y), 'k', linewidth=0.5)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./mesh.png", dpi=300)
    plt.show()


plot_mesh(mesh_x, mesh_y)


def compute_flux(rho, u, v, p):
    """Compute the flux vectors."""
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
    """Initial conditions for density, velocity, and pressure."""
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


def roe_flux_F(U_L, U_R):
    rho_L = U_L[0]
    u_L = U_L[1] / rho_L
    v_L = U_L[2] / rho_L
    p_L = (gamma - 1) * (U_L[3] - 0.5 * rho_L * (u_L**2 + v_L**2))

    rho_R = U_R[0]
    u_R = U_R[1] / rho_R
    v_R = U_R[2] / rho_R
    p_R = (gamma - 1) * (U_R[3] - 0.5 * rho_R * (u_R**2 + v_R**2))

    assert rho_L >= 0 and rho_R >= 0

    r_L = np.sqrt(rho_L)
    r_R = np.sqrt(rho_R)
    u_hat = (r_L * u_L + r_R * u_R) / (r_L + r_R)
    v_hat = (r_L * v_L + r_R * v_R) / (r_L + r_R)
    H_L = (p_L / (gamma - 1) + 0.5 * rho_L * (u_L**2 + v_L**2)) / rho_L
    H_R = (p_R / (gamma - 1) + 0.5 * rho_R * (u_R**2 + v_R**2)) / rho_R
    H_hat = (r_L * H_L + r_R * H_R) / (r_L + r_R)
    c_hat = np.sqrt((gamma - 1) * (H_hat - 0.5 * (u_hat**2 + v_hat**2)))

    F_L = np.array([rho_L * u_L,
                    rho_L * u_L**2 + p_L,
                    rho_L * u_L * v_L + p_L,
                    u_L * (p_L / (gamma - 1) + 0.5 * rho_L * (u_L**2 + v_L**2) + p_L)])

    F_R = np.array([rho_R * u_R,
                    rho_R * u_R**2 + p_R,
                    rho_R * u_R * v_R + p_R,
                    u_R * (p_R / (gamma - 1) + 0.5 * rho_R * (u_R**2 + v_R**2) + p_R)])

    delta_U = U_R - U_L

    S = np.array([u_hat - c_hat, u_hat, u_hat, u_hat + c_hat])

    eig_vals = np.abs(S)

    flux = 0.5 * (F_L + F_R) - 0.5 * np.dot(eig_vals, delta_U)
    # flux = 0.5 * (F_L + F_R)

    return flux


def roe_flux_G(U_L, U_R):
    rho_L = U_L[0]
    u_L = U_L[1] / rho_L
    v_L = U_L[2] / rho_L
    p_L = (gamma - 1) * (U_L[3] - 0.5 * rho_L * (u_L**2 + v_L**2))

    rho_R = U_R[0]
    u_R = U_R[1] / rho_R
    v_R = U_R[2] / rho_R
    p_R = (gamma - 1) * (U_R[3] - 0.5 * rho_R * (u_R**2 + v_R**2))

    r_L = np.sqrt(rho_L)
    r_R = np.sqrt(rho_R)
    u_hat = (r_L * u_L + r_R * u_R) / (r_L + r_R)
    v_hat = (r_L * v_L + r_R * v_R) / (r_L + r_R)
    H_L = (p_L / (gamma - 1) + 0.5 * rho_L * (u_L**2 + v_L**2)) / rho_L
    H_R = (p_R / (gamma - 1) + 0.5 * rho_R * (u_R**2 + v_R**2)) / rho_R
    H_hat = (r_L * H_L + r_R * H_R) / (r_L + r_R)
    c_hat = np.sqrt((gamma - 1) * (H_hat - 0.5 * (u_hat**2 + v_hat**2)))

    F_L = np.array([rho_L * v_L,
                    rho_L * u_L * v_L + p_L,
                    rho_L * v_L**2 + p_L,
                    v_L * (p_L / (gamma - 1) + 0.5 * rho_L * (u_L**2 + v_L**2) + p_L)])

    F_R = np.array([rho_R * v_R,
                    rho_R * u_R * v_R + p_R,
                    rho_R * v_R**2 + p_R,
                    v_R * (p_R / (gamma - 1) + 0.5 * rho_R * (u_R**2 + v_R**2) + p_R)])
    # print(F_L, F_R)

    delta_U = U_R - U_L

    S = np.array([v_hat - c_hat, v_hat, v_hat, v_hat + c_hat])

    eig_vals = np.abs(S)

    flux = 0.5 * (F_L + F_R) - 0.5 * np.dot(eig_vals, delta_U)
    # flux = 0.5 * (F_L + F_R)

    # print(flux)

    return flux


def roe_step(U, dt):
    """Perform one Lax-Friedrichs time step."""
    F, G = compute_flux(U[0], U[1] / U[0], U[2] / U[0], (U[3] -
                        0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1))

    U_new = U.copy()

    F_hat = np.zeros((4, nx-1, ny))
    for i in range(nx-1):
        for j in range(ny):
            F_hat[:, i, j] = roe_flux_F(U[:, i, j], U[:, i+1, j])
    dFdx = (F_hat[:, 1:, :] - F_hat[:, :-1, :]) / (2*dx)
    dFdx = dFdx[:, :, 1:-1]

    G_hat = np.zeros((4, nx, ny-1))
    for i in range(nx):
        for j in range(ny-1):
            G_hat[:, i, j] = roe_flux_G(U[:, i, j], U[:, i, j+1])
    dGdy = (G_hat[:, :, 1:] - G_hat[:, :, :-1]) / (2*dy)
    dGdy = dGdy[:, 1:-1, :]

    print(U)

    print("="*80)
    # print(G[3, 20:30, :])
    # print("*"*80)
    # print(G_hat[3, 20:30, :])
    # print("="*80)

    # print(dFdx)

    dUdt = -(dFdx + dGdy)
    U_new[:, 1:-1, 1:-1] += dt * dUdt
    # dU_dt = -((F[:, 2:, 1:-1] - F[:, :-2, 1:-1]) / (2 * dx)) * \
    #     F_x[:, 2:, 1:-1] - ((F[:, 1:-1, 2:] - F[:, 1:-1, :-2]) / (2 * dy)) * \
    #     F_y[:, 2:, 1:-1] - ((G[:, 1:-1, 2:] - G[:, 1:-1, :-2]
    #                          ) / (2 * dy)) * G_y[:, 2:, 1:-1]
    # U_new[:, 1:-1, 1:-1] = 0.25 * \
    #     (U[:, 2:, 1:-1] + U[:, :-2, 1:-1] +
    #      U[:, 1:-1, 2:] + U[:, 1:-1, :-2]) + dt * dU_dt

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

    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (gamma - 1) * (U[3] - 0.5 * rho * (u**2 + v**2))
    T = p / (R * rho)

    # Plot results
    plt.figure(figsize=(16, 12))

    plt.subplot(321)
    # plt.imshow(rho, origin='lower', aspect='auto')
    plt.pcolormesh(mesh_x, mesh_y, rho, shading='auto', cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Density')

    plt.subplot(322)
    # plt.imshow(u, origin='lower', aspect='auto')
    plt.pcolormesh(mesh_x, mesh_y, u, shading='auto', cmap='viridis')
    plt.colorbar(label='Velocity u')
    plt.title('Velocity u')

    plt.subplot(323)
    # plt.imshow(v, origin='lower', aspect='auto')
    plt.pcolormesh(mesh_x, mesh_y, v, shading='auto', cmap='viridis')
    plt.colorbar(label='Velocity v')
    plt.title('Velocity v')

    plt.subplot(324)
    # plt.imshow(p, origin='lower', aspect='auto')
    plt.pcolormesh(mesh_x, mesh_y, p, shading='auto', cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title('Pressure')

    plt.subplot(325)
    # plt.imshow(p, origin='lower', aspect='auto')
    plt.pcolormesh(mesh_x, mesh_y, np.sqrt(u**2+v**2),
                   shading='auto', cmap='viridis')
    plt.colorbar(label='Velocity')
    plt.title('Velocity')

    plt.subplot(326)
    # plt.imshow(p, origin='lower', aspect='auto')
    plt.pcolormesh(mesh_x, mesh_y, T, shading='auto', cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.title('Temperature')

    plt.tight_layout()
    plt.savefig('./fig/'+name+'.png')
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
    U = roe_step(U, dt)
    if (t+1) % 10 == 0:
        paint(U, name=f"shock_wave_{t//10 + 1}s")
