import numpy as np
from matplotlib import pyplot as plt
from mesh import generate_mesh

gamma = 1.4


def compute_flux(rho, u, v, p):
    """Compute the flux vectors."""
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    F = np.zeros(4)
    G = np.zeros(4)

    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v + p
    F[3] = u * (E + p)

    G[0] = rho * v
    G[1] = rho * u * v + p
    G[2] = rho * v**2 + p
    G[3] = v * (E + p)

    return F, G


def compute_U(rho, u, v, p):
    """Compute the flux vectors."""
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    F = np.zeros(4)

    F[0] = rho
    F[1] = rho * u
    F[2] = rho * v
    F[3] = E

    return F


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

    X = np.zeros((4, 4))

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

    X_inv = np.zeros((4, 4))

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

    diag_p = np.zeros((4, 4))

    diag_p[0, 0] = np.maximum(u - c, 0)
    diag_p[1, 1] = np.maximum(u, 0)
    diag_p[2, 2] = np.maximum(u, 0)
    diag_p[3, 3] = np.maximum(u + c, 0)

    # X_inv_U = np.einsum('pqkr,qkr->pkr', X_inv, U)
    # diag_X_inv_U_p = np.einsum('pqkr,qkr->pkr', diag_p, X_inv_U)
    # F_p = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_p)

    # print(X@X_inv)

    F_p = X @ diag_p @ X_inv @ U

    diag_m = np.zeros((4, 4))

    diag_m[0, 0] = np.minimum(u - c, 0)
    diag_m[1, 1] = np.minimum(u, 0)
    diag_m[2, 2] = np.minimum(u, 0)
    diag_m[3, 3] = np.minimum(u + c, 0)

    # diag_X_inv_U_m = np.einsum('pqkr,qkr->pkr', diag_m, X_inv_U)
    # F_m = np.einsum('pqkr,qkr->pkr', X, diag_X_inv_U_m)

    F_m = X @ diag_m @ X_inv @ U
    A = X @ (diag_m + diag_p) @ X_inv
    print(A)
    print(A @ U)

    return F_p, F_m


# e = gamma * p / (rho * (gamma-1))
# c = np.sqrt(gamma * p / rho)
# ek = (u**2 + v**2)/2

# X = np.array(
#     [[1, 0, 1, 1],
#      [u, 1, u, u],
#      [v-c, 0, v, v+c],
#      [e+ek-c*v, u, ek, e+ek+c*v]]
# )

# X_inv = np.array(
#     [[0.5*(v/c + ek/e), -0.5*u/e, -0.5*(v/e+1/c),  0.5/e],
#      [-u, 1, 0, 0],
#      [1-ek/e, u/e, v/e, -1/e],
#      [0.5*(- v/c + ek/e), -0.5*u/e, 0.5*(-v/e+1/c), 0.5/e]]
# )

# diag = np.diag(np.array([v-c, v, v, v+c]))

# print(diag)
# print(X)
# print(X_inv)
# print(X @ X_inv)
# print(np.linalg.norm((X@X_inv - np.eye(4))))
# print(X_inv @ X)

# A = X @ diag @ X_inv

# print(A)

# U1 = compute_U(rho0, u0, v0, p0)
# U2 = compute_U(rho0+drho, u0+du, v0+dv, p0+dp)

# F1, _ = compute_flux(rho0, u0, v0, p0)
# F2, _ = compute_flux(rho0+drho, u0+du, v0+dv, p0+dp)
# print("="*80)
# print(F1, F2)
# print("="*80)

# Fp1, Fm1 = steger_split_x(U1)
# Fp2, Fm2 = steger_split_x(U2)
# A1 = Fp1+Fm1
# A2 = Fp2+Fm2
# print(A2 - A1)
# print("="*80)
# print(Fp2 - Fp1)
# print("="*80)
# print(Fm2 - Fm1)
# print("="*80)
# print(F2 - F1)


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


# F = np.random.rand(20)

# G = weno5_flux(F[:-4], F[1:-3], F[2:-2], F[3:-1], F[4:])

# G_aligned = np.concatenate((np.full(2, np.nan), G, np.full(2, np.nan)))

# plt.plot(F, '-o', label='original')
# plt.plot(G_aligned, '-o', label='smoothed by WENO')
# plt.legend()
# plt.savefig('./weno.png')
# plt.show()

# x = np.array([1, 2, 3, 4, 5])
# print(x.shape[0])


AB = 1
AE = 2.4
DE = 4
angle = 15

nx = 400
ny = 200
dx = DE / nx
dy = AE / ny

R = 287.14
gamma = 1.4

nt = 10000
interval = 500

dt = 1e-6

eta = 0.3


mesh_x, mesh_y, nx1 = generate_mesh(AB, AE, DE, angle, nx, ny)

ans = np.zeros((nx, ny))

beta = 79.8

rho1 = 1.185

rho2 = rho1 * (4*2.4*np.sin(beta)**2) / (4*0.4*np.sin(beta)**2 + 2)

print(rho2)

ftcs = np.load('./ftcs_state.npy')
lax = np.load('lax_state.npy')
roe = np.load('./roe_state.npy')

for i in range(nx):
    for j in range(ny):
        x, y = mesh_x[i, j], mesh_y[i, j]
        area = y - np.tan(np.deg2rad(beta)) * (x - AB)
        ans[i, j] = rho1 if area >= 0 else rho2


plt.figure(figsize=(20, 10))

plt.subplot(221)
plt.pcolormesh(mesh_x, mesh_y, ans, shading='auto',
               cmap='viridis', vmin=1, vmax=3.2)
plt.colorbar(label='Density')
plt.title('Theoretical result')

plt.subplot(222)
plt.pcolormesh(mesh_x, mesh_y, ftcs[0, :, :], shading='auto',
               cmap='viridis', vmin=1, vmax=3.2)
plt.colorbar(label='Density')
plt.title('FTCS - numerical result')

plt.subplot(223)
plt.pcolormesh(mesh_x, mesh_y, lax[0, :, :], shading='auto',
               cmap='viridis', vmin=1, vmax=3.2)
plt.colorbar(label='Density')
plt.title('L-F method - numerical result')

plt.subplot(224)
plt.pcolormesh(mesh_x, mesh_y, roe[0, :, :], shading='auto',
               cmap='viridis', vmin=1, vmax=3.2)
plt.colorbar(label='Density')
plt.title('Roe method - numerical result')
plt.savefig('theoretical_density.png')
plt.close()


p1 = 99719

p2 = p1 * (1 + 2.8/2.4 * (4*np.sin(beta)**2 - 1))

print(p2)


for i in range(nx):
    for j in range(ny):
        x, y = mesh_x[i, j], mesh_y[i, j]
        area = y - np.tan(np.deg2rad(beta)) * (x - AB)
        ans[i, j] = p1 if area >= 0 else p2

plt.figure(figsize=(20, 10))

plt.subplot(221)
plt.pcolormesh(mesh_x, mesh_y, ans, shading='auto',
               cmap='viridis', vmin=100000, vmax=600000)
plt.colorbar(label='Pressure')
plt.title('Theoretical result')

U = ftcs.copy()
p0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

plt.subplot(222)
plt.pcolormesh(mesh_x, mesh_y, p0, shading='auto',
               cmap='viridis', vmin=100000, vmax=600000)
plt.colorbar(label='Pressure')
plt.title('FTCS - numerical result')

U = lax.copy()
p0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

plt.subplot(223)
plt.pcolormesh(mesh_x, mesh_y, p0, shading='auto',
               cmap='viridis', vmin=100000, vmax=600000)
plt.colorbar(label='Pressure')
plt.title('L-F method - numerical result')

U = roe.copy()
p0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) / U[0]**2) * (gamma - 1)

plt.subplot(224)
plt.pcolormesh(mesh_x, mesh_y, p0, shading='auto',
               cmap='viridis', vmin=100000, vmax=600000)
plt.colorbar(label='Pressure')
plt.title('Roe method - numerical result')
plt.savefig('theoretical_pressure.png')
plt.close()


T1 = 293.15

T2 = T1 * (p2/p1) * (rho1/rho2)

print(T2)


for i in range(nx):
    for j in range(ny):
        x, y = mesh_x[i, j], mesh_y[i, j]
        area = y - np.tan(np.deg2rad(beta)) * (x - AB)
        ans[i, j] = T1 if area >= 0 else T2

plt.figure(figsize=(20, 10))

plt.subplot(221)
plt.pcolormesh(mesh_x, mesh_y, ans, shading='auto',
               cmap='viridis', vmin=250, vmax=600)
plt.colorbar(label='Temperature')
plt.title('Theoretical result')

U = ftcs.copy()
T0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) /
      U[0]**2) * (gamma - 1) / (U[0] * R)

plt.subplot(222)
plt.pcolormesh(mesh_x, mesh_y, T0, shading='auto',
               cmap='viridis', vmin=250, vmax=600)
plt.colorbar(label='Temperature')
plt.title('FTCS - numerical result')

U = lax.copy()
T0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) /
      U[0]**2) * (gamma - 1) / (U[0] * R)


plt.subplot(223)
plt.pcolormesh(mesh_x, mesh_y, T0, shading='auto',
               cmap='viridis', vmin=250, vmax=600)
plt.colorbar(label='Temperature')
plt.title('L-F method - numerical result')

U = roe.copy()
T0 = (U[3] - 0.5 * U[0] * (U[1]**2 + U[2]**2) /
      U[0]**2) * (gamma - 1) / (U[0] * R)


plt.subplot(224)
plt.pcolormesh(mesh_x, mesh_y, T0, shading='auto',
               cmap='viridis', vmin=250, vmax=600)
plt.colorbar(label='Temperature')
plt.title('Roe method - numerical result')
plt.savefig('theoretical_temperature.png')
