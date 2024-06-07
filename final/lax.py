# 2100011025, Yibo Wang, PKU CFD CLASS (2024)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mesh import generate_mesh
from splitflux import lax_split_x, lax_split_y
from weno import weno5_flux, apply_boundary_condition
import os

os.makedirs("./fig/lax/gif", exist_ok=True)

AB = 1
AE = 2.4
DE = 4
angle = 15

nx = 200
ny = 100
dx = DE / nx
dy = AE / ny

R = 287.14
gamma = 1.4

nt = 25000
interval = 500

dt = 1e-6

mesh_x, mesh_y, nx1 = generate_mesh(AB, AE, DE, angle, nx, ny)


def plot_mesh(mesh_x, mesh_y):
    """plot grid"""
    plt.figure()
    plt.plot(mesh_x, mesh_y, 'k', linewidth=0.5)
    plt.plot(np.transpose(mesh_x), np.transpose(mesh_y), 'k', linewidth=0.5)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./fig/lax/mesh.png", dpi=300)
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

    u1 = 686.47
    p1 = 99719
    rho1 = 1.185

    rho = np.ones((nx, ny)) * rho1
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.ones((nx, ny)) * p1

    u[:nx//4, :] = u1
    # u[:, ny-1] = u1
    # u[:nx//4, 0] = u1

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


def lax_step(U, dt):
    """perform one Lax-Friedrichs time step."""
    F = lax_split_x(U)
    G = lax_split_y(U)
    F0, _ = compute_flux(U)
    Fy = (F0[:, :, :-1] + F0[:, :, 1:]) / 2

    Fx_weno = weno5_flux(F[:, :-4, :], F[:, 1:-3, :],
                         F[:, 2:-2, :], F[:, 3:-1, :], F[:, 4:, :])
    Fx_weno = Fx_weno[:, :, 3:-3]

    Fy_weno = weno5_flux(Fy[:, :, :-4], Fy[:, :, 1:-3],
                         Fy[:, :, 2:-2], Fy[:, :, 3:-1], Fy[:, :, 4:])
    Fy_weno = Fy_weno[:, 3:-3, :]

    Gy_weno = weno5_flux(G[:, :, :-4], G[:, :, 1:-3],
                         G[:, :, 2:-2], G[:, :, 3:-1], G[:, :, 4:])
    Gy_weno = Gy_weno[:, 3:-3, :]

    U_new = U.copy()

    dFdx = (Fx_weno[:, 1:, :] - Fx_weno[:, :-1, :]) / (dx)
    dFdy = (Fy_weno[:, :, 1:] - Fy_weno[:, :, :-1]) / (dy)
    dGdy = (Gy_weno[:, :, 1:] - Gy_weno[:, :, :-1]) / (dy)

    dU_dt = -dFdx * F_x[:, 3:-3, 3:-3] - dFdy * \
        F_y[:, 3:-3, 3:-3] - dGdy * G_y[:, 3:-3, 3:-3]

    U_ave = 0.25 * (U[:, 2:-4, 3:-3] + U[:, 4:-2, 3:-3] +
                    U[:, 3:-3, 2:-4] + U[:, 3:-3, 4:-2])
    U_in = U[:, 3:-3, 3:-3]

    eta = 0.3

    U_new[:, 3:-3, 3:-3] = eta * U_ave + (1-eta) * U_in + dt * dU_dt

    U_new = apply_boundary_condition(U_new, dx, dy, nx1, angle)

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
    plt.savefig('./fig/lax/'+name+'.png')
    plt.show()
    plt.close()


def paint_with_line(U, name="shock_wave"):
    """plot the results"""
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (gamma - 1) * (U[3] - 0.5 * rho * (u**2 + v**2))
    T = p / (R * rho)

    line_x = [1, 3.4]
    line_y = [0, 2.4]

    plt.figure(figsize=(16, 12))

    plt.subplot(321)
    plt.pcolormesh(mesh_x, mesh_y, rho, shading='auto',
                   cmap='viridis')
    plt.plot(line_x, line_y, color='gray',
             linestyle='-.', linewidth=2, alpha=0.5)
    plt.colorbar(label='Density')
    plt.title('Density')

    plt.subplot(322)
    plt.pcolormesh(mesh_x, mesh_y, u, shading='auto',
                   cmap='viridis')
    plt.plot(line_x, line_y, color='gray',
             linestyle='-.', linewidth=2, alpha=0.5)
    plt.colorbar(label='Velocity u')
    plt.title('Velocity u')

    plt.subplot(323)
    plt.pcolormesh(mesh_x, mesh_y, v, shading='auto',
                   cmap='viridis')
    plt.plot(line_x, line_y, color='gray',
             linestyle='-.', linewidth=2, alpha=0.5)
    plt.colorbar(label='Velocity v')
    plt.title('Velocity v')

    plt.subplot(324)
    plt.pcolormesh(mesh_x, mesh_y, p, shading='auto',
                   cmap='viridis')
    plt.plot(line_x, line_y, color='gray',
             linestyle='-.', linewidth=2, alpha=0.5)
    plt.colorbar(label='Pressure')
    plt.title('Pressure')

    plt.subplot(325)
    plt.pcolormesh(mesh_x, mesh_y, np.sqrt(u**2+v**2),
                   shading='auto', cmap='viridis')
    plt.plot(line_x, line_y, color='gray',
             linestyle='-.', linewidth=2, alpha=0.5)
    plt.colorbar(label='Velocity')
    plt.title('Velocity')

    plt.subplot(326)
    plt.pcolormesh(mesh_x, mesh_y, T, shading='auto',
                   cmap='viridis')
    plt.plot(line_x, line_y, color='gray',
             linestyle='-.', linewidth=2, alpha=0.5)
    plt.colorbar(label='Temperature')
    plt.title('Temperature')

    plt.tight_layout()
    plt.savefig('./fig/lax/'+name+'.png')
    plt.show()
    plt.close()


def draw_gif():
    """ produce a GIF file out of PNGs """
    from PIL import Image
    import os
    print("Making GIF...")
    image_folder = './fig/lax/gif/'
    image_files = sorted([img for img in os.listdir(
        image_folder) if img.endswith('.png')])

    images = [Image.open(os.path.join(image_folder, img))
              for img in image_files]

    gif_path = './fig/lax/gif/output.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   optimize=False, duration=100, loop=0)

    print(f"GIF saved as {gif_path}")


rho, u, v, p = initial_conditions(nx, ny)
E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
U = np.zeros((4, nx, ny))
U[0] = rho
U[1] = rho * u
U[2] = rho * v
U[3] = E

paint(U, name="initial_condition")

for t in tqdm(range(nt), desc="time step"):
    U = lax_step(U, dt)
    if (t+1) % interval == 0:
        index = t//interval + 1
        paint(U, name=f"gif/shock_wave_{index:03d}s")
