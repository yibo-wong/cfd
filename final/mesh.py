# FROM PKU CFD CLASS (2024)
import numpy as np
import matplotlib.pyplot as plt
# AB = 1
# AE = 2.4
# DE = 4
# angle = 15

# nx = 70
# ny = 50


def generate_mesh(AB, AE, DE, angle, nx, ny):
    x_l = y_b = 0
    x_B = AB
    x_r = DE
    y_t = AE
    y_C = (DE - AB) * np.tan(np.deg2rad(angle)) + y_b

    nx1 = int(nx * (AB / DE))
    nx2 = nx + 1 - nx1
    nx = nx1 + nx2 - 1

    X_b1 = np.linspace(x_l, x_B, nx1)
    X_b2 = np.linspace(x_B, x_r, nx2)
    Y_b1 = np.ones(nx1) * y_b
    Y_b2 = y_b + (X_b2 - x_B) * (np.deg2rad(angle))
    X_t1 = np.linspace(x_l, x_B, nx1)
    X_t2 = np.linspace(x_B, x_r, nx2)
    X_t = np.linspace(x_l, x_r, nx)
    X_bottom = np.concatenate((X_b1, X_b2[1:]))
    Y_bottom = np.concatenate((Y_b1, Y_b2[1:]))
    X_top = np.concatenate((X_t1, X_t2[1:]))
    Y_top = np.ones(nx) * y_t

    x = np.zeros((nx, ny))
    y = np.zeros((nx, ny))
    for i in range(0, nx):
        for j in range(0, ny):
            x[i, j] = X_bottom[i] + (X_top[i]-X_bottom[i]) * j / (ny - 1)
            y[i, j] = Y_bottom[i] + (Y_top[i]-Y_bottom[i]) * j / (ny - 1)
    return x, y, nx1
