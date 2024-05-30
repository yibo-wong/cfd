import numpy as np
import matplotlib.pyplot as plt
import meshio
from matplotlib.tri import Triangulation
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg
import time


parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str)
args = parser.parse_args()
filename = args.name

mesh = meshio.read(filename+".msh")

points = mesh.points
points = [[point[0], point[1]] for point in points]
points = np.array(points)

cells = mesh.cells
cells = [cell for cell in cells if cell.type == 'triangle'][0].data
cells = np.array(cells)

NP = points.shape[0]
NC = cells.shape[0]

K = np.zeros((NP, NP))
f = np.zeros(NP)

lx = 5
ly = 5
r = 1

v_inf = 1.0
Gamma = 5.0

for cell in cells:
    point = np.array([points[cell[i]] for i in range(3)])
    x0, y0 = point[0, 0], point[0, 1]
    x1, y1 = point[1, 0], point[1, 1]
    x2, y2 = point[2, 0], point[2, 1]

    beta0, beta1, beta2 = y1-y2, y2-y0, y0-y1
    gamma0, gamma1, gamma2 = x2-x1, x0-x2, x1-x0
    A = 0.5 * ((x1-x0) * (y2-y0) - (y1-y0) * (x2-x0))

    K[cell[0], cell[0]] += (beta0 * beta0 + gamma0 * gamma0) / (4*A)
    K[cell[1], cell[1]] += (beta1 * beta1 + gamma1 * gamma1) / (4*A)
    K[cell[2], cell[2]] += (beta2 * beta2 + gamma2 * gamma2) / (4*A)

    K[cell[0], cell[1]] += (beta0 * beta1 + gamma0 * gamma1) / (4*A)
    K[cell[1], cell[0]] += (beta1 * beta0 + gamma1 * gamma0) / (4*A)

    K[cell[0], cell[2]] += (beta0 * beta2 + gamma0 * gamma2) / (4*A)
    K[cell[2], cell[0]] += (beta2 * beta0 + gamma2 * gamma0) / (4*A)

    K[cell[1], cell[2]] += (beta1 * beta2 + gamma1 * gamma2) / (4*A)
    K[cell[2], cell[1]] += (beta2 * beta1 + gamma2 * gamma1) / (4*A)


def equal(a, b, error=1e-4):
    return abs(a - b) <= error


psi0 = np.zeros(NP)
boundary = []
inner = []

# ======================== FOR 1/4 or 1/2 AREA ========================
for i, point in enumerate(points):
    px, py = point[0], point[1]
    if equal(py, 0):
        psi0[i] = 0
        psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
        boundary.append(i)
    elif equal(py, ly):
        psi0[i] = ly
        psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
        boundary.append(i)
    elif equal(px, lx):
        psi0[i] = py
        psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
        boundary.append(i)
    elif equal(py, -ly):
        psi0[i] = -ly
        psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
        boundary.append(i)
    elif equal(px, -lx):
        psi0[i] = py
        psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
        boundary.append(i)
    elif equal(np.sqrt(px**2 + py**2), r):
        psi0[i] = 0
        psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
        boundary.append(i)
    else:
        inner.append(i)

# ======================== FOR FULL AREA ========================
# for i, point in enumerate(points):
#     px, py = point[0], point[1]
#     if equal(py, ly):
#         psi0[i] = ly
#         psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
#         psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
#         boundary.append(i)
#     elif equal(py, -ly):
#         psi0[i] = -ly
#         psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
#         psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
#         boundary.append(i)
#     elif equal(px, lx):
#         psi0[i] = py
#         psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
#         psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
#     elif equal(px, -lx):
#         psi0[i] = py
#         psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
#         psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
#         boundary.append(i)
#     elif equal(np.sqrt(px**2 + py**2), r):
#         psi0[i] = 0
#         psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
#         psi0[i] += Gamma / (2*np.pi) * np.arctan2(py, px)
#         boundary.append(i)
#     else:
#         inner.append(i)

f = - K @ psi0
K_sol = K[np.ix_(inner, inner)]
f_sol = f[np.ix_(inner)]

print(f"dimension: {len(f)}")

# ================= SOLVE =================

# psi_sol = np.linalg.solve(K_sol, f_sol)

# 将NumPy数组转换为稀疏矩阵格式
K_sparse = csr_matrix(K_sol)

# 计时并使用spsolve求解
start_time = time.time()
psi_spsolve = spsolve(K_sparse, f_sol)
end_time = time.time()
print("spsolve time:", end_time - start_time, "seconds")

# 计时并使用共轭梯度法求解
start_time = time.time()
psi_cg, info = cg(K_sparse, f_sol)
end_time = time.time()
print("cg time:", end_time - start_time, "seconds")
print("cg convergence info:", info)

# 计时并使用numpy.linalg.solve求解
start_time = time.time()
psi_linalg = np.linalg.solve(K_sol, f_sol)
end_time = time.time()
print("np.linalg.solve time:", end_time - start_time, "seconds")

print("error between linalg and spsolve:",
      np.linalg.norm(psi_linalg-psi_spsolve))

print("error between linalg and cg:",
      np.linalg.norm(psi_linalg-psi_cg))

psi = psi0.copy()
psi_sol = psi_cg

for i, j in enumerate(inner):
    psi[j] = psi_sol[i]

# print(psi)

tri = Triangulation(points[:, 0], points[:, 1], triangles=cells)
levels = np.linspace(psi.min(), psi.max(), 50)

psi_a = np.zeros(NP)
for i, point in enumerate(points):
    px, py = point[0], point[1]
    psi_a[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
    psi_a[i] += Gamma / (2*np.pi) * np.arctan2(py, px)

error = psi_a - psi
mean_error = np.linalg.norm(error, ord=1) / len(error)
print("error per node:", mean_error)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * lx, ly))

contour1 = ax1.tricontour(tri, psi, levels=levels, colors='k', linewidths=0.5)
contourf1 = ax1.tricontourf(tri, psi, levels=levels, cmap='viridis')
fig.colorbar(contourf1, ax=ax1)
ax1.set_aspect(1)
ax1.set_title('Numerical Solution')

contour2 = ax2.tricontour(tri, psi_a, levels=levels,
                          colors='k', linewidths=0.5)
contourf2 = ax2.tricontourf(tri, psi_a, levels=levels, cmap='viridis')
fig.colorbar(contourf2, ax=ax2)
ax2.set_aspect(1)
ax2.set_title('Analytical Solution')

plt.savefig(filename+'_solution.png', dpi=500)
plt.show()
