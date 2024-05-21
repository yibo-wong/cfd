import numpy as np
import matplotlib.pyplot as plt
import meshio
from matplotlib.tri import Triangulation

mesh = meshio.read("cylinder.msh")

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

for cell in cells:
    point = np.array([points[cell[i]] for i in range(3)])
    A = 0.5 * ((point[1, 0] - point[0, 0]) * (point[2, 1] - point[0, 1]) -
               (point[1, 1] - point[0, 1]) * (point[2, 0] - point[0, 0]))

    b0 = (point[1, 1] - point[2, 1]) / (2*A)
    b1 = (point[2, 1] - point[0, 1]) / (2*A)
    b2 = (point[0, 1] - point[1, 1]) / (2*A)

    c0 = (-point[1, 0] + point[2, 0]) / (2*A)
    c1 = (-point[2, 0] + point[0, 0]) / (2*A)
    c2 = (-point[0, 0] + point[1, 0]) / (2*A)

    K[cell[0], cell[0]] += b0**2 + c0**2
    K[cell[1], cell[1]] += b1**2 + c1**2
    K[cell[2], cell[2]] += b2**2 + c2**2

    K[cell[0], cell[1]] += b0*b1 + c0*c1
    K[cell[1], cell[0]] += b0*b1 + c0*c1

    K[cell[0], cell[2]] += b0*b2 + c0*c2
    K[cell[2], cell[0]] += b0*b2 + c0*c2

    K[cell[1], cell[2]] += b1*b2 + c1*c2
    K[cell[2], cell[1]] += b1*b2 + c1*c2


def equal(a, b, error=1e-4):
    return abs(a - b) <= error


psi0 = np.zeros(NP)
boundary = []
inner = []

for i, point in enumerate(points):
    px, py = point[0], point[1]
    if equal(py, 0):
        psi0[i] = 0
        # psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        boundary.append(i)
    elif equal(py, ly):
        psi0[i] = ly
        # psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        boundary.append(i)
    elif equal(px, lx):
        psi0[i] = py
        # psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        boundary.append(i)
    elif equal(np.sqrt(px**2 + py**2), r):
        psi0[i] = 0
        # psi0[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))
        boundary.append(i)
    else:
        inner.append(i)

f = - K @ psi0
K_sol = K[np.ix_(inner, inner)]
f_sol = f[np.ix_(inner)]

psi_sol = np.linalg.solve(K_sol, f_sol)

psi = psi0.copy()

for i, j in enumerate(inner):
    psi[j] = psi_sol[i]

print(psi)

tri = Triangulation(points[:, 0], points[:, 1], triangles=cells)
levels = np.linspace(psi.min(), psi.max(), 50)
# plt.figure(figsize=(lx, ly))
# plt.tricontour(tri, psi, levels=levels, colors='k', linewidths=0.5)
# plt.tricontourf(tri, psi, levels=levels, cmap='viridis')
# plt.colorbar()
# plt.gca().set_aspect(1)
# plt.savefig('cylinder_flow.png', dpi=500)

psi_a = np.zeros(NP)
for i, point in enumerate(points):
    px, py = point[0], point[1]
    psi_a[i] = v_inf * py * (1 - r**2 / (px**2 + py**2))

# plt.figure(figsize=(lx, ly))
# plt.tricontour(tri, psi_a, levels=levels, colors='k', linewidths=0.5)
# plt.tricontourf(tri, psi_a, levels=levels, cmap='viridis')
# plt.colorbar()
# plt.gca().set_aspect(1)
# plt.savefig('cylinder_flow_analytical.png', dpi=500)

error = psi_a - psi
mean_error = np.linalg.norm(error, ord=1) / len(error)
print(mean_error)

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

plt.savefig('cylinder_flow_comparison.png', dpi=500)
plt.show()
