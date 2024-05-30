import meshio
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str)
parser.add_argument("--line", action="store_true")
args = parser.parse_args()
filename = args.name
with_line = args.line

mesh = meshio.read(filename+".msh")
points = mesh.points
fig, ax = plt.subplots()

# 绘制所有点
ax.plot(points[:, 0], points[:, 1], 'o',
        markersize=1, label='points', color='r',  alpha=0.5)

# 绘制单元格
if with_line:
    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            for cell_conn in cell_block.data:
                # 获取单元格的连接点的坐标
                cell_points = points[cell_conn]
                # 添加第一个点到末尾以便闭合图形
                cell_points = np.vstack([cell_points, cell_points[0]])
                # 在图上绘制单元格
                ax.plot(cell_points[:, 0],
                        cell_points[:, 1], 'k-', linewidth=0.2)

ax.set_aspect('equal')
plt.legend()
plt.savefig(filename+'.png', dpi=500)
