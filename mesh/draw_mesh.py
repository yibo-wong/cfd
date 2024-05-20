import meshio
import matplotlib.pyplot as plt
import numpy as np

# 读取.msh文件
mesh = meshio.read("cylinder.msh")

# 提取点的坐标
points = mesh.points

# 提取边界单元（线单元）
boundary_cells = None
for cell_block in mesh.cells:
    if cell_block.type == 'line':  # 查找边界上的线单元
        boundary_cells = cell_block.data
        break

# 找出边界上的点的索引
boundary_points_indices = np.unique(boundary_cells.flatten())

# 提取边界点的坐标
boundary_points = points[boundary_points_indices]

# 输出边界点
print("Boundary points indices:", boundary_points_indices)
print("Boundary points coordinates:")
for i, point in zip(boundary_points_indices, boundary_points):
    print(f"Point {i}: {point}")

# 可视化边界点和整个网格
fig, ax = plt.subplots()

# 绘制所有点
ax.plot(points[:, 0], points[:, 1], 'o',
        markersize=1, color='r', label='All Points', alpha=0.5)

# 绘制边界点
ax.plot(boundary_points[:, 0], boundary_points[:, 1],
        'o', markersize=1, color='b', label='Boundary Points', alpha=0.5)

# 绘制单元格
for cell_block in mesh.cells:
    if cell_block.type == 'triangle':  # 只绘制三角形单元格
        for cell_conn in cell_block.data:
            # 获取单元格的连接点的坐标
            cell_points = points[cell_conn]
            # 添加第一个点到末尾以便闭合图形
            cell_points = np.vstack([cell_points, cell_points[0]])
            # 在图上绘制单元格
            ax.plot(cell_points[:, 0], cell_points[:, 1], 'k-', linewidth=0.2)

# 设置轴的比例
ax.set_aspect('equal')
plt.legend()
plt.savefig('cylinder.png', dpi=500)
