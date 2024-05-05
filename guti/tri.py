import numpy as np

# nodes_1 = np.array([[30, 20], [0, 20], [0, 0]])
# nodes_2 = np.array([[30, 0], [30, 20], [0, 0]])
nodes_1 = np.array([[1, 0], [1, 1], [0, 1]])
nodes_2 = np.array([[0, 0], [0, 1], [1, 0]])
index_1 = [2, 3, 4]
index_2 = [1, 2, 4]
# nodes_1 = np.array([[2, 1], [2, 0], [0, 1]])
# nodes_2 = np.array([[2, 0], [0, 1], [0, 0]])


def TriStiff(nodes: np.array, nu: float, E: float, t: float = 1):
    D = np.zeros((3, 3))
    D[0, 0] = 1
    D[1, 1] = 1
    D[0, 1] = nu
    D[1, 0] = nu
    D[2, 2] = (1-nu)/2
    D = E/(1-nu**2)*D

    x1 = nodes[0, 0]
    y1 = nodes[0, 1]
    x2 = nodes[1, 0]
    y2 = nodes[1, 1]
    x3 = nodes[2, 0]
    y3 = nodes[2, 1]

    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    A = (x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2
    print(A)

    B = 1/(2*A)*np.array([[b1, 0, b2, 0, b3, 0],
                          [0, c1, 0, c2, 0, c3],
                          [c1, b1, c2, b2, c3, b3]])

    K = t*np.abs(A)*(B.T @ D @ B)
    return K


def get_ele(K: np.array, index: list, x: int, y: int, u: bool, v: bool):
    assert K.shape[0] == K.shape[1] == 2*len(index)

    if (x not in index) or (y not in index):
        # print(x, y, "not in", index)
        return 0
    else:
        x_index = index.index(x)
        y_index = index.index(y)
        x_mat_idx = 2*x_index + (1 if u == False else 0)
        y_mat_idx = 2*y_index + (1 if v == False else 0)
        return K[x_mat_idx, y_mat_idx]


def select_ele(K: np.array, index: list):
    K_local = np.zeros((len(index), len(index)))
    for i in range(len(index)):
        for j in range(len(index)):
            K_local[i, j] = K[index[i], index[j]]
    return K_local


if __name__ == "__main__":
    K1 = TriStiff(nodes_1, 0.25, 1, 1)
    K2 = TriStiff(nodes_2, 0.25, 1, 1)
    # K1 = TriStiff(nodes_1, 1/3, 32/9, 1)
    # K2 = TriStiff(nodes_2, 1/3, 32/9, 1)
    np.set_printoptions(linewidth=np.inf)
    print(K1)
    print("="*80)
    print(K2)
    K_total = np.zeros((8, 8))
    K_index = [1, 2, 3, 4]
    for i in range(1, 5):
        for j in range(1, 5):
            K11 = get_ele(K1, index_1, i, j, True, True) + \
                get_ele(K2, index_2, i, j, True, True)
            K12 = get_ele(K1, index_1, i, j, True, False) + \
                get_ele(K2, index_2, i, j, True, False)
            K21 = get_ele(K1, index_1, i, j, False, True) + \
                get_ele(K2, index_2, i, j, False, True)
            K22 = get_ele(K1, index_1, i, j, False, False) + \
                get_ele(K2, index_2, i, j, False, False)
            K_x_index = K_index.index(i)
            K_y_index = K_index.index(j)
            K_total[2*K_x_index, 2*K_y_index] = K11
            K_total[2*K_x_index, 2*K_y_index+1] = K12
            K_total[2*K_x_index+1, 2*K_y_index] = K21
            K_total[2*K_x_index+1, 2*K_y_index+1] = K22

    print(K_total)

    # K_local = select_ele(K_total, [0, 2, 3])
    # print(K_local)
    # f_ext = np.array([0, 0, -1])
    # u = np.linalg.solve(K_local, f_ext)
    # print(u)
    # u_total = np.array([u[0], 0, u[1], u[2], 0, 0, 0, 0])
    # print(u_total)
    # f_total = K_total@u_total
    # print(f_total)
