import numpy as np

def gaussseidel(A, b):
    x_k = np.zeros_like(b)
    error = 1e3
    ex = 1e-12
    LD_tr_m = np.tril(A)
    U_tr_m = A - LD_tr_m
    L_inv = np.linalg.inv(LD_tr_m)

    while error > ex:
        x_k1 = np.dot(L_inv, b - np.dot(U_tr_m, x_k))
        error = np.linalg.norm(x_k1 - x_k)
        x_k = x_k1
    return x_k

def element_mtrix(ele_num):
    chi = np.zeros([16, 16])

    ele_x_i = coordinates[0][no_ijk[0][ele_num]]
    ele_x_j = coordinates[0][no_ijk[1][ele_num]]
    ele_x_k = coordinates[0][no_ijk[2][ele_num]]

    ele_y_i = coordinates[1][no_ijk[0][ele_num]]
    ele_y_j = coordinates[1][no_ijk[1][ele_num]]
    ele_y_k = coordinates[1][no_ijk[2][ele_num]]

    a_1 = ele_x_k - ele_x_j
    a_2 = ele_x_i - ele_x_k
    a_3 = ele_x_j - ele_x_i
    b_1 = ele_y_j - ele_y_k
    b_2 = ele_y_k - ele_y_i
    b_3 = ele_y_i - ele_y_j

    a_mat = np.array([[a_1], [a_2], [a_3]]) @ np.array([[a_1, a_2, a_3]])
    b_mat = np.array([[b_1], [b_2], [b_3]]) @ np.array([[b_1, b_2, b_3]])

    sum_mat = 1/(4*Area)*(a_mat)

    for i in range(3):
        chi[no_ijk[0][ele_num]][no_ijk[i][ele_num]] = sum_mat[0][i]
        chi[no_ijk[1][ele_num]][no_ijk[i][ele_num]] = sum_mat[1][i]
        chi[no_ijk[2][ele_num]][no_ijk[i][ele_num]] = sum_mat[2][i]
    return chi

boundary = np.array([[80], [60], [60], [20], [80], [0], [0], [20], [80], [0], [0], [20], [100], [100], [100], [100]])

no_ijk = np.array([[0,1,2,0,1,2,4,5,6,4,5,6,8,9,10,8,9,10],[5,6,7,4,5,6,9,10,11,8,9,10,13,14,15,12,13,14],[1,2,3,5,6,7,5,6,7,9,10,11,9,10,11,13,14,15]])

coordinates = np.array([[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]])

Area = 1/2
phi = np.zeros([16, 16])

for i in range(no_ijk.shape[1]):
    phi += element_mtrix(i)

for i in range(phi.shape[1]):
    if i != 5 and i != 6 and i != 9 and i != 10:
        phi[i][:] = 0
        phi[i][i] = 1

x = gaussseidel(phi, boundary).reshape(-1,1)

print(x)