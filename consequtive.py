import numpy as np
from scipy.linalg import lu

# 行列と右辺ベクトル
A = np.array([[4.0, 1.0, 2.0],
              [3.0, 5.0, 1.0],
              [1.0, 1.0, 3.0]])
b = np.array([4.0, 7.0, 3.0])

# Gauss消去法（NumPyで直接）
x_gauss = np.linalg.solve(A, b)

# LU分解（scipy）
P, L, U = lu(A)
y = np.linalg.solve(L, P @ b)
x_lu = np.linalg.solve(U, y)

# Jacobi法
def jacobi(A, b, x0=None, tol=1e-8, max_iter=100):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise Exception("収束しませんでした")

# Gauss-Seidel法
def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=100):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x
    raise Exception("収束しませんでした")

# 実行
x_jacobi = jacobi(A, b)
x_gs = gauss_seidel(A, b)

# 結果表示
print("Gauss消去法       :", x_gauss)
print("LU分解             :", x_lu)
print("Jacobi法           :", x_jacobi)
print("Gauss-Seidel法     :", x_gs)
