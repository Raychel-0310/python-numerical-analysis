import numpy as np
import matplotlib.pyplot as plt

# 微分方程式 dy/dt = f(t, y)
def f(t, y):
    return -2 * y

# 初期条件と時間設定
t0, y0 = 0, 1
T = 5
N = 100
h = (T - t0) / N
t = np.linspace(t0, T, N + 1)

# オイラー法
def euler_method(f, t, y0):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return y

# Runge-Kutta法（4次）
def runge_kutta_4(f, t, y0):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

# 真の解
y_true = np.exp(-2 * t)

# 数値解
y_euler = euler_method(f, t, y0)
y_rk4 = runge_kutta_4(f, t, y0)

# グラフ描画
plt.figure(figsize=(10, 6))
plt.plot(t, y_true, 'k--', label='Exact')
plt.plot(t, y_euler, 'r-', label='Euler Method')
plt.plot(t, y_rk4, 'b-', label='Runge-Kutta (4th)')
plt.title("ODE Solution: dy/dt = -2y")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()
