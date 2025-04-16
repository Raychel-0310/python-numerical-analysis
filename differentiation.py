import numpy as np
import matplotlib.pyplot as plt

# 関数 f(x) = sin(x) とその真の導関数 f'(x) = cos(x)
def f(x):
    return np.sin(x)

def df_exact(x):
    return np.cos(x)

# 区間と刻み幅
x = np.linspace(0, 2 * np.pi, 100)
h = x[1] - x[0]

# 関数の値
f_x = f(x)

# 数値微分
df_forward = (f_x[1:] - f_x[:-1]) / h                # 前進差分
df_backward = (f_x[1:] - f_x[:-1]) / h               # 後退差分（例の都合で同じにしてます）
df_center = (f_x[2:] - f_x[:-2]) / (2 * h)           # 中心差分

# 真の微分値
df_exact_vals = df_exact(x)

# グラフ表示
plt.figure(figsize=(10, 6))
plt.plot(x[1:-1], df_center, label="Central Difference", linestyle='-', marker='o')
plt.plot(x[1:-1], df_exact_vals[1:-1], label="Exact Derivative", linestyle='--')
plt.title("Numerical Differentiation of sin(x)")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()
