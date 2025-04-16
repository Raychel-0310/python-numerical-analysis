import numpy as np

# 解きたい非線形関数とその導関数
def f(x):
    return x**3 - x - 2

def df(x):
    return 3 * x**2 - 1

# ニュートン法
def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    raise Exception("収束しませんでした")

# 二分法
def bisection_method(f, a, b, tol=1e-8, max_iter=100):
    if f(a) * f(b) >= 0:
        raise Exception("f(a)とf(b)の符号が異なる必要があります")
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or abs(b - a) < tol:
            return c, i + 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    raise Exception("収束しませんでした")

# 割線法
def secant_method(f, x0, x1, tol=1e-8, max_iter=100):
    for i in range(max_iter):
        if f(x1) - f(x0) == 0:
            raise Exception("ゼロ割りを回避できません")
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2, i + 1
        x0, x1 = x1, x2
    raise Exception("収束しませんでした")

# 実行
root_newton, iter_newton = newton_method(f, df, x0=1.5)
root_bisect, iter_bisect = bisection_method(f, a=1, b=2)
root_secant, iter_secant = secant_method(f, x0=1, x1=2)

print(f"ニュートン法: 解 = {root_newton:.10f}, 反復回数 = {iter_newton}")
print(f"二分法: 解 = {root_bisect:.10f}, 反復回数 = {iter_bisect}")
print(f"割線法: 解 = {root_secant:.10f}, 反復回数 = {iter_secant}")
