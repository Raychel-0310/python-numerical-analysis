import numpy as np
from scipy.integrate import quad

def f(x):
    return np.sin(x)

a, b = 0, np.pi
n = 10

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return h * s

def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even")
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n - 1, 2):
        s += 2 * f(a + i * h)
    return h * s / 3

romberg_result, _ = quad(f, a, b)

print("正確な値:", 2)
print("台形則:", trapezoidal_rule(f, a, b, n))
print("シンプソン則:", simpson_rule(f, a, b, n))
print("Romberg法（quad）:", romberg_result)
