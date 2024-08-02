import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, x0, y0, h, n):
    x = np.linspace(x0, x0 + n*h, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

def heun_method(f, x0, y0, h, n):
    x = np.linspace(x0, x0 + n*h, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + h * (k1 + k2) / 2
    return x, y

def taylor2_method(f, df_dx, df_dy, x0, y0, h, n):
    x = np.linspace(x0, x0 + n*h, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i]) + (h**2 / 2) * (df_dx(x[i], y[i]) + df_dy(x[i], y[i]) * f(x[i], y[i]))
    return x, y

# EDO número 2.

def f2(t, y):
    return -np.sin(t)

def df2_dx(t, y):
    return 0

def df2_dy(t, y):
    return 0

def exact_solution_2(t):
    return 1 - np.cos(t)

x0_2, y0_2 = 0, 1
h_2 = 0.1
n_2 = int((6 * np.pi) / h_2)

x_euler_2, y_euler_2 = euler_method(f2, x0_2, y0_2, h_2, n_2)
x_heun_2, y_heun_2 = heun_method(f2, x0_2, y0_2, h_2, n_2)
x_taylor2_2, y_taylor2_2 = taylor2_method(f2, df2_dx, df2_dy, x0_2, y0_2, h_2, n_2)

# Gráfica de la solución obtenida contra la solución exacta.

x_exact_2 = np.linspace(x0_2, x0_2 + n_2*h_2, n_2 + 1)
y_exact_2 = exact_solution_2(x_exact_2)

plt.figure(figsize=(10, 10))
plt.plot(x_exact_2, y_exact_2, 'k-', label='Exacta')
plt.plot(x_euler_2, y_euler_2, 'r--', label='Euler')
plt.plot(x_heun_2, y_heun_2, 'g-', label='Heun')
plt.plot(x_taylor2_2, y_taylor2_2, 'b:', label='Taylor de Orden 2')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparación de soluciones numéricas y exacta para la Ecuación Diferencial Ordinaria 2')
plt.legend()
plt.grid(True)
plt.show()

# Campo Vectorial con la solución de Taylor de orden 2:

T, Y = np.meshgrid(np.linspace(0, 6*np.pi, 20), np.linspace(-2, 2, 20))
U = 1
V = -np.sin(T)

plt.figure(figsize=(10, 10))
plt.quiver(T, Y, U, V, color='gray')
plt.plot(x_taylor2_2, y_taylor2_2, 'b:', label='Taylor de Orden 2')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Campo vectorial y solución numérica para la Ecuación Diferencial Ordinaria 2')
plt.legend()
plt.grid(True)
plt.show()
