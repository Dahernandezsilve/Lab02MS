import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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

#EDO numero 1
def f1(x, y):
    return 22 * np.exp(x / 5) - 5 * x - 25

def df1_dx(x, y):
    return (22 / 5) * np.exp(x / 5) - 5

def df1_dy(x, y):
    return 0

# Solucion EDO
x = sp.symbols('x')
y = sp.Function('y')(x)
edo = sp.Eq(y.diff(x), 22 * sp.exp(x / 5) - 5 * x - 25)
sol = sp.dsolve(edo, y)

C1 = sp.symbols('C1')
const = sp.solve(sol.rhs.subs(x, 0) - (-3), C1)
sol_exacta = sol.subs(C1, const[0])

sol_exacta_np = sp.lambdify(x, sol_exacta.rhs, 'numpy')
print("\nSolucion Exacta EDO 1: ", sol_exacta)

x0_1, y0_1 = 0, -3
h_1 = 0.1
n_1 = int(5 / h_1)

x_euler_1, y_euler_1 = euler_method(f1, x0_1, y0_1, h_1, n_1)
x_heun_1, y_heun_1 = heun_method(f1, x0_1, y0_1, h_1, n_1)
x_taylor2_1, y_taylor2_1 = taylor2_method(f1, df1_dx, df1_dy, x0_1, y0_1, h_1, n_1)

# Valores de x para la solución exacta
x_exact = np.linspace(x0_1, x0_1 + 5, 100)
y_exact = sol_exacta_np(x_exact)

# Graficar soluciones EDO 1
plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, color='skyblue', label='Solución Exacta')
plt.plot(x_euler_1, y_euler_1, 'r--', label='Euler')
plt.plot(x_heun_1, y_heun_1, 'o-.', label='Heun', color='orange')
plt.plot(x_taylor2_1, y_taylor2_1, 'b:', label='Taylor de orden 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Soluciones numéricas y exacta para EDO 1')
plt.legend()
plt.grid(True)
plt.show()

# Graficar campo vectorial
x_vals = np.linspace(0, 5, 30)
y_vals = np.linspace(-15, 15, 30)
X, Y = np.meshgrid(x_vals, y_vals)
U = np.ones_like(X)
V = f1(X, Y)

plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=6, color='lightgreen')
plt.plot(x_heun_1, y_heun_1, 'o-.', label='Heun', color='orange')
plt.plot(x_taylor2_1, y_taylor2_1, 'b:', label='Taylor de orden 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Campo vectorial y solución numérica de la EDO 1')
plt.legend()
plt.grid(True)
plt.ylim([-12, 0])  # Ajusta el rango del eje y según sea necesario
plt.show()

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