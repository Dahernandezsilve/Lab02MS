import numpy as np

def F(x):
    x1, x2, x3 = x
    return np.array([
        12 * x1 - 3 * x2**2 - 4 * x3 - 7.17,
        x1 + 10 * x2 - x3 - 11.54,
        x2**3 - 7 * x3**3 - 7.631
    ])

def DF(x):
    x1, x2, x3 = x
    return np.array([
        [12, -6 * x2, -4], 
        [1, 10, -1],         
        [0, 3 * x2**3, -21 * x3**2]
    ])

def newtonMultidimensionalMethod(F, DF, x0, max_iter=100, tol=1e-7):
    print("\n🚀 Iniciando el método de Newton multidimensional...")
    x_k = x0
    approximations = [x_k]
    converged = False
    
    for k in range(max_iter):
        J_k = DF(x_k) 
        F_k = F(x_k)  
        
        delta = -np.dot(np.linalg.inv(J_k), F_k) 
        x_k1 = x_k + delta  

        approximations.append(x_k1)
        
        print(f"🔄 Iteración {k + 1}: x_k = {x_k.flatten()}, F(x_k) = {F_k.flatten()}, delta = {delta.flatten()}")

        if np.linalg.norm(delta) < tol:
            converged = True
            print("\n✅ Convergencia alcanzada.")
            break
        
        x_k = x_k1  
    
    return approximations, x_k, converged  

x0 = np.array([0.1, 0.1, 0.1])
max_iter = 100
tol = 1e-7

approximations, solution, converged = newtonMultidimensionalMethod(F, DF, x0, max_iter, tol)

print("📊 Aproximaciones realizadas:")
for i, approx in enumerate(approximations):
    print(f"✍️  Iteración {i}: {approx.flatten()}")

if converged:
    print(f"\n🔚 Solución aproximada: {solution.flatten()} (Convergió)\n")
else:
    print(f"\n🔚 Solución aproximada: {solution.flatten()} (No convergió)\n")
