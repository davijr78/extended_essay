import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

matplotlib.use('Agg')

os.makedirs("plots", exist_ok=True)

def matrix_2ndOrder(n, maxVal):
    h = maxVal/(n-1)
    size = n*n
    A = np.zeros((size, size))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = -4
            if i > 0:
                A[idx, idx - n] = 1  # up
            if i < n - 1:
                A[idx, idx + n] = 1  # down
            if j > 0:
                A[idx, idx - 1] = 1  # left
            if j < n - 1:
                A[idx, idx + 1] = 1  # right
    return A / h**2

def matrix_4thOrder(n, maxVal):
    h = maxVal / (n-1)
    size = n*n
    A = np.zeros((size, size))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = -60

            if i > 0:
                A[idx, idx - n] = 16  # up
            if i < n - 1:
                A[idx, idx + n] = 16  # down
            if j > 0:
                A[idx, idx - 1] = 16  # left
            if j < n - 1:
                A[idx, idx + 1] = 16  # right

            if i > 1:
                A[idx, idx - 2*n] = -1  # 2 up
            if i < n - 2:
                A[idx, idx + 2*n] = -1  # 2 down
            if j > 1:
                A[idx, idx - 2] = -1  # 2 left
            if j < n - 2:
                A[idx, idx + 2] = -1  # 2 right
    return A / (12 * h ** 2)

def rhs_2ndOrder(n, maxVal):
    b = np.zeros((n, n))
    h = maxVal / (n-1)
    for i in range(n):
        for j in range(n):
            x = j * h
            y = i * h
            b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            x = j * h
            y = i * h
            # Boundaries
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) / h**2
    return b
    
def rhs_4thOrder(n, maxVal):
    b = np.zeros((n, n))
    h = maxVal / (n-1)
    for i in range(n):
        for j in range(n):
            x = j * h
            y = i * h
            b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))

            # corner boundaries
            if i == 0 and j == 0:
               b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (x-h)) + np.sin(2 * np.pi * (y-h)))) / (12*h**2)
            if i == 0 and j == n-1:
               b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (x-h)) + np.sin(2 * np.pi * (y+h)))) / (12*h**2)
            if i == n-1 and j == 0:
               b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (x+h)) + np.sin(2 * np.pi * (y-h)))) / (12*h**2)
            if i == n-1 and j == n-1:
               b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (x+h)) + np.sin(2 * np.pi * (y+h)))) / (12*h**2)
            
            # 1st edge boundaries
            if i == 0 and j > 0 and j < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (x-h)))) / (12*h**2)
            if i == n-1 and j > 0 and j < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (x+h)))) / (12*h**2)
            if j == 0 and i > 0 and i < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (y-h)))) / (12*h**2)
            if j == n-1 and i > 0 and i < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) - 16*(np.sin(2 * np.pi * (y+h)))) / (12*h**2)

            # 2nd edge boundaries
            if i == 1 and j > 0 and j < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))) / (12*h**2)
            if i == n-2 and j > 0 and j < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))) / (12*h**2)
            if j == 1 and i > 0 and i < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))) / (12*h**2)
            if j == n-2 and i > 0 and i < n-1:
                b[i,j] += ((np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))) / (12*h**2)

    return b

def matrix_setup(n, maxVal, order=4):
    if order == 2:
        return matrix_2ndOrder(n, maxVal)
    elif order == 4:
        return matrix_4thOrder(n, maxVal)
    else:
        print("order not yet programed")
        return None

def rhs_setup(n, maxVal, A, order=4):
    if order == 2:
        return rhs_2ndOrder(n, maxVal)
    elif order == 4:
        return rhs_4thOrder(n, maxVal)
    else:
        print("order not yet programed")
        return None

def numeric_solution_setup(n, maxVal, order=4):
    if order == 2:
        A = matrix_setup(n, maxVal, order)
        b = rhs_setup(n, maxVal, order)
        return A, b
    if order == 4:
        A = matrix_setup(n, maxVal, order)
        b = rhs_setup(n, maxVal, order)
        return A, b
    else:
        print("order not setup yet!")
        return None
        
def plot_solution(U, X, Y, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, U, cmap=cm.viridis)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(f"plots/{title}.png", dpi=300, bbox_inches='tight')
    plt.close()

def compute_error(U_numeric, U_exact):
    return np.sqrt(np.mean((U_numeric - U_exact)**2))

def perfect_solution(n, max_val, plot=False):
    analytic_solution = lambda x, y: np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)
    if plot:
        x = np.linspace(0,max_val,n)
        y = np.linspace(0,max_val,n)
        X,Y = np.meshgrid(x,y)
        U = analytic_solution(X,Y)
        plot_solution(U, X, Y, "Analytical Solution Surface Plot")
        return U,X,Y
    else:
        x = np.linspace(0,max_val,n)
        y = np.linspace(0,max_val,n)
        X,Y = np.meshgrid(x,y)
        U = analytic_solution(X,Y)
        return U,X,Y

def error_analysis(resolutions, maxVal, plot=False):
    errors_2nd = []
    errors_4th = []
    hs = []

    for res in resolutions:
        A_2nd, b_2nd = numeric_solution_setup(res, maxVal, order=2)
        A_4th, b_4th = numeric_solution_setup(res, maxVal, order=4)
        U,X,Y = perfect_solution(res, maxVal)

        solution_2nd = np.linalg.solve(A_2nd, b_2nd.flatten())
        U_2nd = solution_2nd.reshape((res,res))
        solution_4th = np.linalg.solve(A_4th, b_4th.flatten())
        U_4th = solution_4th.reshape((res,res))

        if plot:
            plot_solution(U_2nd, X, Y, f"Numeric Solution (2nd Order) ({res})")
            plot_solution(U_4th, X, Y, f"Numeric Solution (4th Order) ({res})")
                
        err_2nd = compute_error(solution_2nd, np.array(U).flatten())
        err_4th = compute_error(solution_4th, np.array(U).flatten())
        hs.append(maxVal / (res-1))
        errors_2nd.append(err_2nd)
        errors_4th.append(err_4th)

    return hs, errors_2nd, errors_4th

def covergence_analysis(res, hs, errs2, errs4, writeOut=True, plot=True):
    if writeOut:
        for i in range(len(res)):
            print(f"2nd order error for resolution: {res[i]}\n{errs2[i]}")
            print(f"4th order error for resolution: {res[i]}\n{errs4[i]}")
            print("")
    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(hs, errs2, 'o-', label='2nd Order Error')
        plt.loglog(hs, errs4, 's-', label='4th Order Error')
        plt.xlabel('Grid Spacing (h)')
        plt.ylabel('L2 Error')
        plt.title('Error vs Grid Spacing (Log-Log Plot)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig("plots/error_convergence_plot.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    max_val = 1
    resolutions = [10, 20, 40, 60, 80]
    hs, errs2, errs4 = error_analysis(resolutions, max_val, plot=True)
    covergence_analysis(resolutions, hs, errs2, errs4, writeOut=True, plot=True)
    
    




    


