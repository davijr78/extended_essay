import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # Color map
matplotlib.use('Agg')


def linear_system_2d_2nd_order(n, maxVal):
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
    return A / (maxVal / (n-1))**2


def linear_system_2d_4th_order(n, maxVal):
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

def rhs_2ndOrder(n, maxVal, A):
    b = np.zeros((n, n))
    h = maxVal / (n-1)
    for i in range(n):
        for j in range(n):
            x = j * h
            y = i * h
            b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))

            idx = i * n + j
            x_val = j * h
            y_val = i * h
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[idx, : ] = 0
                A[idx, idx] = 1
                b[i, j] = -4 * (np.pi ** 2) * np.sin(2 * np.pi * x) + (np.sin(2 * np.pi * y)) + (np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val)/(12 * h ** 2))

    return b



def rhs_4thOrder(n, maxVal, A):
    b = np.zeros((n, n))
    h = maxVal / (n-1)
    for i in range(n):
        for j in range(n):
            x = j * h
            y = i * h
            b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))

            # Enforcing Boundaries
            idx = i * n + j
            x_val = j * h
            y_val = i * h
            if i == 0 and j == 0:
                A[idx, idx] = 1
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) + (-16*(np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val))/(12 * h ** 2)) + (np.sin(2 * np.pi * x_val-h) + np.sin(2 * np.pi * y_val-h)/(12 * h ** 2))
            if i == 1 and j == 0:
                A[idx, idx] = 1
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) + (np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val)/(12 * h ** 2)/(12 * h ** 2))
            if i == 0 and j == 1:
                A[idx, idx] =
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) + (np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val)/(12 * h ** 2)/(12 * h ** 2))
            if i == n-1 and j == n-1:
                A[idx, idx] = 1
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) + (-16*(np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val))/(12 * h ** 2)) + (np.sin(2 * np.pi * x_val+h) + np.sin(2 * np.pi * y_val+h)/(12 * h ** 2))
            if i == n-2 and j == n-1:
                A[idx, idx] = 1
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) + (np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val)/(12 * h ** 2))
            if i == n-1 and j == n-2:
                A[idx, idx] = 1
                b[i, j] = -4 * (np.pi**2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) + (np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val)/(12 * h ** 2))
    return b


def compute_error(U_numeric, U_exact):
    return np.sqrt(np.mean((U_numeric - U_exact)**2))

if __name__ == '__main__':
    points = 10
    max_val = 1
    A = linear_system_2d_4th_order(points, max_val)
    b = rhs_4thOrder(points, max_val, A)

    x = np.linspace(0,max_val,points)
    y = np.linspace(0,max_val,points)
    X,Y = np.meshgrid(x,y)
    analytic_solution = lambda x, y: np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)
    Z = analytic_solution(X,Y)

    # Enforce boundary conditions on A and b
    f = lambda x,y: np.sin(2*np.pi*x)+np.sin(2*np.pi*y)

    solution = np.linalg.solve(A, b.flatten())

    # Reshape the solution vector to a 2D grid (n x n)
    U = solution.reshape((points, points))

    # Plot 3D surface of the analytical solution
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Analytical Solution Surface Plot')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(f"analytical_surface_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, U, cmap=cm.viridis)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Numeric Solution Plot')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(f"solution_plot-{points}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Log-log error vs grid spacing plot for several resolutions
    resolutions = [10, 20, 40, 60, 80, 100]
    errors_2ndOrder = []
    errors_4ndOrder = []
    hs = []

    for res in resolutions:
        A_2nd_order = linear_system_2d_2nd_order(res, max_val)
        A_4th_order = linear_system_2d_4th_order(res, max_val)
        rhs_2nd_Order = rhs_2ndOrder(res, max_val, A_2nd_order)
        rhs_4th_Order = rhs_4thOrder(res, max_val, A_4th_order)

        # Enforcing Boundaries

        sol_2ndOrder = np.linalg.solve(A_2nd_order, rhs_2nd_Order.flatten())
        sol_4thOrder = np.linalg.solve(A_4th_order, rhs_4th_Order.flatten())

        x = np.linspace(0, max_val, res)
        y = np.linspace(0, max_val, res)
        X, Y = np.meshgrid(x, y)

        U_4th = sol_4thOrder.reshape((res, res))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, U_4th, cmap=cm.viridis)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x, y)')
        ax.set_title('Numeric Solution Plot')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(f"4th_order_solution_plot-{res}.png", dpi=300, bbox_inches='tight')
        plt.close()

        U_2nd = sol_2ndOrder.reshape((res, res))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, U_2nd, cmap=cm.viridis)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x, y)')
        ax.set_title('Numeric Solution Plot')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(f"2nd_order_solution_plot-{res}.png", dpi=300, bbox_inches='tight')
        plt.close()


        err_2ndOrder = compute_error(sol_2ndOrder, np.array(analytic_solution(X,Y)).flatten())
        err_4ndOrder = compute_error(sol_4thOrder, np.array(analytic_solution(X,Y)).flatten())

        print(err_2ndOrder)
        print(err_4ndOrder)
        print("\n")

        hs.append(max_val / (res-1))
        errors_2ndOrder.append(err_2ndOrder)
        errors_4ndOrder.append(err_4ndOrder)

# After the loop ends, generate the log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(hs, errors_2ndOrder, 'o-', label='2nd Order Error')
plt.loglog(hs, errors_4ndOrder, 's-', label='4th Order Error')
plt.xlabel('Grid Spacing (h)')
plt.ylabel('L2 Error')
plt.title('Error vs Grid Spacing (Log-Log Plot)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("error_convergence_plot.png", dpi=300, bbox_inches='tight')
plt.close()

'''A = [-60, 16, -1, 0, 0]   b = [-4 * (np.pi ** 2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)) -16*(np.sin(2 * np.pi * x_val) + np.sin(2 * np.pi * y_val)) + 1 * (2 * np.pi * y)) +1*(np.sin(2 * np.pi * x_val), 
                                                                                   -4 * (np.pi ** 2) * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))  + 1 * (2 * np.pi * y)) +1*(np.sin(2 * np.pi * x_val),
                                                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, bc]
    [-16, -60, 0, 0, 0]
    [-1, 0, 0, 0, 0]
    [0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0]
    [ . . . . . . ]
'''
