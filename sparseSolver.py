import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

os.makedirs("plots", exist_ok=True)


def matrix_2ndOrder(n, maxVal):
    h = maxVal / (n - 1)
    size = n * n
    A = lil_matrix((size, size), dtype=float)
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[idx, idx] = 1.0  # Dirichlet identity row (unscaled)
            else:
                inv2 = 1.0 / (h * h)
                A[idx, idx] = -4.0 * inv2
                A[idx, idx - n] = 1.0 * inv2  # up
                A[idx, idx + n] = 1.0 * inv2  # down
                A[idx, idx - 1] = 1.0 * inv2  # left
                A[idx, idx + 1] = 1.0 * inv2  # right
    return A.tocsr()


def matrix_4thOrder(n, maxVal):
    h = maxVal / (n - 1)
    size = n * n
    A = lil_matrix((size, size), dtype=float)
    inv = 1.0 / (12.0 * h * h)
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[idx, idx] = 1.0  # Dirichlet identity row (unscaled)
                continue

            A[idx, idx] = -60.0 * inv
            # ±1 neighbors (on-grid for interior rows)
            A[idx, idx - n] = 16.0 * inv  # up    (i-1,j)
            A[idx, idx + n] = 16.0 * inv  # down  (i+1,j)
            A[idx, idx - 1] = 16.0 * inv  # left  (i,j-1)
            A[idx, idx + 1] = 16.0 * inv  # right (i,j+1)

            # ±2 neighbors only if on-grid
            if i > 1:
                A[idx, idx - 2 * n] = -1.0 * inv  # (i-2,j)
            if i < n - 2:
                A[idx, idx + 2 * n] = -1.0 * inv  # (i+2,j)
            if j > 1:
                A[idx, idx - 2] = -1.0 * inv  # (i,j-2)
            if j < n - 2:
                A[idx, idx + 2] = -1.0 * inv  # (i,j+2)
    return A.tocsr()


def rhs_2ndOrder(n, maxVal):
    b = np.zeros((n, n))
    h = maxVal / (n - 1)
    for i in range(n):
        for j in range(n):
            x = j * h
            y = i * h
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                b[i, j] = (np.sin(2.0 * np.pi * x) + np.sin(2.0 * np.pi * y))  # Dirichlet value
            else:
                b[i, j] = -4.0 * (np.pi ** 2) * (np.sin(2.0 * np.pi * x) + np.sin(2.0 * np.pi * y))
    return b


def rhs_4thOrder(n, maxVal):
    b = np.zeros((n, n))
    h = maxVal / (n - 1)
    inv = 1.0 / (12.0 * h * h)
    # Neighbor offsets (di = row/y, dj = col/x)
    nbrs = [(-2, 0, -1.0), (-1, 0, 16.0), (1, 0, 16.0), (2, 0, -1.0),
            (0, -2, -1.0), (0, -1, 16.0), (0, 1, 16.0), (0, 2, -1.0)]
    for i in range(n):
        for j in range(n):
            x = j * h
            y = i * h
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                b[i, j] = (np.sin(2.0 * np.pi * x) + np.sin(2.0 * np.pi * y))  # Dirichlet value
                continue
            val = -4.0 * (np.pi ** 2) * (np.sin(2.0 * np.pi * x) + np.sin(2.0 * np.pi * y))
            # Ghost corrections for ±2 neighbors that go off-grid
            for di, dj, c in nbrs:
                ii, jj = i + di, j + dj
                if not (0 <= ii < n and 0 <= jj < n):
                    xg = x + dj * h  # IMPORTANT: x uses column offset dj
                    yg = y + di * h  # IMPORTANT: y uses row offset di
                    u_g = (np.sin(2.0 * np.pi * xg) + np.sin(2.0 * np.pi * yg))
                    val -= c * inv * u_g  # move missing term to RHS
            b[i, j] = val
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
        b = rhs_setup(n, maxVal, A, order)
        return A, b
    if order == 4:
        A = matrix_setup(n, maxVal, order)
        b = rhs_setup(n, maxVal, A, order)
        return A, b
    else:
        print("order not setup yet!")
        return None


def compute_error(U_numeric, U_exact):
    return np.sqrt(np.mean((U_numeric - U_exact) ** 2))


def perfect_solution(n, max_val, plot=False):
    x = np.linspace(0, max_val, n)
    y = np.linspace(0, max_val, n)
    X, Y = np.meshgrid(x, y)
    U = np.sin(2.0 * np.pi * X) + np.sin(2.0 * np.pi * Y)
    if plot:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, U, cmap=cm.viridis)
        ax.set_xlabel('x');
        ax.set_ylabel('y');
        ax.set_zlabel('u(x,y)')
        ax.set_title("Analytical Solution Surface Plot")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig("plots/Analytical Solution Surface Plot.png", dpi=300, bbox_inches='tight')
        plt.close()
    return U, X, Y


def error_analysis(resolutions, maxVal, plot=False):
    errors_2nd = []
    errors_4th = []
    hs = []

    for res in resolutions:
        A_2nd, b_2nd = numeric_solution_setup(res, maxVal, order=2)
        A_4th, b_4th = numeric_solution_setup(res, maxVal, order=4)
        U, X, Y = perfect_solution(res, maxVal, plot=False)

        solution_2nd = spsolve(A_2nd, b_2nd.flatten())
        U_2nd = solution_2nd.reshape((res, res))
        solution_4th = spsolve(A_4th, b_4th.flatten())
        U_4th = solution_4th.reshape((res, res))

        if plot:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, U_2nd, cmap=cm.viridis)
            ax.set_title(f"Numeric Solution (2nd Order) ({res})")
            plt.savefig(f"plots/Numeric Solution (2nd Order) ({res}).png", dpi=300, bbox_inches='tight')
            plt.close()

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, U_4th, cmap=cm.viridis)
            ax.set_title(f"Numeric Solution (4th Order) ({res})")
            plt.savefig(f"plots/Numeric Solution (4th Order) ({res}).png", dpi=300, bbox_inches='tight')
            plt.close()

        err_2nd = compute_error(solution_2nd, np.array(U).flatten())
        err_4th = compute_error(solution_4th, np.array(U).flatten())
        hs.append(maxVal / (res - 1))
        errors_2nd.append(err_2nd)
        errors_4th.append(err_4th)

    return hs, errors_2nd, errors_4th


def covergence_analysis(res, hs, errs2, errs4, writeOut=True, plot=True):
    rates2 = []
    rates4 = []
    if writeOut:
        for i in range(len(res)):
            print(f"2nd order error for resolution: {res[i]}\n{errs2[i]}")
            print(f"4th order error for resolution: {res[i]}\n{errs4[i]}")
            print("")
        for k in range(1, len(hs)):
            p2 = np.log(errs2[k - 1] / errs2[k]) / np.log(hs[k - 1] / hs[k])
            p4 = np.log(errs4[k - 1] / errs4[k]) / np.log(hs[k - 1] / hs[k])
            rates2.append(p2);
            rates4.append(p4)
        print("Observed rates (2nd):", rates2)
        print("Observed rates (4th):", rates4)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(hs, errs2, 'o-', label='2nd Order Error')
        plt.loglog(hs, errs4, 's-', label='4th Order Error')
        plt.xlabel('Grid Spacing (h)')
        plt.ylabel('RMS Error')
        plt.title('Error vs Grid Spacing (Log-Log Plot)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig("plots/error_convergence_plot_LUCAS.png", dpi=300, bbox_inches='tight')
        plt.close()
    return rates2, rates4


if __name__ == "__main__":
    max_val = 1.0
    resolutions = [10, 20, 50, 100, 200]
    hs, errs2, errs4 = error_analysis(resolutions, max_val, plot=False)
    rates2, rates4 = covergence_analysis(resolutions, hs, errs2, errs4, writeOut=True, plot=True)
