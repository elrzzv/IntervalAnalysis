import numpy as np
from scipy.optimize import minimize, linprog
import time

########### 1 часть: Прямоугольная матрица
print("=" * 65)
print('1 ЧАСТЬ: ПРЯМОУГОЛЬНАЯ МАТРИЦА')
def analyze_rectangular(mode='tomography'):
    """
    mode = 'tomography' -> rad = delta * ones(3,2)
    mode = 'regression' -> rad = delta * [[1,0],[1,0],[1,0]]
    """
    mid_A1 = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00],
    ])
    if mode == 'tomography':
        rad_unit = np.ones((3, 2))
    else:
        rad_unit = np.array([[1.0, 0.0],
                             [1.0, 0.0],
                             [1.0, 0.0]])

    def min_delta_for_c(c):
        """
        Для фиксированного c: min delta при условии
        A'[i,2] = c * A'[i,1], i=0,1,2, A' ∈ A1(delta).
        Переменные: x = [A'11, A'21, A'31, delta].
        """
        c_obj = np.array([0, 0, 0, 1.0])  # минимизируем delta

        A_ub, b_ub = [], []
        for i in range(3):
            m1, r1 = mid_A1[i, 0], rad_unit[i, 0]
            m2, r2 = mid_A1[i, 1], rad_unit[i, 1]
            # A'[i,1] - delta*r1 <= m1
            row = [0, 0, 0, -r1]; row[i] = 1.0
            A_ub.append(row); b_ub.append(m1)
            # -A'[i,1] - delta*r1 <= -m1
            row = [0, 0, 0, -r1]; row[i] = -1.0
            A_ub.append(row); b_ub.append(-m1)
            # c*A'[i,1] - delta*r2 <= m2
            row = [0, 0, 0, -r2]; row[i] = c
            A_ub.append(row); b_ub.append(m2)
            # -c*A'[i,1] - delta*r2 <= -m2
            row = [0, 0, 0, -r2]; row[i] = -c
            A_ub.append(row); b_ub.append(-m2)

        bounds = [(None, None)] * 3 + [(0, None)]
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method='highs')
        if res.success:
            return res.fun, res.x
        return np.inf, None

    # Перебор c
    c_grid = np.linspace(-10, 10, 4001)
    best = (np.inf, None, None)
    for c in c_grid:
        d, x = min_delta_for_c(c)
        if d < best[0]:
            best = (d, c, x)

    delta_min, c_opt, x_opt = best
    A_prime = np.column_stack([x_opt[:3], c_opt * x_opt[:3]])

    print(f"\n[{mode}] delta_min = {delta_min:.6f}, c_opt = {c_opt:.6f}")
    print("A' =")
    print(A_prime)
    print("rank(A') =", np.linalg.matrix_rank(A_prime))
    print("A' ∈ A1:", np.all(np.abs(A_prime - mid_A1) <= delta_min * rad_unit + 1e-9))

    return delta_min, A_prime


analyze_rectangular('tomography')
analyze_rectangular('regression')


########### 2 часть: Квадратная матрица
print("=" * 65)
print('2 ЧАСТЬ: КВАДРАТНАЯ МАТРИЦА')


np.set_printoptions(precision=6, suppress=True)

# --- входные данные ---
mid_A2 = np.array([
    [1.10, 0.90, 1.10],
    [1.40, 1.00, 0.80],
    [0.80, 1.40, 1.20],
])
rad_unit = np.ones((3, 3))


# --- ядро: минимум |det| при заданном delta ---
def min_abs_det_at_delta(delta, n_starts=30, seed=0):
    R = delta * rad_unit

    def f_local(s):
        S = s.reshape(3, 3)
        return abs(np.linalg.det(mid_A2 + R * S))

    rng = np.random.default_rng(seed)
    bounds = [(-1, 1)] * 9
    best = np.inf
    for _ in range(n_starts):
        s0 = rng.uniform(-1, 1, 9)
        res = minimize(f_local, s0, method='L-BFGS-B', bounds=bounds,
                       options={'gtol': 1e-12, 'maxiter': 500})
        if res.fun < best:
            best = res.fun
    return best


# --- бисекция поверх этой функции ---
def find_true_delta_min(verbose=True):
    lo_d, hi_d = 0.0, 2.0
    while min_abs_det_at_delta(hi_d) > 1e-6 and hi_d < 100:
        lo_d = hi_d
        hi_d *= 2

    for _ in range(30):
        m = 0.5 * (lo_d + hi_d)
        val = min_abs_det_at_delta(m)
        if verbose:
            print(f"  delta={m:.6f},  min|det|={val:.3e}")
        if val > 1e-6:
            lo_d = m
        else:
            hi_d = m

    return 0.5 * (lo_d + hi_d)

delta_true = find_true_delta_min(verbose=False)
print(f"delta_min = {delta_true:.6f}")

# --- построение A' при найденном delta ---
R = delta_true * rad_unit

def f_final(s):
    S = s.reshape(3, 3)
    return abs(np.linalg.det(mid_A2 + R * S))

rng = np.random.default_rng(0)
best_val, best_s = np.inf, None
for _ in range(200):
    s0 = rng.uniform(-1, 1, 9)
    res = minimize(f_final, s0, method='L-BFGS-B',
                   bounds=[(-1,1)]*9, options={'gtol': 1e-14})
    if res.fun < best_val:
        best_val, best_s = res.fun, res.x

A_prime = mid_A2 + R * best_s.reshape(3, 3)
print(f"\ndet(A') = {np.linalg.det(A_prime):.3e}")
print("A' =")
print(A_prime)

lo = mid_A2 - R
up = mid_A2 + R
print("\nA' ∈ A2:", np.all(A_prime >= lo - 1e-9) and np.all(A_prime <= up + 1e-9))