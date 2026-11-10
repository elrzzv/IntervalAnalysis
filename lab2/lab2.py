# lab2_variant5.py
# Лабораторная работа №2 по интервальному анализу.
# Вариант 5.

import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution


# ============================================================
# Интервальная арифметика
# ============================================================
class Interval:
    def __init__(self, lo, hi):
        self.lo = float(lo)
        self.hi = float(hi)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other, other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other, other)
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other, other)
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other, other)
        a, b = self.lo, self.hi
        c, d = other.lo, other.hi
        products = [a * c, a * d, b * c, b * d]
        return Interval(min(products), max(products))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other, other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("division by interval containing zero")
        inv = Interval(1.0 / other.hi, 1.0 / other.lo)
        return self.__mul__(inv)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __pow__(self, n):
        if n == 0:
            return Interval(1, 1)
        if n % 2 == 0:
            if self.lo >= 0:
                return Interval(self.lo ** n, self.hi ** n)
            elif self.hi <= 0:
                return Interval(self.hi ** n, self.lo ** n)
            else:
                return Interval(0, max(self.lo ** n, self.hi ** n))
        else:
            return Interval(self.lo ** n, self.hi ** n)

    def mid(self):
        return 0.5 * (self.lo + self.hi)

    def rad(self):
        return 0.5 * (self.hi - self.lo)

    def wid(self):
        return self.hi - self.lo

    def intersect(self, other):
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def __repr__(self):
        return f"[{self.lo:.2f}, {self.hi:.2f}]"


def sin_int(x):
    """Интервальное расширение sin."""
    if x.wid() >= 2 * math.pi:
        return Interval(-1, 1)
    lo, hi = x.lo, x.hi
    candidates = [lo, hi]
    k_min = math.floor((lo - math.pi / 2) / math.pi)
    k_max = math.ceil((hi - math.pi / 2) / math.pi)
    for k in range(k_min, k_max + 1):
        t = math.pi / 2 + k * math.pi
        if lo <= t <= hi:
            candidates.append(t)
    vals = [math.sin(t) for t in candidates]
    return Interval(min(vals), max(vals))


def cos_int(x):
    return sin_int(x + math.pi / 2)


def sinc_int(S):
    """Интервальное расширение sinc(x)=sin(x)/x, sinc(0)=1."""
    if S.lo == S.hi:
        x = S.lo
        val = 1.0 if x == 0 else math.sin(x) / x
        return Interval(val, val)
    if S.lo <= 0 <= S.hi:
        vals = [1.0]
        if S.lo != 0:
            vals.append(math.sin(S.lo) / S.lo)
        if S.hi != 0:
            vals.append(math.sin(S.hi) / S.hi)
        return Interval(min(vals), max(vals))
    else:
        def sinc(x):
            return math.sin(x) / x if x != 0 else 1.0
        res_min = minimize_scalar(lambda x: sinc(x), bounds=(S.lo, S.hi), method='bounded')
        res_max = minimize_scalar(lambda x: -sinc(x), bounds=(S.lo, S.hi), method='bounded')
        return Interval(res_min.fun, -res_max.fun)


# ============================================================
# Точные области значений функций и их производных
# ============================================================
def _exact_scalar_range(f, lo, hi):
    """Точная область значений f на [lo, hi] через Brent + границы."""
    res_min = minimize_scalar(lambda x: f(x), bounds=(lo, hi), method='bounded')
    res_max = minimize_scalar(lambda x: -f(x), bounds=(lo, hi), method='bounded')
    min_val = min(res_min.fun, f(lo), f(hi))
    max_val = max(-res_max.fun, f(lo), f(hi))
    return min_val, max_val

def interval_hull(I1, I2):
    return Interval(min(I1.lo, I2.lo), max(I1.hi, I2.hi))


def lipschitz_f3(X, Y):
    """Покоординатные константы Липшица для f3 на брусе X × Y."""
    xs = [X.lo, X.hi]
    ys = [Y.lo, Y.hi]
    L1 = max(abs(2 * x + y) for x in xs for y in ys)
    L2 = max(abs(x + 2 * y - 2) for x in xs for y in ys)
    return L1, L2

# f1
def f1(x):
    return x ** 3 - 3 * x ** 2 + 2

def f1_der(x):
    return 3 * x ** 2 - 6 * x

def f1_der_int(X):
    return 3 * X ** 2 - 6 * X

def f1_nat(X):
    return X ** 3 - 3 * (X ** 2) + Interval(2, 2)

def f1_horner_int(X):
    return X * (X * (X - 3)) + Interval(2, 2)

def f1_mv(X, c, D):
    """D — интервал производной (точный или естественный)."""
    return f1(c) + D * (X - c)

def f1_slope_int(X, c):
    return X ** 2 + (c - 3) * X + (c * c - 3 * c)

def f1_sl(X, c):
    return f1(c) + f1_slope_int(X, c) * (X - c)


# f2
def f2(x):
    return (x - 1) ** 3 + math.sin(2 * x)

def f2_der(x):
    return 3 * (x - 1) ** 2 + 2 * math.cos(2 * x)

def f2_der_int(X):
    return 3 * (X - 1) ** 2 + 2 * cos_int(2 * X)

def f2_nat(X):
    return (X - 1) ** 3 + sin_int(2 * X)

def f2_equiv_int(X):
    return X ** 3 - 3 * X ** 2 + 3 * X - 1 + sin_int(2 * X)

def f2_mv(X, c, D):
    return f2(c) + D * (X - c)

def f2_slope_int(X, c):
    term1 = (X - 1) ** 2
    term2 = (X - 1) * (c - 1)
    term3 = (c - 1) ** 2
    S = X - c
    sincS = sinc_int(S)
    cosxc = cos_int(X + c)
    return term1 + term2 + term3 + 2 * cosxc * sincS

def f2_sl(X, c):
    return f2(c) + f2_slope_int(X, c) * (X - c)


# f3
def f3(x, y):
    return x ** 2 + x * y + y ** 2 - 2 * y

def f3_dx_int(X, Y):
    return 2 * X + Y

def f3_dy_int(X, Y):
    return X + 2 * Y - 2

def f3_nat(X, Y):
    return X ** 2 + X * Y + Y ** 2 - 2 * Y

def f3_equiv_int(X, Y):
    return (X + Y / 2) ** 2 + Interval(0.75, 0.75) * Y ** 2 - 2 * Y

def f3_mv(X, Y, cx, cy, Dx, Dy):
    return f3(cx, cy) + Dx * (X - cx) + Dy * (Y - cy)


# ============================================================
# Точные области значений производных
# ============================================================
def exact_der_f1(X):
    lo, hi = X.lo, X.hi
    a, b = _exact_scalar_range(f1_der, lo, hi)
    return Interval(a, b)

def exact_der_f2(X):
    lo, hi = X.lo, X.hi
    a, b = _exact_scalar_range(f2_der, lo, hi)
    return Interval(a, b)

def exact_der_f3_x(X, Y):
    """Точная область d f3 / dx = 2x + y на брусе X × Y."""
    # 2x + y, минимум = 2*X.lo + Y.lo, максимум = 2*X.hi + Y.hi
    return Interval(2 * X.lo + Y.lo, 2 * X.hi + Y.hi)

def exact_der_f3_y(X, Y):
    """Точная область d f3 / dy = x + 2y - 2 на брусе X × Y."""
    return Interval(X.lo + 2 * Y.lo - 2, X.hi + 2 * Y.hi - 2)


# ============================================================
# Точные области значений самих функций
# ============================================================
def exact_f1(X):
    return _exact_scalar_range(f1, X.lo, X.hi)

def exact_f2(X):
    return _exact_scalar_range(f2, X.lo, X.hi)

def exact_f3(X, Y):
    def f3_scalar(v):
        x, y = v
        return f3(x, y)
    bounds = [(X.lo, X.hi), (Y.lo, Y.hi)]
    res_min = differential_evolution(f3_scalar, bounds=bounds,
                                     seed=0, tol=1e-12, polish=True)
    res_max = differential_evolution(lambda v: -f3_scalar(v), bounds=bounds,
                                     seed=0, tol=1e-12, polish=True)
    return (res_min.fun, -res_max.fun)


# ============================================================
# Вспомогательные
# ============================================================
def dist_interval(I, exact):
    return max(abs(I.lo - exact[0]), abs(I.hi - exact[1]))

def baumann_centers_1d(D, X):
    """Центры Бауманна по точному интервалу производной D."""
    mid = D.mid()
    rad = D.rad()
    if rad == 0:
        p = 0.0
    else:
        p = max(-1.0, min(1.0, mid / rad))
    c_star = X.mid() - p * X.rad()
    c_sup = X.mid() + p * X.rad()
    return c_star, c_sup

def f1_bic(X, D):
    c1, c2 = baumann_centers_1d(D, X)
    I1 = f1_mv(X, c1, D)
    I2 = f1_mv(X, c2, D)
    return I1.intersect(I2)

def f2_bic(X, D):
    c1, c2 = baumann_centers_1d(D, X)
    I1 = f2_mv(X, c1, D)
    I2 = f2_mv(X, c2, D)
    return I1.intersect(I2)

def f3_bic(X, Y, Dx, Dy):
    midx, radx = X.mid(), X.rad()
    midy, rady = Y.mid(), Y.rad()
    px = max(-1.0, min(1.0, Dx.mid() / Dx.rad())) if Dx.rad() != 0 else 0.0
    py = max(-1.0, min(1.0, Dy.mid() / Dy.rad())) if Dy.rad() != 0 else 0.0
    cx1, cx2 = midx - px * radx, midx + px * radx
    cy1, cy2 = midy - py * rady, midy + py * rady
    I1 = f3_mv(X, Y, cx1, cy1, Dx, Dy)
    I2 = f3_mv(X, Y, cx2, cy2, Dx, Dy)
    return I1.intersect(I2)

def fmt(I):
    if I is None:
        return "пусто"
    return f"[{I.lo:.2f}, {I.hi:.2f}]"


# ============================================================
# Генерация таблиц
# ============================================================
def write_table_file(filename, title, exact, rows):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\begin{tabular}{|l|c|c|}\n\\hline\n")
        f.write("Метод & Оценка & dist \\\\\n\\hline\n")
        f.write(f"Точная область & [{exact[0]:.2f}, {exact[1]:.2f}] & 0 \\\\\n")
        for name, I in rows:
            d = dist_interval(I, exact)
            f.write(f"{name} & {fmt(I)} & {d:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write(f"\\caption{{{title}}}\n\\end{{table}}\n")


def main():
    # ---------- f1 ----------
    X1 = Interval(0, 3)
    exact1 = exact_f1(X1)
    D1 = exact_der_f1(X1)            # точная область производной [-3, 9]
    rows1 = []
    rows1.append(("Естественное $f_{\\natural}$", f1_nat(X1)))
    rows1.append(("Горнер", f1_horner_int(X1)))
    for c in [0.0, 1.5, 2.0, 3.0]:
        rows1.append((f"Дифф. центр. $c={c}$", f1_mv(X1, c, D1)))
    for c in [0.0, 1.5, 2.0, 3.0]:
        rows1.append((f"Наклонная $c={c}$", f1_sl(X1, c)))
    rows1.append(("Бицентрированная", f1_bic(X1, D1)))
    write_table_file("lab2/table_f1.tex",
                     "Функция $f_1(x)=x^3-3x^2+2$, $X=[0,3]$",
                     exact1, rows1)

    c1_star, c1_sup = baumann_centers_1d(D1, X1)
    print(f"f1: c_*= {c1_star:.2f}, c^* = {c1_sup:.2f}")

    # ---------- f2 ----------
    X2 = Interval(0, 2)
    exact2 = exact_f2(X2)
    D2 = exact_der_f2(X2)            # точная область производной
    rows2 = []
    rows2.append(("Естественное $f_{\\natural}$", f2_nat(X2)))
    rows2.append(("Эквивалентное", f2_equiv_int(X2)))
    for c in [0.0, 1.0, 2.0]:
        rows2.append((f"Дифф. центр. $c={c}$", f2_mv(X2, c, D2)))
    for c in [0.0, 1.0, 2.0]:
        rows2.append((f"Наклонная $c={c}$", f2_sl(X2, c)))
    rows2.append(("Бицентрированная", f2_bic(X2, D2)))
    write_table_file("lab2/table_f2.tex",
                     "Функция $f_2(x)=(x-1)^3+\\sin(2x)$, $X=[0,2]$",
                     exact2, rows2)

    c2_star, c2_sup = baumann_centers_1d(D2, X2)
    print(f"f2: c_*= {c2_star:.2f}, c^* = {c2_sup:.2f}")

    # ---------- f3 ----------
    X = Interval(-2, 1)
    Y = Interval(-1, 2)
    exact3 = exact_f3(X, Y)
    Dx = exact_der_f3_x(X, Y)        # точная область df3/dx = 2x+y
    Dy = exact_der_f3_y(X, Y)        # точная область df3/dy = x+2y-2
    rows3 = []
    rows3.append(("Естественное $f_{\\natural}$", f3_nat(X, Y)))
    rows3.append(("Эквивалентное", f3_equiv_int(X, Y)))
    c1 = (-0.5, 0.5)
    c2 = (0.0, 0.0)
    for (cx, cy), name in [(c1, "mid"), (c2, "(0,0)")]:
        rows3.append((f"Дифф. центр. {name}",
                      f3_mv(X, Y, cx, cy, Dx, Dy)))
    rows3.append(("Бицентрированная", f3_bic(X, Y, Dx, Dy)))

    # E.7: оболочки после дробления
    Xa = Interval(-2, -0.5)
    Xb = Interval(-0.5, 1)
    Ix = interval_hull(f3_nat(Xa, Y), f3_nat(Xb, Y))
    rows3.append(("Оболочка после дробления по $x$", Ix))

    Ya = Interval(-1, 0.5)
    Yb = Interval(0.5, 2)
    Iy = interval_hull(f3_nat(X, Ya), f3_nat(X, Yb))
    rows3.append(("Оболочка после дробления по $y$", Iy))

    # E.8: липшицевская оценка
    L1, L2 = lipschitz_f3(X, Y)
    F_nat = f3_nat(X, Y)
    lhs = F_nat.rad()
    rhs = L1 * X.rad() + L2 * Y.rad()
    print(f"E.8: rad F_nat = {lhs:.2f}, L1*radX + L2*radY = {rhs:.2f}")
    print(f"     {lhs:.2f} <= {rhs:.2f}: {lhs <= rhs}")
    print(f"     L1 = {L1}, L2 = {L2}")

    write_table_file("lab2/table_f3.tex",
                     "Функция $f_3(x,y)=x^2+xy+y^2-2y$, $X=[-2,1]\\times[-1,2]$",
                     exact3, rows3)

    px = max(-1.0, min(1.0, Dx.mid() / Dx.rad())) if Dx.rad() != 0 else 0.0
    py = max(-1.0, min(1.0, Dy.mid() / Dy.rad())) if Dy.rad() != 0 else 0.0
    cx_star = X.mid() - px*X.rad()
    cx_sup = X.mid() + px*X.rad()
    cy_star = Y.mid() - py*Y.rad()
    cy_sup = Y.mid() + py*Y.rad()
    print(f"f3: c_*= ({cx_star:.2f}, {cy_star:.2f}), c^* = ({cx_sup:.2f}, {cy_sup:.2f})")

    print("Готово: table_f1.tex, table_f2.tex, table_f3.tex")

if __name__ == "__main__":
    main()