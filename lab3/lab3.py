"""Лабораторная работа 3, вариант 4. Запуск: python lab3/lab3.py.

Максимизация Tol сведена к ЛП по всем знакам, без сеточного поиска.
Коррекции сохраняют середины: R'=max(R-delta,0), r'=r+beta.
"""
from pathlib import Path
from itertools import product, combinations
import json
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
C = np.array([[.95, -1.], [1.05, -.96], [1.10, -1.06], [0., -.90]])
R = np.array([[.30, .30], [.30, .28], [.30, .30], [.30, .28]])
d = np.array([2.85, 2.95, 3., 1.90])
r = np.full(4, .05)
SIGNS = np.array(list(product([-1., 1.], repeat=2)))
RESERVE = .01
ROW_COLORS = ['#b85d12', '#126aa0', '#408235', '#854ca3']


def generators(x, c, radius, center, rhs_radius):
    x = np.asarray(x)
    return rhs_radius - np.abs(x @ c.T - center) - np.abs(x) @ radius.T


def halfspaces(c, radius, center, rhs_radius):
    h, k = [], []
    for ci, ri, di, bi in zip(c, radius, center, rhs_radius):
        for s in [-1., 1.]:
            for q in SIGNS:
                h.append(s * ci + ri * q)
                k.append(bi + s * di)
    return np.array(h), np.array(k)


def maximize(c, radius, center, rhs_radius):
    h, k = halfspaces(c, radius, center, rhs_radius)
    a = np.column_stack([h, np.ones(len(h))])
    obj = np.array([0., 0., -1.])
    sol = linprog(obj, A_ub=a, b_ub=k, bounds=[(None, None)] * 3, method='highs-ds')
    if not sol.success:
        raise RuntimeError(sol.message)
    x, t = sol.x[:2], sol.x[2]
    # Прямая подстановка и двойственный сертификат глобального максимума.
    assert abs(generators(x, c, radius, center, rhs_radius).min() - t) < 1e-8
    y = sol.ineqlin.marginals
    assert np.max(y) < 1e-8 and np.max(np.abs(a.T @ y - obj)) < 1e-8
    assert abs(k @ y - sol.fun) < 1e-8
    return x, float(t)


def corrected(m, delta, beta):
    return C[:m], np.maximum(R[:m] - delta, 0.), d[:m], r[:m] + beta


def threshold(m, kind, target=0.):
    def params(v):
        return (0., v) if kind == 'b' else ((v, 0.) if kind == 'A' else (v, v))
    lo, hi = 0., .3 if kind == 'A' else 1.
    if maximize(*corrected(m, *params(hi)))[1] < target - 1e-10:
        return None
    for _ in range(48):
        mid = (lo + hi) / 2
        if maximize(*corrected(m, *params(mid)))[1] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def polygon(args, x, t):
    h, k = halfspaces(*args)
    points = []
    for i, j in combinations(range(len(h)), 2):
        a = h[[i, j]]
        if abs(np.linalg.det(a)) < 1e-12:
            continue
        p = np.linalg.solve(a, k[[i, j]])
        if np.max(h @ p - k) <= 1e-8:
            points.append(p)
    if not points:
        return np.array([x]) if t >= -1e-9 else np.empty((0, 2))
    points = np.unique(np.round(points, 10), axis=0)
    if len(points) > 2:
        points = points[ConvexHull(points).vertices]
    assert generators(points, *args).min() >= -1e-8
    return points


def clipped_region(h, k, low, high):
    """Пересечение полуплоскостей с окном рисунка (отсечение многоугольника)."""
    p = np.array([[low[0], low[1]], [high[0], low[1]],
                  [high[0], high[1]], [low[0], high[1]]])
    for normal, bound in zip(h, k):
        clipped = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            fa, fb = normal @ a - bound, normal @ b - bound
            if fa <= 1e-12:
                clipped.append(a)
            if (fa < 0 < fb) or (fb < 0 < fa):
                clipped.append(a + fa / (fa - fb) * (b - a))
        p = np.array(clipped).reshape(-1, 2)
        if not len(p):
            break
    if len(p):
        assert np.max(p @ h.T - k) < 1e-8
    return p


def draw_generators(ax, args, low, high, shade=False):
    """Рисует точные кусочно-линейные границы t_i=0 без контурной сетки."""
    for i in range(len(args[0])):
        row_args = tuple(a[i:i+1] for a in args)
        h, k = halfspaces(*row_args)
        p = clipped_region(h, k, low, high)
        if len(p) < 2:
            continue
        if shade and len(p) >= 3:
            ax.fill(*p.T, color=ROW_COLORS[i], alpha=.04, lw=0)
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            # Не показываем искусственные ребра, образованные окном рисунка.
            if np.any((np.abs(h @ a-k) < 1e-8) & (np.abs(h @ b-k) < 1e-8)):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=ROW_COLORS[i],
                        lw=1.4, ls='--', zorder=3)


def generator_legend(m, emphasized=False):
    return [Line2D([], [], color=ROW_COLORS[i], ls='--', lw=1.5,
                   label=f'$t_{i+1}=0$') for i in range(m)] + [
        Patch(facecolor='#78c5ed' if emphasized else '#cee5f3',
              edgecolor='#132e45' if emphasized else '#263541',
              linewidth=2 if emphasized else 1, label='Общее пересечение'),
        Line2D([], [], marker='*', color='red', lw=0, markersize=9, label='$x^*$')]


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9})
    rng = np.random.default_rng(4)
    summary = []
    for m in [2, 3, 4]:
        rows = []
        for kind in ['original', 'b', 'A', 'Ab']:
            minimal = None if kind == 'original' else threshold(m, kind)
            v = None if kind == 'original' else threshold(m, kind, RESERVE)
            delta, beta = 0., 0.
            if kind == 'b': beta = v
            if kind == 'A': delta = .3 if v is None else v
            if kind == 'Ab': delta, beta = min(v, .3), v
            args = corrected(m, delta, beta)
            x, t = maximize(*args)
            p = polygon(args, x, t)
            h, k = halfspaces(*args)
            probes = rng.normal(size=(100, 2)) * 3
            # Проверка эквивалентности полуплоскостей исходному функционалу.
            np.testing.assert_allclose((k - probes @ h.T).min(axis=1),
                                       generators(probes, *args).min(axis=1), atol=1e-12)
            area = 0. if len(p) < 3 else float(ConvexHull(p).volume)
            rows.append(dict(kind=kind, minimum=minimal, delta=delta, beta=beta,
                             x=x.tolist(), T=t, generators=generators(x, *args).tolist(),
                             radii=args[1].tolist(), rhs_radius=args[3].tolist(),
                             vertices=p.tolist(), area=area))
        summary.append(dict(system=m-1, m=m, cases=rows))
        draw(m, rows)
    (OUT / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    draw_construction(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def draw(m, rows):
    fig = plt.figure(figsize=(14.5, 7.2), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .06, 1, .94))
    # Одинаковые пределы для всех способов одной системы.
    allpoints = np.vstack([np.array(row['vertices']).reshape(-1, 2) for row in rows
                           if row['vertices']] + [np.array([row['x'] for row in rows])])
    low, high = allpoints.min(axis=0), allpoints.max(axis=0)
    padding = np.maximum((high - low) * .15, .12)
    low, high = low - padding, high + padding
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 85), np.linspace(low[1], high[1], 85))
    grid = np.stack([xx, yy], axis=-1)
    names = ['Исходная', 'b-коррекция', 'A-коррекция', 'Ab-коррекция']
    for j, row in enumerate(rows):
        args = corrected(m, row['delta'], row['beta'])
        z = generators(grid, *args).min(axis=-1)
        ax = fig.add_subplot(2, 4, j+1, projection='3d')
        ax.plot_surface(xx, yy, z, cmap='viridis', linewidth=0, alpha=.9, rasterized=True)
        ax.scatter(*row['x'], row['T'], color='red', s=26, depthshade=False)
        ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='Tol', title=names[j])
        ax.set_title(names[j], fontsize=13)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(3, prune='both'))
            axis.label.set_fontsize(12)
        ax.view_init(elev=27, azim=-130)
        ax.tick_params(labelsize=10, pad=0)
        ax = fig.add_subplot(2, 4, j+5)
        draw_generators(ax, args, low, high, shade=True)
        p = np.array(row['vertices'])
        if len(p) >= 3:
            ax.fill(*p.T, color='#78c5ed', ec='#132e45', lw=2.8, zorder=5)
        elif len(p):
            ax.plot(*p.T, 'o-', color='#132e45', lw=2.8, zorder=5)
        else:
            ax.text(.5, .88, 'Множество пусто', transform=ax.transAxes, ha='center', fontsize=12, zorder=7)
        ax.plot(*row['x'], 'r*', ms=10, zorder=6)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='$x_1$', ylabel='$x_2$')
        ax.set_title(f"max Tol = {row['T']:.5f}", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.grid(alpha=.25)
    fig.legend(handles=generator_legend(m, emphasized=True), loc='lower center', ncol=m+2, fontsize=12)
    fig.savefig(OUT / f'system_{m-1}.png', dpi=210)
    plt.close(fig)


def draw_construction(summary):
    """Увеличенная иллюстрация пересечения ограничений после b-коррекции."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .12, 1, .88))
    for ax, system in zip(axes, summary):
        row = system['cases'][1]
        p = np.array(row['vertices'])
        low, high = p.min(axis=0), p.max(axis=0)
        pad = np.maximum((high-low)*.25, .06)
        low, high = low-pad, high+pad
        args = corrected(system['m'], row['delta'], row['beta'])
        draw_generators(ax, args, low, high, shade=True)
        ax.fill(*p.T, color='#78c5ed', ec='#132e45', lw=2.8, zorder=5)
        ax.plot(*row['x'], 'r*', ms=10, zorder=6)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
               xlabel='$x_1$', ylabel='$x_2$', title=f"Система {system['system']}")
        ax.tick_params(labelsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.20)
    fig.legend(handles=generator_legend(4, emphasized=True), loc='lower center', ncol=3, fontsize=11)
    fig.savefig(OUT / 'construction.png', dpi=230)
    plt.close(fig)


if __name__ == '__main__':
    main()
