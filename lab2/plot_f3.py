# plot_f3.py
import numpy as np
import matplotlib.pyplot as plt

# Функция варианта 5
def f3(x, y):
    return x**2 + x*y + y**2 - 2*y

# Брус
x = np.linspace(-2, 1, 200)
y = np.linspace(-1, 2, 200)
X, Y = np.meshgrid(x, y)
Z = f3(X, Y)

# --- Рисунок с двумя панелями ---
fig = plt.figure(figsize=(13, 5.5))

# Левая панель: поверхность
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.95)
ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$y$', fontsize=12)
ax1.set_zlabel('$z$', fontsize=12)
ax1.set_title('Поверхность $z = f_3(x,y)$', fontsize=13)
fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=15, pad=0.1)

# Правая панель: линии уровня
ax2 = fig.add_subplot(1, 2, 2)
levels = np.linspace(Z.min(), Z.max(), 15)
cs = ax2.contour(X, Y, Z, levels=levels, cmap='viridis')
ax2.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
cf = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.85)
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$y$', fontsize=12)
ax2.set_title('Линии уровня $f_3(x,y)$', fontsize=13)
ax2.set_aspect('equal')
fig.colorbar(cf, ax=ax2, shrink=0.9, aspect=15, pad=0.05)

plt.tight_layout()
plt.savefig('lab2/f3_graph.png', dpi=200, bbox_inches='tight')
plt.close()

print("Готово: f3_graph.png")