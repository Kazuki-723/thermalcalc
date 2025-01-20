import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 定数の設定
L = 0.1  # サイズ (100mm = 0.1m)
W = 0.1
Nx = 50  # x方向の分割数
Ny = 50  # y方向の分割数
dx = L / (Nx - 1)
dy = W / (Ny - 1)
alpha = 97e-6  # アルミニウムの熱拡散率 (m^2/s)
dt = 1e-4  # 時間ステップ (s)
time_steps = 500  # 時間ステップ数

# 初期条件の設定
T = np.ones((Ny, Nx)) * 300  # 全域300K

# 境界条件の設定
T[:, 0] = 300  # 左側の辺
T[:, -1] = 300  # 右側の辺
T[0, :] = 300  # 上側の辺
T[-1, :] = 1000  # 下側の辺

# 疎行列の設定
N = Nx * Ny
main_diag = np.ones(N) * (1 + 2 * alpha * dt / dx**2 + 2 * alpha * dt / dy**2)
off_diag = np.ones(N-1) * (-alpha * dt / dx**2)
off_diag_y = np.ones(N-Nx) * (-alpha * dt / dy**2)

diagonals = [main_diag, off_diag, off_diag, off_diag_y, off_diag_y]
A = diags(diagonals, [0, -1, 1, -Nx, Nx], format='csr')

# 時間発展の計算
for _ in range(time_steps):
    T_flat = T.flatten()
    T_flat = spsolve(A, T_flat)
    T = T_flat.reshape((Ny, Nx))

    # ノイマン境界条件の適用
    T[1:-1, 0] = T[1:-1, 1]  # 左側の辺
    T[1:-1, -1] = T[1:-1, -2]  # 右側の辺
    T[0, 1:-1] = T[1, 1:-1]  # 上側の辺
    T[-1, 1:-1] = 1000  # 下側の辺はディリクレ境界条件

# 結果のプロット
plt.imshow(T, cmap='hot', origin='lower', extent=[0, L, 0, W])
plt.colorbar(label='Temperature (K)')
plt.title('Heat Diffusion in 2D Plane (Neumann Boundary Condition)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()