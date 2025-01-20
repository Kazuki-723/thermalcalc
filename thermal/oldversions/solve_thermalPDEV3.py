import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

# 定数の設定
L = 0.1  # サイズ (100mm = 0.1m)
W = 0.01
Nx = 1000  # x方向の分割数
Ny = 100  # y方向の分割数
dx = L / (Nx - 1)
dy = W / (Ny - 1)
alpha = 97e-6  # アルミニウムの熱拡散率 (m^2/s)
dt = 1e-4  # 時間ステップ (s)
time_steps = 1000  # 時間ステップ数

# 初期条件の設定
T = np.ones((Ny, Nx)) * 300  # 全域300K

# 境界条件の設定
T[:, 0] = 300  # 左側の辺
T[:, -1] = 300  # 右側の辺
T[0, :] = 300  # 上側の辺
T[-1, :] = 1000  # 下側の辺

# 時間発展の計算
for _ in range(time_steps):
    T_new = T.copy()
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )
    T = T_new

end = time.time()
print(end-start)

# 結果のプロット
plt.imshow(T, cmap='hot', origin='lower', extent=[0, L, 0, W])
plt.colorbar(label='Temperature (K)')
plt.title('Heat Diffusion in 2D Plane')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()