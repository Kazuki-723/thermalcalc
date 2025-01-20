import numpy as np
import matplotlib.pyplot as plt

# 定数の設定
nx, ny = 100, 41  # 配管の長さ方向に100分割、直径方向に41分割
dx, dy = 0.001, 0.001  # メッシュサイズ
nt = 1000  # タイムステップ数
dt = 0.01  # タイムステップ幅
rho_air = 1.225  # 空気の密度（kg/m^3）
nu_air = 1.5e-5  # 空気の動粘性係数（m^2/s）
alpha_aluminum = 9.7e-5  # アルミニウムの熱拡散率（m^2/s）

# 初期条件の設定
T = np.ones((ny, nx)) * 300  # アルミニウムの初期温度
T_air = 3000  # 空気の温度

# 境界条件の設定
T[:, 0] = T_air  # 空気が供給される側の温度

# クランク・ニコルソン法の解法
def solve(T, nt, dt, dx, dy, alpha_aluminum, T_air):
    for n in range(nt):
        Tn = T.copy()
        
        # 温度の更新（クランク・ニコルソン法）
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T[i, j] = Tn[i, j] + 0.5 * alpha_aluminum * dt / dx**2 * (
                    (Tn[i+1, j] - 2*Tn[i, j] + Tn[i-1, j]) +
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j])
                ) + 0.5 * alpha_aluminum * dt / dy**2 * (
                    (Tn[i, j+1] - 2*Tn[i, j] + Tn[i, j-1]) +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1])
                )

        # 境界条件の更新
        T[:, 0] = T_air  # 空気が供給される側の温度
        T[:, -1] = T[:, -2]  # 反対側の境界条件（絶熱）
        T[0, :] = T[1, :]  # 上側の境界条件（絶熱）
        T[-1, :] = T[-2, :]  # 下側の境界条件（絶熱）

    return T

# シミュレーションの実行
T = solve(T, nt, dt, dx, dy, alpha_aluminum, T_air)

# 結果のプロット
plt.contourf(T, cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.title('Temperature Distribution in Aluminum Cylinder (2D Slice)')
plt.show()
