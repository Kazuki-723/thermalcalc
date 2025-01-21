import numpy as np
import matplotlib.pyplot as plt

def maxwell_boltzmann_2d(N, T, mass):
    k_B = 1.380649e-23  # ボルツマン定数 (m^2 kg s^-2 K^-1)
    sigma = np.sqrt(k_B * T / mass)
    
    # x方向とy方向の速度成分を生成
    vx = np.random.normal(0, sigma, N)
    vy = np.random.normal(0, sigma, N)
    
    return vx, vy

# パラメータの設定
N = 1000  # 粒子数
T = 300  # 温度 (K)
mass = 4.65e-26  # 質量 (kg) (例えば、水素分子)

# 速度成分を生成
vx, vy = maxwell_boltzmann_2d(N, T, mass)

# ヒストグラムを描画して確認
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(vx, bins=50, density=True, alpha=0.6, color='b')
plt.title("Velocity distribution in x direction")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Probability density")

plt.subplot(1, 2, 2)
plt.hist(vy, bins=50, density=True, alpha=0.6, color='r')
plt.title("Velocity distribution in y direction")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Probability density")

plt.tight_layout()
plt.show()
