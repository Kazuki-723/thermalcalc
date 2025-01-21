import numpy as np
import matplotlib.pyplot as plt

def maxwell_boltzmann_2d(N, T, M, speed_sys):
    k_B = 1.380649e-23  # ボルツマン定数 (m^2 kg s^-2 K^-1)
    mass = M * 1.66053906660e-27
    sigma = np.sqrt(k_B * T / mass)
    
    # x方向とy方向の速度成分を生成
    vx = np.random.normal(0, sigma, N) - speed_sys  # vxをシフト
    vy = np.random.normal(0, sigma, N)

    #simulationを考えて時間dt辺りの移動量を推定
    vx = vx * 0.0002
    vy = vy * 0.0002
    
    return vx, vy

def classify_particles(vx, vy):
    counts = {"+x": 0, "+y": 0, "-x": 0, "-y": 0}
    
    for x, y in zip(vx, vy):
        angle = np.arctan2(y, x) * 180 / np.pi
        if -45 <= angle < 45:
            counts["+x"] += 1
        elif 45 <= angle < 135:
            counts["+y"] += 1
        elif 135 <= angle <= 180 or -180 <= angle < -135:
            counts["-x"] += 1
        elif -135 <= angle < -45:
            counts["-y"] += 1

    return counts

# パラメータの設定
N = 1000  # 粒子数
T = 2300  # 温度 (K)
M = 29
speed_sys = 800 * 2.5 #系の速度[m/s](正の値を指定するとxが負の方向に進む)

# 速度成分を生成
vx, vy = maxwell_boltzmann_2d(N, T, M, speed_sys)

# 隣接メッシュ方向に向かう粒子数を分類
counts = classify_particles(vx, vy)

# 結果を表示
print("Particles moving towards adjacent meshes:")
for mesh, count in counts.items():
    print(f"{mesh}: {count} particles")

# 矢印をプロット
plt.figure(figsize=(10, 10))
plt.quiver(np.zeros(N), np.zeros(N), vx, vy, angles='xy', scale_units='xy', scale=1)
plt.title("Particle movement directions from origin")
plt.xlabel("Velocity in x direction (m/s)")
plt.ylabel("Velocity in y direction (m/s)")
plt.grid()
plt.xlim(-0.25,0.25)
plt.ylim(-0.25,0.25)
plt.show()