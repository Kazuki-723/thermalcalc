import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from tqdm import tqdm
import csv
import sys
import time

start = time.time()
# 定数の設定
dx = 0.5e-3  # 1mm
dy = 0.5e-3  # 1mm
dt = 0.001  # 0.001秒

alpha_PMMA = 1.14e-7  # PMMAの温度拡散率 (計算値)
alpha_Al = 9.8e-5  # アルミニウムの温度拡散率 (計算値)
alpha_insulation = 2.1e-7  # GFRPの温度拡散率 (計算値)
alpha_graphite = 1.0e-4  # グラファイトの温度拡散率 (参考値)
max_alpha = 9.8e-5  # 上のうち最大値(大体アルミ)

# 安定条件の算出
dt_max = dx ** 2 / (2 * max_alpha)
if dt_max > dt:
    print("Go")
elif dt_max < dt:
    print("Nogo")
    sys.exit()
print("CFLvalue", dt_max)

# 空気の識別用定数
AIR_INSIDE = 5
AIR_OUTSIDE = 0

# CSVファイルからメッシュを読み込む
csv_file = 'thermal/mesh_output_with_internal_air.csv'
mesh = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        mesh.append([float(cell) for cell in row])
mesh = np.array(mesh)

# 材料分布の設定
materials = np.zeros_like(mesh)
materials[mesh == 1] = alpha_Al  # アルミニウム
materials[mesh == 2] = alpha_insulation  # GFRP
materials[mesh == 3] = alpha_graphite  # グラファイト
materials[mesh == 4] = alpha_PMMA  # PMMA
materials[mesh == 5] = AIR_INSIDE  # 内部の空気
materials[mesh == 0] = AIR_OUTSIDE  # 外部の空気

# 初期温度分布
T_air_inside = 3000  # K
T_air_outside = 288  # K
T_initial = 288  # K
temperature = np.full_like(mesh, T_initial)
temperature[mesh == 5] = T_air_inside
temperature[mesh == 0] = T_air_outside

# 計算時間の設定
total_time = 3.0  # 計算したい時間を入れる
num_steps = int(total_time / dt)

def heat_conduction(num_steps, temperature, materials, dt, dx, dy, T_initial, T_air_inside):
    for step in range(num_steps):
        temp_new = temperature.copy()
        for i in range(1, mesh.shape[0] - 1):
            for j in range(1, mesh.shape[1] - 1):
                if materials[i, j] not in [AIR_INSIDE, AIR_OUTSIDE]:
                    alpha = materials[i, j]
                    temp_new[i, j] = temperature[i, j] + alpha * dt * (
                            (temperature[i+1, j] - 2 * temperature[i, j] + temperature[i-1, j]) / dx**2 +
                            (temperature[i, j+1] - 2 * temperature[i, j] + temperature[i, j-1]) / dy**2)
                else:
                    continue
                if temp_new[i,j] < T_initial:
                    temp_new[i,j] = T_initial
                elif temp_new[i,j] > T_air_inside:
                    temp_new[i,j] = T_air_inside
        temperature = temp_new.copy()
    return temperature

# 高速化された伝熱計算
temperature = heat_conduction(num_steps, temperature, materials, dt, dx, dy, T_initial, T_air_inside)

# エッジ検出
edges = np.ones((*mesh.shape, 3)) * 255  # RGB画像用の配列
colors = {
    1: (0, 0, 1),  # アルミニウム（青）
    2: (0, 1, 0),  # GFRP（緑）
    3: (1, 0, 0),  # グラファイト（赤）
    4: (0.5, 0, 0.5)  # PMMA（紫）
}

for value, color in colors.items():
    mask = (mesh == value).astype(float)
    edge_x = sobel(mask, axis=0)
    edge_y = sobel(mask, axis=1)
    edge_magnitude = np.hypot(edge_x, edge_y)
    edge_indices = edge_magnitude > 0
    edges[edge_indices] = color

end = time.time()
print(end-start)
# 最終結果を画像として表示
plt.figure(figsize=(10, 8))
plt.imshow(temperature, cmap='hot', interpolation='nearest', vmin=300, vmax=3000)
#plt.imshow(edges, alpha=0.6)  # 縁取りを重ねて表示
plt.title('Final Temperature Distribution with Material Edges')
plt.xlabel('X-axis (mm)')
plt.ylabel('Y-axis (mm)')
plt.colorbar(label='Temperature (K)')
plt.grid(True)
plt.axis('equal')
plt.show()


