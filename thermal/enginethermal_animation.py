import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import csv
import sys
from numba import jit
import time
import os

plt.rcParams["animation.ffmpeg_path"] = "C:\\ffmpeg/bin/ffmpeg.exe"
start = time.time()
# 定数の設定
dx = 0.5e-3  # makingmesh.pyのmeshの大きさを入れる
dy = 0.5e-3  # 正方形meshしか切れないので、dx = dy
dt = 0.0002  # 秒

alpha_PMMA = 1.14e-7  # PMMAの温度拡散率 (計算値)
alpha_Al = 9.8e-5  # アルミニウムの温度拡散率 (計算値)
alpha_insulation = 2.1e-7  # GFRPの温度拡散率 (計算値)
alpha_graphite = 1.0e-7  # グラファイトの温度拡散率 (参考値) 拡散防止に1/1000にしてます
alpha_air = 2.2e-5 # 空気の温度拡散率 u = 0
alpha_air_inside =25e-5 #空気の温度拡散率 u = 1000[m/s]
max_alpha = max(alpha_PMMA,alpha_Al,alpha_insulation,alpha_graphite,alpha_air,alpha_air_inside)
print(max_alpha)

# 安定条件の算出
courant = dt / (dx ** 2 /(2 * max_alpha))
if courant < 0.5:
    print("Go")
    print("courant value",courant)
elif courant > 0.5:
    print("Nogo")
    print("courant value",courant)
    print("minimize dt or bigger dx,dy")
    sys.exit()

# CSVファイルからメッシュを読み込む
csv_file = 'thermal/mesh_output_with_internal_air.csv'
mesh = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        mesh.append([float(cell) for cell in row])
mesh = np.array(mesh)

# 材料分布の設定
#7がinjectorの空気、8がノズルの空気
materials = np.zeros_like(mesh)
materials[mesh == 1] = alpha_Al  # アルミニウム
materials[mesh == 2] = alpha_insulation  # GFRP
materials[mesh == 3] = alpha_graphite  # グラファイト
materials[mesh == 4] = alpha_PMMA  # PMMA
materials[mesh == 5] = 0 #injectorを一回伝熱しないものとする
materials[mesh == 6] = alpha_air  # 燃焼室の空気
materials[mesh == 7] = alpha_air #injectorの空気
materials[mesh == 8] = alpha_air_inside #ノズルの空気
materials[mesh == 0] = alpha_air_inside  # 外部の空気

# 初期温度分布
T_air_injector = 3000  # K
T_air_outside = 288  # K
T_initial = 288  # K
T_nozzle = 2500 # K
T_exit = 1500 # K
temperature = np.full_like(mesh, T_initial)
temperature[mesh == 6] = T_air_injector
temperature[mesh == 7] = T_air_injector
temperature[mesh == 8] = T_nozzle
temperature[mesh == 0] = T_air_outside

#燃焼室内を線形分布
combuston_index = np.where(mesh == 6)
nozzle_index = min(combuston_index[1])
injector_index = max(combuston_index[1])

for i in range(1,mesh.shape[0]-1):
    for j in range(nozzle_index,injector_index-1):
        if mesh[i,j] == 6:
            temperature[i,j] = T_nozzle + (T_air_injector - T_nozzle)*(j/(injector_index-nozzle_index-1))

#ノズル部も同じ処理をする
nozzle_index = np.where(mesh == 8)
exit_index = min(combuston_index[1])
nozzle_index = max(combuston_index[1])

for i in range(1,mesh.shape[0]-1):
    for j in range(nozzle_index,injector_index-1):
        if mesh[i,j] == 8:
            temperature[i,j] = T_exit + (T_nozzle - T_exit)*(j/(nozzle_index-exit_index-1))

# 計算時間の設定
total_time = 10  # 計算したい時間を入れる
num_steps = int(total_time / dt)

@jit(nopython=True)
def heat_conduction(num_steps, temperature, materials, dt, dx, dy, T_initial, T_air_injector):
    for step in range(num_steps):
        temp_new = temperature.copy()
        for i in range(1, mesh.shape[0] - 1):
            for j in range(1, mesh.shape[1] - 1):
                if mesh[i, j] not in [6, 7, 8]:
                    alpha = materials[i, j]
                    temp_new[i, j] = temperature[i, j] + alpha * dt * (
                            (temperature[i+1, j] - 2 * temperature[i, j] + temperature[i-1, j]) / dx**2 +
                            (temperature[i, j+1] - 2 * temperature[i, j] + temperature[i, j-1]) / dy**2)
                    temp_new[i, j] = round(temp_new[i,j],5)
                else:
                    continue

                #発散防止
                if temp_new[i, j] < T_initial:
                    temp_new[i, j] = T_initial
                elif temp_new[i, j] > T_air_injector:
                    temp_new[i, j] = T_air_injector
                else:
                    continue
        temperature = temp_new.copy()
    return temperature

# 高速化された伝熱計算
output_dir = 'thermal_output'
os.makedirs(output_dir, exist_ok=True)

# エッジ検出
edges = np.ones((*mesh.shape, 3)) * 255  # RGB画像用の配列
colors = {
    1: (0, 0, 1),  # アルミニウム（青）
    2: (0, 1, 0),  # GFRP（緑）
    3: (1, 0, 0),  # グラファイト（赤）
    4: (0.5, 0, 0.5),  # PMMA（紫）
    5: (0, 0, 1)   #injectorを別で処理する関係
}

for value, color in colors.items():
    mask = (mesh == value).astype(float)
    edge_x = sobel(mask, axis=0)
    edge_y = sobel(mask, axis=1)
    edge_magnitude = np.hypot(edge_x, edge_y)
    edge_indices = edge_magnitude > 0
    edges[edge_indices] = color

# Find bounds 
non_zero_indices = np.argwhere(temperature > 0) 
(top, left), (bottom, right) = non_zero_indices.min(0), non_zero_indices.max(0)

i = 0
for step in range(0, num_steps, 50):  # 50ステップごとに保存
    i = i + 1
    temperature = heat_conduction(50, temperature, materials, dt, dx, dy, T_initial, T_air_injector)
    
    fig, ax = plt.subplots()
    cax = ax.imshow(temperature[top:bottom+1, left:right+1], cmap='plasma', interpolation='nearest', vmin=300, vmax=3000)
    ax.imshow(edges[top:bottom+1, left:right+1], alpha=0.6) #縁取り要員
    plt.title(f'Step {i * 50} Temperature Distribution')
    plt.xlabel('X-axis (mm)')
    plt.ylabel('Y-axis (mm)')
    plt.grid(True)
    fig.colorbar(cax, ax=ax, label='Temperature (K)', orientation='vertical')
    plt.savefig(f'{output_dir}/frame_{i:04d}.png')
    plt.close(fig)

end = time.time()
print(f'Time elapsed: {end-start} seconds')

# ffmpegを使って画像を動画に変換
os.system(f'ffmpeg -framerate 100 -i {output_dir}/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p thermal_simulation.mp4')