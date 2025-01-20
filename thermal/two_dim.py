import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#諸元メモ(2号機)
#PMMA内径50mm 外径60mm
#アブレーター厚さ2.5mm

# 定数の設定
dx = 0.1e-3  # 0.1mm
dy = 0.1e-3  # 0.1mm
dt = 0.01  # 0.01秒

#熱拡散率[m^2/s]
alpha_PMMA = 1.14e-7  # PMMAの熱拡散率 (仮の値)
alpha_Al = 9.8e-5  # アルミニウムの熱拡散率 (仮の値)
alpha_insulation = 9.03e-7  # 石英ガラスの熱拡散率 (仮の値)

# 空気の識別用定数
AIR_INSIDE = -1
AIR_OUTSIDE = -2

# シミュレーション領域の設定
xy_range = 60e-3  # ±60mm
x = np.arange(-xy_range, xy_range + dx, dx)
y = np.arange(-xy_range, xy_range + dy, dy)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# 材料分布の設定（直径を半径に変換）
PMMA_inner_radius = 15e-3  # 30mm の直径 -> 半径
PMMA_outer_radius = 25e-3  # 50mm の直径 -> 半径
Al_inner_radius = 25.5e-3  # 51mm の直径 -> 半径
Al_outer_radius = 40e-3  # 80mm の直径 -> 半径
insulation_thickness = 0.5e-3  # 1mm の直径 -> 半径

materials = np.zeros_like(R)
materials[(R >= PMMA_inner_radius) & (R <= PMMA_outer_radius)] = alpha_PMMA
materials[(R > PMMA_outer_radius) & (R <= PMMA_outer_radius + insulation_thickness)] = alpha_insulation
materials[(R > PMMA_outer_radius + insulation_thickness) & (R <= Al_outer_radius)] = alpha_Al

# 空気部の識別設定
materials[R < PMMA_inner_radius] = AIR_INSIDE
materials[R > Al_outer_radius] = AIR_OUTSIDE

# 初期温度分布
T_air_inside = 3000  # K
T_air_outside = 300  # K
T_initial = 300  # K
temperature = np.full_like(R, T_initial)
temperature[R < PMMA_inner_radius] = T_air_inside
temperature[R > Al_outer_radius] = T_air_outside

# 時間経過後の温度分布計算
def calculate_temperature(temp, materials, dt, dx, dy):
    temp_new = temp.copy()
    for i in range(1, temp.shape[0] - 1):
        for j in range(1, temp.shape[1] - 1):
            if materials[i, j] in (AIR_INSIDE, AIR_OUTSIDE):
                continue
            alpha = materials[i, j]
            temp_new[i, j] = temp[i, j] + alpha * dt * (
                (temp[i+1, j] - 2*temp[i, j] + temp[i-1, j]) / dx**2 +
                (temp[i, j+1] - 2*temp[i, j] + temp[i, j-1]) / dy**2
            )
    return temp_new

# シミュレーションの実行
time_steps = 100  # 時間ステップ数
for _ in tqdm(range(time_steps)):
    temperature = calculate_temperature(temperature, materials, dt, dx, dy)

# 境界部分を目立つ色で縁取る処理
def plot_boundaries(ax, materials, extent): 
    boundaries = np.gradient(materials) 
    magnitude = np.sqrt(boundaries[0]**2 + boundaries[1]**2) 
    ax.contour(X, Y, materials, colors='green', linewidths=0.5, extent=extent) 
    
    # 断熱層の部分を縁取る
    insulation_mask = (materials == alpha_insulation) 
    ax.contour(X, Y, insulation_mask, colors='blue', linewidths=0.5, extent=extent)

# グラフの描画（別タブに表示）
# 材料の分布
fig1, ax1 = plt.subplots()
ax1.set_title("材料の分布")
cax1 = ax1.imshow(materials, extent=[-xy_range, xy_range, -xy_range, xy_range])
fig1.colorbar(cax1, ax=ax1)
plot_boundaries(ax1, materials, [-xy_range, xy_range, -xy_range, xy_range])
plt.show()

# 時間経過後の温度分布
fig3, ax3 = plt.subplots()
ax3.set_title("時間経過後の温度分布")
cax3 = ax3.imshow(temperature, extent=[-xy_range, xy_range, -xy_range, xy_range], cmap='hot')
fig3.colorbar(cax3, ax=ax3)
plot_boundaries(ax3, materials, [-xy_range, xy_range, -xy_range, xy_range])
plt.show()
