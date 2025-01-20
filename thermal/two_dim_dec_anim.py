import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm 
import sys

plt.rcParams["animation.ffmpeg_path"] = "C:\\ffmpeg/bin/ffmpeg.exe"
fig, ax = plt.subplots()

#諸元メモ(2号機)
#PMMA内径50mm 外径60mm
#アブレーター厚さ2.5mm
#アブレーター外径65mm
#モーターケース外径75mm
#燃焼時間2.47s 消費厚さ2.8mm

# 定数の設定
dx = 0.5e-3  # 0.5mm
dy = 0.5e-3  # 0.5mm
dt = 0.001  # 0.001秒

alpha_PMMA = 1.14e-7  # PMMAの温度拡散率 (計算値)
alpha_Al = 9.8e-5  # アルミニウムの温度拡散率 (計算値)
alpha_insulation = 2.1e-7  # GFRPの温度拡散率 (計算値)
max_alpha = 9.8e-5 #上のうち最大値(大体アルミ)

#安定条件の算出
dt_max = dx ** 2 /(2 * max_alpha)
if dt_max > dt:
    print("Go")
elif dt_max < dt:
    print("Nogo")
    sys.exit()
print("CFLvalue",dt_max)

# 空気の識別用定数
AIR_INSIDE = -1
AIR_OUTSIDE = -2

# シミュレーション領域の設定
xy_range = 40e-3  # ±40mm
x = np.arange(-xy_range, xy_range + dx, dx)
y = np.arange(-xy_range, xy_range + dy, dy)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
ims = []

# 材料分布の設定（直径を半径に変換）
PMMA_inner_radius = 25e-3  # 50mm の直径 -> 半径
PMMA_outer_radius = 30e-3  # 60mm の直径 -> 半径
Al_inner_radius = 32.5e-3  # 65mm の直径 -> 半径
Al_outer_radius = 37.5e-3  # 75mm の直径 -> 半径
insulation_thickness = 2.5e-3  # 5.0mm の直径 -> 半径

materials = np.zeros_like(R)
materials[(R >= PMMA_inner_radius) & (R <= PMMA_outer_radius)] = alpha_PMMA
materials[(R > PMMA_outer_radius) & (R <= PMMA_outer_radius + insulation_thickness)] = alpha_insulation
materials[(R > PMMA_outer_radius + insulation_thickness) & (R <= Al_outer_radius)] = alpha_Al

# 空気部の識別設定
materials[R < PMMA_inner_radius] = AIR_INSIDE
materials[R > Al_outer_radius] = AIR_OUTSIDE

# 初期温度分布
T_air_inside = 3000  # K
T_air_outside = 288  # K
T_initial = 288  # K
temperature = np.full_like(R, T_initial)
temperature[R < PMMA_inner_radius] = T_air_inside
temperature[R > Al_outer_radius] = T_air_outside

#計算時間の設定
time = 0.1 #計算したい時間を入れる

# PMMAの減少速度
PMMA_reduction_rate = 0.57e-3  # 1.14mm/sec(直径,二号機シミュより)

def update(frame):
    # データを更新
    temp_new = np.random.rand(100, 100) * 3000
    ax.clear()
    im = ax.imshow(temp_new, cmap='hot', interpolation='nearest', vmin=300, vmax=3000, animated=True)
    
    # 等高線を追加
    X, Y = np.meshgrid(np.arange(temp_new.shape[1]), np.arange(temp_new.shape[0]))
    contours = ax.contour(X, Y, temp_new, colors='white',linewidths=0.5)
    #ax.clabel(contours, inline=True, fontsize=8)
    return [im]

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=100, blit=True)

# アニメーションをmp4ファイルとして保存
ani.save('thermal/anim.mp4', writer='ffmpeg')

plt.show()
