import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm 
import sys

plt.rcParams["animation.ffmpeg_path"] = "C:\\ffmpeg/bin/ffmpeg.exe"
fig,ax = plt.subplots()

#諸元メモ(2号機)
#PMMA内径50mm 外径60mm
#アブレーター厚さ2.5mm
#アブレーター外径65mm
#モーターケース外径75mm
#燃焼時間2.47s 消費厚さ2.8mm

# 定数の設定
dx = 0.5e-3  # 0.1mm
dy = 0.5e-3  # 0.1mm
dt = 0.0005  # 0.001秒

alpha_PMMA = 1.14e-7  # PMMAの温度拡散率 (計算値)
alpha_Al = 9.8e-5  # アルミニウムの温度拡散率 (計算値)
alpha_insulation = 2.1e-7  # GFRPの温度拡散率 (計算値)
max_alpha = 9.8e-5 #上のうち最大値(大体アルミ)

#安定条件の算出
courant = dt / (dx ** 2 /(2 * max_alpha))
if courant < 1:
    print("Go")
    print("courant value",courant)
elif courant > 1:
    print("Nogo")
    print("courant value",courant)
    print("minimize dt or bigger dx,dy")
    sys.exit()

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
PMMA_inner_radius = 30e-3  # 50mm の直径 -> 半径
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
time = 2.47 #計算したい時間を入れる

# PMMAの減少速度
PMMA_reduction_rate = 0  # 1.14mm/sec(直径,二号機シミュより)

# 時間経過後の温度分布計算
def calculate_temperature(temp, materials, dt, dx, dy):
    temp_new = temp.copy()
    for i in range(1, temp.shape[0] - 1):
        for j in range(1, temp.shape[1] - 1):
            if materials[i, j] in (AIR_INSIDE, AIR_OUTSIDE):
                continue
            elif temp[i,j] == temp[i+1, j] and temp[i,j] == temp[i-1, j] and temp[i,j] == temp[i, j + 1] and temp[i,j] == temp[i, j-1]:
                temp_new[i,j] = temp[i,j]
            else:
                alpha = materials[i, j]
                temp_new[i, j] = temp[i, j] + alpha * dt * (
                    (temp[i+1, j] - 2*temp[i, j] + temp[i-1, j]) / dx**2 +
                    (temp[i, j+1] - 2*temp[i, j] + temp[i, j-1]) / dy**2
                )
                temp_new[i,j] = round(temp_new[i,j],5)
    #temp_initialを下回るのがいたらtemp_initialに上書き
    temp_new[temp_new < T_initial] = T_initial
    #発散も抑える
    temp_new[T_air_inside < temp_new] = T_air_inside

    #思いつかなかった(ax.contourが動かなかった)ので、コピーの境界部を無理やり温度変えていじる
    temp_show = temp_new.copy()

    #アブラレーター内径
    temp_show[(R >= PMMA_outer_radius) & (R <= PMMA_outer_radius + dx)] = 2000
    #アブラレーター外径
    temp_show[(R >= Al_inner_radius) & (R <= Al_inner_radius + dx)] = 2000
    #モーターケース外径
    temp_show[(R >= Al_outer_radius) & (R <= Al_outer_radius + dx)] = 2000
    im = ax.imshow(temp_show,cmap='coolwarm', interpolation='nearest',vmin=T_initial,vmax=T_air_inside,animated=True)
    ims.append([im])
    return temp_new

# PMMAの減少をシミュレート
def reduce_PMMA(materials, temperature, dt, reduction_rate, current_time):
    reduced_radius = PMMA_inner_radius + reduction_rate * dt * current_time
    materials[(R >= PMMA_inner_radius) & (R <= reduced_radius)] = AIR_INSIDE
    temperature[(R >= PMMA_inner_radius) & (R <= reduced_radius)] = T_air_inside

# 境界部分と断熱層の縁取りを行う処理
def plot_boundaries(ax, materials, extent):
    contour1 = ax.contour(X, Y, materials, colors='green', linewidths=0.5, extent=extent)
    
    # 断熱層の部分を縁取る
    insulation_mask = (materials == alpha_insulation)
    contour2 = ax.contour(X, Y, insulation_mask, colors='blue', linewidths=0.5, extent=extent)
    return contour1, contour2

# シミュレーションの実行
time_steps = int(time/dt)  # 時間ステップ数 
for t in tqdm(range(time_steps)):
    temperature = calculate_temperature(temperature, materials, dt, dx, dy)
    reduce_PMMA(materials, temperature, dt, PMMA_reduction_rate, t)

# 最終的なPMMAの内径を計算
final_PMMA_inner_radius = PMMA_inner_radius + PMMA_reduction_rate * dt * time_steps
final_PMMA_diameter = final_PMMA_inner_radius * 2 * 1e3  # mmに変換

# グラフの描画（別タブに表示）
ani = animation.ArtistAnimation(fig, ims, interval=(1000*dt)/10, blit=True,repeat_delay=1000)

ani.save('thermal/anim.mp4', writer="ffmpeg")
plt.show()

# # 時間経過後の温度分布
# fig3, ax3 = plt.subplots()
# ax3.set_title("thermal")
# cax3 = ax3.imshow(temperature, extent=[-xy_range, xy_range, -xy_range, xy_range], cmap='hot')
# fig3.colorbar(cax3, ax=ax3)
# _, _ = plot_boundaries(ax3, materials, [-xy_range, xy_range, -xy_range, xy_range])
# plt.savefig('thermal/thermalresult.png')
# plt.show()

# 最終的なPMMAの径を表示
print(f"PMMA_inret: {final_PMMA_diameter:.2f} mm")

#断面図を出す

# fig.clear()
ax.clear()
radius = temperature.shape[0]
L = np.arange(radius)
T = temperature[int(radius/2),:]

plt.plot(L,T)
plt.show()