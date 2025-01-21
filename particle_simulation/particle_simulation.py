import numpy as np
import matplotlib.pyplot as plt
import csv
from numba import jit

# 初期化

# mesh size
width = 101
height = 101
N = width*height
X = np.ones(N) * 100.5
Y = np.ones(N) * 51.5
x = np.linspace(0, 100, width)
y = np.linspace(0, 100, height)
X_mesh, Y_mesh = np.meshgrid(x, y)

# パラメータの設定
N = 10000  # 粒子数
T = 2300  # 温度 (K)
M = 29 #分子量
initialvalue = 1000 #初期の粒子数
time_steps = 1000 #時間発展数
inlet_particle = 10000 #一回ごとに流入する粒子

#流入速度
u_x = -500
u_y = 0

# リスト

#位置ベクトルの成分リスト、大きさ
pos_x_list = np.zeros((width, height))
pos_y_list = np.zeros((width, height))
magnitude_list = np.zeros((width, height))

#内積リスト
dot_product_list = np.zeros((width, height))

#系の速度リスト
speed_sys_x_list = np.zeros((width, height))
speed_sys_y_list = np.zeros((width, height))

#拡散方向リスト 順に+x,+y,-x,-y
spread =  np.zeros((width, height, 4))

#粒子数リスト
particles = np.ones((width, height)) * initialvalue

#位置ベクトル計算
def calc_pos_vector(i, j):
    point_x = i + 0.5
    point_y = j + 0.5

    pos_vector_x = point_x - 100.5
    pos_vector_y = point_y - 51.5

    # normalized
    magnitude = np.sqrt(pos_vector_x ** 2 + pos_vector_y ** 2)
    if magnitude ==0:
        pass
    else:
        pos_vector_x /= magnitude
        pos_vector_y /= magnitude

    return pos_vector_x, pos_vector_y, magnitude

#内積計算
def calc_dotprod(pos_x_list,pos_y_list,u_x,u_y):
    u_x_normalized = u_x/np.sqrt(u_x ** 2 + u_y ** 2)
    u_y_normalized = u_y/np.sqrt(u_x ** 2 + u_y ** 2)
    for i in range(width):
        for j in range(height):
            dot_product_list[i,j] = pos_x_list[i,j] * u_x_normalized + pos_y_list[i,j] * u_y_normalized
    return dot_product_list

#系の速度計算
def clac_speed_sys(dot_product_list,pos_x_list,pos_y_list):
    for i in range(width):
        for j in range(height):
            if dot_product_list[i,j] !=0:
                speed_sys_x_list[i,j] =  1000 * pos_x_list[i,j] / dot_product_list[i,j]
                speed_sys_y_list[i,j] =  1000 * pos_y_list[i,j] / dot_product_list[i,j]
            else:
                speed_sys_x_list[i,j] = u_y
                speed_sys_y_list[i,j] = u_x
    return speed_sys_x_list, speed_sys_y_list

#拡散シミュレーション
@jit(nopython=True)
def maxwell_boltzmann_2d(N, T, M, speed_sys_x, speed_sys_y):
    k_B = 1.380649e-23  # ボルツマン定数 (m^2 kg s^-2 K^-1)
    mass = M * 1.66053906660e-27
    sigma = np.sqrt(k_B * T / mass)
    
    # x方向とy方向の速度成分を生成
    vx = np.random.normal(0, sigma, N) + speed_sys_x
    vy = np.random.normal(0, sigma, N) + speed_sys_y
    
    counts = np.zeros(4)
    
    for x, y in zip(vx, vy):
        angle = np.arctan2(y, x) * 180 / np.pi
        if -45 <= angle < 45:
            counts[0] += 1
        elif 45 <= angle < 135:
            counts[1] += 1
        elif 135 <= angle <= 180 or -180 <= angle < -135:
            counts[2] += 1
        elif -135 <= angle < -45:
            counts[3] += 1

    return counts

#シミュ本番
def mesh_simulation(spread, particles):
    particles_new = particles.copy()
    #時間発展作業
    for i in range(1,width-1):
        for j in range(1,height-1):
            if i != width and j != height:
                particles_new[i,j] = particles[i-1,j] * (spread[i-1,j,3]/N) +\
                    particles[i+1,j] * (spread[i+1,j,1]/N) +\
                    particles[i,j-1] * (spread[i,j-1,0]/N) +\
                          particles[i,j+1] * (spread[i,j-1,2]/N)
    
    return particles_new

# 位置ベクトルリストの作成
for i in range(width):
    for j in range(height):
        pos_vector_x, pos_vector_y, magnitude = calc_pos_vector(i, j)
        pos_x_list[i, j] = pos_vector_x
        pos_y_list[i, j] = pos_vector_y
        magnitude_list[i, j] = magnitude

#内積計算
dot_product_list =  calc_dotprod(pos_x_list,pos_y_list,u_x,u_y)

#系の速度計算
speed_sys_x_list, speed_sys_y_list = clac_speed_sys(dot_product_list,pos_x_list,pos_y_list)

#拡散シミュ計算
for i in range(width):
    for j in range(height):
        spread[i,j] = maxwell_boltzmann_2d(N, T, M, speed_sys_x_list[i,j], speed_sys_y_list[i,j])
    print(i)

#拡散分布の出力
# csvファイルのパス 
csv_file_path = 'maxwell_boltzmann.csv' 

# CSVファイルに出力する 
with open(csv_file_path, mode='w', newline='') as file: 
    writer = csv.writer(file) 
    # 各要素を行に追加 
    for a in range(i): 
        for b in range(j): 
            writer.writerow(spread[a][b])

#mesh拡散のシミュレーション
for i in range(time_steps):
    particles = mesh_simulation(spread, particles)

    #右端に入力
    for j in range(40,60,1):
        particles[100,j] = inlet_particle
    print(i)

particles = particles.T

#結果のプロット

fig, ax = plt.subplots()
cax = ax.imshow(particles, cmap='plasma', interpolation='nearest')
plt.title(f'Step {i * 50} Particles Distribution')
plt.xlabel('X-axis (mm)')
plt.ylabel('Y-axis (mm)')
plt.grid(True)
fig.colorbar(cax, ax=ax, label='particles', orientation='vertical')
plt.show()

# 矢印をプロット
plt.figure(figsize=(10, 10))
plt.quiver(Y_mesh, X_mesh, speed_sys_x_list, speed_sys_y_list, angles='xy', scale_units='xy', scale=1)
plt.title("system_speed_list")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0,101)
plt.ylim(0,101)
plt.grid()
plt.show()

