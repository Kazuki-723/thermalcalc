import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
from tqdm import tqdm

plt.rcParams["animation.ffmpeg_path"] = "C:\\ffmpeg/bin/ffmpeg.exe"

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
N = 100000  # 拡散分布計算粒子数
T = 2300  # 温度 (K)
M = 29 #分子量
initialvalue = 100 #初期の粒子数
time_steps = 10000 #時間発展数
inlet_particle = 1000 #一回ごとに流入する粒子

#流入速度
u_x = 5
u_y = 0

# リスト

#位置ベクトルの成分リスト、大きさ
pos_x_list = np.zeros((width, height))
pos_y_list = np.zeros((width, height))
magnitude_list = np.zeros((width, height))
magnitude_inv_list = np.zeros((width, height))

#系の速度リスト
speed_sys_x_list = np.zeros((width, height))
speed_sys_y_list = np.zeros((width, height))
magnitude_sys_list = np.zeros((width, height))

#拡散方向リスト 順に+x,+y,-x,-y
spread =  np.zeros((width, height, 4))

#粒子数リスト
particles = np.ones((width, height)) * initialvalue

#位置ベクトル計算
def calc_pos_vector(i, j):
    if 40 < j < 60:
        pos_vector_x = i - 100
        pos_vector_y = 0
    elif j <= 40:
        point_x = i + 0.5
        point_y = j - 0.5

        pos_vector_x = point_x - 100.5
        pos_vector_y = point_y - 40.5
    elif j >= 60:
        point_x = i + 0.5
        point_y = j + 0.5

        pos_vector_x = point_x - 100.5
        pos_vector_y = point_y - 60.5        

    # normalized
    magnitude = np.sqrt(pos_vector_x ** 2 + pos_vector_y ** 2)
    if magnitude == 0:
        pass
    else:
        pos_vector_x /= magnitude
        pos_vector_y /= magnitude

    return pos_vector_x, pos_vector_y, magnitude

#系の速度計算
def clac_speed_sys(magnitude_list, pos_x_list,pos_y_list,speed_sys_x_list,speed_sys_y_list):
    for i in range(width):
        for j in range(height):
            speed_sys_x_list[i,j] =  pos_x_list[i,j] * magnitude_list[i,j] + u_x * (i - width)
            speed_sys_y_list[i,j] =  pos_y_list[i,j] * magnitude_list[i,j] + u_y
            magnitude_sys_list[i,j] = np.sqrt(speed_sys_x_list[i,j] ** 2 + speed_sys_y_list[i,j] ** 2)
            
    speed_sys_x_list = speed_sys_x_list / magnitude_sys_list
    speed_sys_y_list = speed_sys_y_list / magnitude_sys_list
    return speed_sys_x_list, speed_sys_y_list, magnitude_sys_list

#拡散シミュレーション
@jit(nopython=True)
def maxwell_boltzmann_2d(N, T, M, speed_sys_x, speed_sys_y, magnitude_sys):
    k_B = 1.380649e-23  # ボルツマン定数 (m^2 kg s^-2 K^-1)
    mass = M * 1.66053906660e-27
    sigma = np.sqrt(k_B * T / mass)
    
    # x方向とy方向の速度成分を生成
    vx = np.random.normal(0, sigma, N) + speed_sys_x * magnitude_sys
    vy = np.random.normal(0, sigma, N) + speed_sys_y * magnitude_sys
    
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
                particles_new[i,j] = particles[i-1,j] * (spread[i-1,j,0]/N) +\
                    particles[i+1,j] * (spread[i+1,j,2]/N) +\
                    particles[i,j-1] * (spread[i,j-1,3]/N) +\
                    particles[i,j+1] * (spread[i,j-1,1]/N)
                
                #発散防止
                if particles_new[i,j] < 1:
                    particles_new[i,j] = 1
                else:
                    pass
    
    return particles_new

# 位置ベクトルリストの作成
for i in range(width):
    for j in range(height):
        pos_vector_x, pos_vector_y, magnitude = calc_pos_vector(i, j)
        pos_x_list[i, j] = pos_vector_x
        pos_y_list[i, j] = pos_vector_y
        magnitude_list[i, j] = magnitude
        if magnitude !=0:
            magnitude_inv_list[i,j] = 1/magnitude
        else:
            magnitude_inv_list[i,j] = 0

#系の速度分布計算
speed_sys_x_list, speed_sys_y_list, magnitude_sys_list = clac_speed_sys(magnitude_list,pos_x_list,pos_y_list,speed_sys_x_list,speed_sys_y_list)

#天啓をおろし、素晴らしい関数を用いて速度の絶対値分布をいじる
magnitude_sys_list = 1000 / np.sqrt(np.log10(magnitude_sys_list))

#拡散シミュ計算
for i in range(width):
    for j in range(height):
        spread[i,j] = maxwell_boltzmann_2d(N, T, M, speed_sys_x_list[i,j], speed_sys_y_list[i,j], magnitude_sys_list[i,j])
    print(i)

#mesh拡散のシミュレーション
particles_history = []
for i in tqdm(range(time_steps)):
    particles = mesh_simulation(spread, particles)

    # 右端に入力
    for j in range(40, 60, 1):
        particles[100, j] += inlet_particle

    # アニメーション用に粒子分布を保存
    particles_history.append(particles.copy())

# アニメーションの作成
fig, ax = plt.subplots()
cax = ax.imshow(particles_history[0], cmap='plasma', interpolation='nearest') 


def animate(k):
    global cax
    ax.clear()
    cax = ax.imshow(particles_history[k], cmap='plasma', interpolation='nearest')
    plt.title(f'Step {k+1} Particles Distribution')
    plt.xlabel('X-axis (mesh)')
    plt.ylabel('Y-axis (mesh)')
    plt.grid(True)

ani = animation.FuncAnimation(fig, animate, frames=len(particles_history), interval=50, repeat=False)

# アニメーションを保存 
Writer = animation.FFMpegWriter 
writer = Writer(fps=1000, metadata=dict(artist='Me'), bitrate=1800)
ani.save("diffusion_simulation.mp4", writer=writer)

plt.show()
