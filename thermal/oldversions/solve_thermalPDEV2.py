import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

plt.rcParams["animation.ffmpeg_path"] = "C:\\ffmpeg/bin/ffmpeg.exe"

fig = plt.figure()

start = time.time()

# パラメータ設定
#境界条件の設定用に上下左右全方向1%以上のマージンを取ること
x,y = 102, 102 #物体の大きさ指定[mm]
#nx, ny = 500, 500  # グリッドのサイズ
dx, dy = 0.1, 0.1  # グリッド間隔
nx = int(x/dx)
ny = int(y/dy)
dt = 0.01  # 時間ステップ
nt = 1000  # 時間ステップ数

#物性値(これはアルミニウムの例)
k = 236 #熱伝導率
rho = 2700 #密度
Cp = 880 #比熱容量

# 初期条件
u = np.zeros((nx, ny))
alpha = k/(rho*Cp) * 10**6 #温度拡散率 [m^2 * 10^6/s = mm^2/s]
#alpha = 0.01 #温度拡散率を直接指定

#全体を300[K]へ
u[:,:] = 300

#高温領域の作成

#1.中央を加熱
#u[nx//4:3*nx//4, ny//4:3*ny//4] = 400.0 

#2.上下両端を加熱
u[0:(nx//100), :] = 300
u[(nx-(nx//100)):nx-1,:] = 300 

#3.左右両端を加熱
u[:,0:(ny//100)] = 300
u[:,(ny-(ny//100)):ny-1] = 1000

# 疎行列の生成
A = sp.diags([-alpha*dt/dx**2, 1 + 2*alpha*dt/dx**2, -alpha*dt/dx**2], [-1, 0, 1], shape=(nx, nx)).tocsc()
B = sp.diags([-alpha*dt/dy**2, 1 + 2*alpha*dt/dy**2, -alpha*dt/dy**2], [-1, 0, 1], shape=(ny, ny)).tocsc()
ims = []

# 時間発展
#加熱し続ける等で、特定の場所の温度が変わらない時はここで毎回疎行列を上書きする

for t in range(nt):
    # #加熱し続ける時の上書き
    u[:,(ny-(ny//100)):ny-1] = 1000

    # #端部の温度固定
    u[0:(nx//100), :] = 300
    u[(nx-(nx//100)):nx-1,:] = 300 
    u[:,0:(ny//100)] = 300

    u = spla.spsolve(A, u)

    #端部の再定義
    u[0:(nx//100), :] = 300
    u[(nx-(nx//100)):nx-1,:] = 300 
    u[:,0:(ny//100)] = 300

    u = spla.spsolve(B, u.T).T

    im = plt.imshow(u,cmap='hot', interpolation='nearest',vmin=0,vmax=1000,animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=(1000*dt)/10, blit=True,
                                repeat_delay=1000)

ani.save('thermal/anim.mp4', writer="ffmpeg")
plt.show()

end = time.time()
print(end-start)