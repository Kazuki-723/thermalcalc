import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import sys
import cv2


start = time.time()
# パラメータ設定
x,y = 100, 100 #物体の大きさ指定[mm]
#nx, ny = 500, 500  # グリッドのサイズ
dx, dy = 0.1, 0.1  # グリッド間隔
nx = int(x/dx)
ny = int(y/dy)
dt = 0.1  # 時間ステップ
nt = 1000  # 時間ステップ数

#物性値(これはアルミニウムの例)
k = 236 #熱伝導率
rho = 2700 #密度
Cp = 880 #比熱容量

# 初期条件
u = np.zeros((nx, ny))
alpha = k/(rho*Cp) * 10**6 #温度拡散率 [m^2 * 10^6/s = mm^2/s]
#alpha = 0.01 #温度拡散率を直接指定

#高温領域の作成
#いくつかサンプルを置いてます、配列の値をいじってるくらいです

#1.中央を加熱
#u[nx//4:3*nx//4, ny//4:3*ny//4] = 100.0 

#2.上下両端を加熱
u[0:(nx//100), :] = 100
u[(nx-(nx//100)):nx-1,:] = 100 

#3.左右両端を加熱
u[:,0:(ny//100)] = 100
u[:,(ny-(ny//100)):ny-1] = 100

# 疎行列の生成
A = sp.diags([-alpha*dt/dx**2, 1 + 2*alpha*dt/dx**2, -alpha*dt/dx**2], [-1, 0, 1], shape=(nx, nx)).tocsc()
B = sp.diags([-alpha*dt/dy**2, 1 + 2*alpha*dt/dy**2, -alpha*dt/dy**2], [-1, 0, 1], shape=(ny, ny)).tocsc()

# 時間発展
#加熱し続ける等で、特定の場所の温度が変わらない時はここで毎回疎行列を上書きする

for t in range(nt):
    u = spla.spsolve(A, u)
    u = spla.spsolve(B, u.T).T

    #加熱し続ける時の上書き
    u[0:(nx//100), :] = 100
    u[(nx-(nx//100)):nx-1,:] = 100 
    u[:,0:(ny//100)] = 100
    u[:,(ny-(ny//100)):ny-1] = 100

    #ファイル名用のやつ
    filename = 'thermal\\pictures\\fig'+str(t)+'.png'
    #描画
    plt.imshow(u, cmap='hot', interpolation='nearest',vmin=0,vmax=100)
    plt.colorbar(label='Temperature')
    plt.title('Heat Diffusion')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename)
    plt.close()

print("END calc")

# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter('video.mp4',fourcc,  10*1/dt, (640, 480))

if not video.isOpened():
    print("can't be opened")
    sys.exit()
    
for i in range(0, nt):
    img = cv2.imread('./thermal/pictures/fig'+str(i)+'.png')

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)

video.release()

end = time.time()
print(end-start)