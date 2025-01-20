import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

plt.rcParams["animation.ffmpeg_path"] = "C:\\ffmpeg/bin/ffmpeg.exe"

# temp_newをサンプルデータとして作成
temp_new = np.random.rand(100, 100) * 3000

fig, ax = plt.subplots()

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
