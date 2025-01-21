import numpy as np
import matplotlib.pyplot as plt

# 初期化

# mesh size
width = 101
height = 101
N = width*height
X = np.ones(N) * 100.5
Y = np.ones(N) * 51.5

# 位置ベクトルリスト
pos_x_list = np.zeros((width, height))
pos_y_list = np.zeros((width, height))

def calc_pos_vector(i, j):
    point_x = i + 0.5
    point_y = j + 0.5

    pos_vector_x = point_x - 100.5
    pos_vector_y = point_y - 51.5

    # # normalized
    # magnitude = np.sqrt(pos_vector_x ** 2 + pos_vector_y ** 2)
    # if magnitude ==0:
    #     pass
    # else:
    #     pos_vector_x /= magnitude
    #     pos_vector_y /= magnitude

    return pos_vector_x, pos_vector_y

# 位置ベクトルリストの作成
for i in range(width):
    for j in range(height):
        pos_vector_x, pos_vector_y = calc_pos_vector(i, j)
        pos_x_list[i, j] = pos_vector_x
        pos_y_list[i, j] = pos_vector_y

# 矢印をプロット
plt.figure(figsize=(10, 10))
plt.quiver(X, Y, pos_x_list, pos_y_list, angles='xy', scale_units='xy', scale=1)
plt.title("pos_vector_list")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0,101)
plt.ylim(0,101)
plt.grid()
plt.show()
