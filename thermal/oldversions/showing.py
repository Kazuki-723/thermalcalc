import numpy as np
import matplotlib.pyplot as plt
import csv

# CSVファイルの読み込み
data = []
path = 'thermal\\heat_diffusion.csv'
with open(path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([float(val) for val in row])

data = np.array(data)

# ヒートマップの描画
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature')
plt.title('Heat Diffusion')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()