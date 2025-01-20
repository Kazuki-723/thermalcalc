import ezdxf
import matplotlib.pyplot as plt
import random
import numpy as np

# DXFファイルのパス
dxf_file = '3Dmodel/engine2.dxf'

# DXFファイルを読み込む
doc = ezdxf.readfile(dxf_file)
msp = doc.modelspace()

# 色のリストを作成
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# データを格納するリスト
polylines = []

# LWPOLYLINEエンティティを読み込む
for entity in msp.query('LWPOLYLINE'):
    polyline_x = []
    polyline_y = []
    for vertex in entity:
        # 90度回転
        x, y = vertex[0], vertex[1]
        polyline_x.append(-y)
        polyline_y.append(x)
    polyline_x.append(polyline_x[0])  # 閉じた四角形にする
    polyline_y.append(polyline_y[0])  # 閉じた四角形にする
    polylines.append((polyline_x, polyline_y))

# matplotlibでプロット
plt.figure(figsize=(10, 8))
for polyline in polylines:
    color = random.choice(colors)
    plt.fill(polyline[0], polyline[1], color=color, alpha=0.5)
plt.title('DXF Data Visualization with 90° Rotation using ezdxf')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')  # 軸の縮尺を1:1に設定
plt.show()
