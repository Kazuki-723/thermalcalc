import ezdxf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import csv
import random
from tqdm import tqdm

#全部別の多角形として判定をかけるので、PMMA層とGF層の番号を調べる
#makingmesh.pyにその番号を入れると計算してくれる(ようにする)

#2号機メモ
#GF7,8
#PMMA13,14
#グラファイト9,10
#injector23,24

# DXFファイルのパス
dxf_file = '3Dmodel/flx.dxf'

# DXFファイルを読み込む
doc = ezdxf.readfile(dxf_file)
msp = doc.modelspace()

# メッシュの解像度（0.1mm）
resolution = 1

# メッシュのサイズを決定
min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
for entity in msp.query('LWPOLYLINE'):
    for vertex in entity:
        x, y = vertex[0], vertex[1]
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)

# メッシュの幅と高さ
width = int((max_x - min_x) / resolution) + 1
height = int((max_y - min_y) / resolution) + 1

# 境界条件として四方に50要素分の0を追加
boundary_padding = 50
width += 2 * boundary_padding
height += 2 * boundary_padding

# メッシュを初期化
mesh = np.zeros((height, width), dtype=float)

# 多角形の数をカウント
polygon_count = 0

# LWPOLYLINEエンティティを読み込み
for entity in tqdm(msp.query('LWPOLYLINE')):
    polyline_x = []
    polyline_y = []
    polygon_count += 1
    color_char = polygon_count
    for vertex in entity:
        x, y = vertex[0], vertex[1]
        polyline_x.append(x)
        polyline_y.append(y)
    polyline_x.append(polyline_x[0])  # 閉じた四角形にする
    polyline_y.append(polyline_y[0])  # 閉じた四角形にする

    # 各四角形を塗りつぶし
    polyline_coords = np.array([polyline_x, polyline_y]).T
    polyline_path = Path(polyline_coords)
    for x in range(boundary_padding, boundary_padding + width - 2 * boundary_padding):
        for y in range(boundary_padding, boundary_padding + height - 2 * boundary_padding):
            if polyline_path.contains_point((min_x + (x - boundary_padding) * resolution, min_y + (y - boundary_padding) * resolution)):
                mesh[y][x] = float(color_char)

# メッシュを転置する
mesh = mesh.T

# メッシュをCSVファイルとして出力
csv_file = 'mesh_output_with_boundary_labeled.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(mesh)

# 出力情報を表示
print(f"Total number of polygons: {polygon_count}")

plt.figure(figsize=(10, 8))
plt.imshow(mesh, cmap='tab10', origin='lower', extent=[min_y - boundary_padding * resolution, max_y + boundary_padding * resolution, min_x - boundary_padding * resolution, max_x + boundary_padding * resolution], vmin=1e-2)
plt.title('Mesh Visualization with Boundary Conditions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')  # 軸の縮尺を1:1に設定
plt.show()
