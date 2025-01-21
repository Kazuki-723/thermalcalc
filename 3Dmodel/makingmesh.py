import ezdxf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import csv
import random
from tqdm import tqdm

# DXFファイルのパス
dxf_file = '3Dmodel/engine2.dxf'

# DXFファイルを読み込む
doc = ezdxf.readfile(dxf_file)
msp = doc.modelspace()

# メッシュの解像度[mm]
resolution = 0.5

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
    for x in range(boundary_padding, width - boundary_padding):
        for y in range(boundary_padding, height - boundary_padding):
            if polyline_path.contains_point((min_x + (x - boundary_padding) * resolution, min_y + (y - boundary_padding) * resolution)):
                mesh[y][x] = float(color_char)

# メッシュを転置する
mesh = mesh.T

# 素材ごとに番号を割り当てる()
for x in range(height):
    for y in range(width):
        if mesh[y][x] == 0:
            continue  # 空気はそのまま0
        elif mesh[y][x] in [7, 8]:
            mesh[y][x] = 2  # GFRP
        elif mesh[y][x] in [9, 10]:
            mesh[y][x] = 3  # グラファイト
        elif mesh[y][x] in [13, 14]:
            mesh[y][x] = 4  # PMMA
        elif mesh[y][x] in [23, 24]:
            mesh[y][x] = 5  # injector
        else:
            mesh[y][x] = 1  # アルミニウム

# 燃焼室内部の空気を取る
for col in range(height):
    inside_air = False
    encountered_graphite_pm = False
    for row in range(width):
        if mesh[row][col] in [4]:
            encountered_graphite_pm = True
        elif encountered_graphite_pm and mesh[row][col] == 0:
            inside_air = True
            mesh[row][col] = 6
            encountered_graphite_pm = False
        elif inside_air and mesh[row][col] == 0:
            mesh[row][col] = 6
        elif encountered_graphite_pm and mesh[row][col] not in [4]:
            inside_air = False

# injectorの空気も別にする
for col in range(height):
    injector_air = False
    encountered_injector = False
    for row in range(width):
        if mesh[row][col] in [5]:
            encountered_injector = True
        elif encountered_injector and mesh[row][col] == 0:
            injector_air = True
            mesh[row][col] = 7
            encountered_injector= False
        elif injector_air and mesh[row][col] == 0:
            mesh[row][col] = 7
        elif encountered_injector and mesh[row][col] not in [5]:
            iinjector_air = False

#ノズルの空気も別にする
for col in range(height):
    inside_air = False
    encountered_graphite_pm = False
    for row in range(width):
        if mesh[row][col] in [3]:
            encountered_graphite_pm = True
        elif encountered_graphite_pm and mesh[row][col] == 0:
            inside_air = True
            mesh[row][col] = 8
            encountered_graphite_pm = False
        elif inside_air and mesh[row][col] == 0:
            mesh[row][col] = 8
        elif encountered_graphite_pm and mesh[row][col] not in [3]:
            inside_air = False

# y < 0の部分が正しく処理されている前提で、y > 0の部分をy < 0の部分から対称性を用いて修正
mid_y = width // 2
for x in range(height):
    for y in range(mid_y):
        mesh[width - y - 1][x] = mesh[y][x]

#左側拡張
left_padding = 500
mesh_height = mesh.shape[0]
left = np.zeros((mesh_height,left_padding))
mesh = np.concatenate([left,mesh],1)

# メッシュをCSVファイルとして出力
csv_file = 'mesh_output_with_internal_air.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(mesh)

# 出力情報を表示
print(f"Total number of polygons: {polygon_count}")

plt.figure(figsize=(10, 8))
plt.imshow(mesh, cmap='tab10', origin='lower', extent=[min_y - boundary_padding * resolution, max_y + boundary_padding * resolution, min_x - boundary_padding * resolution, max_x + boundary_padding * resolution], vmin=1e-2)
plt.title('Mesh Visualization with Internal Air Detection')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')  # 軸の縮尺を1:1に設定
plt.show()
