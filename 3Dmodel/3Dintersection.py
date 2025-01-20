import trimesh
import numpy as np

# .objファイルを読み込む
mesh = trimesh.load_mesh('3Dmodel/2-assembly.obj')

# メッシュの頂点を取得
triangles = np.array(mesh.triangles)

# 全部のy座標が0以上の点をフィルタリング
filtered_triangles = triangles[np.all(triangles[:, :, 1] >= 0, axis=1)]

# vertices から y = 0 の点を抽出し、そのインデックスを保存
vertices = filtered_triangles.reshape(-1, 3)
y_zero_points = []

for i, vertex in enumerate(vertices):
    if np.abs(vertex[1]) < 1e-6:  # y 座標が 0 かどうかをチェック
        y_zero_points.append([vertex[0], vertex[1], vertex[2], i])

y_zero_points = np.array(y_zero_points)

faces = np.arange(len(vertices)).reshape(-1, 3)

# trimesh オブジェクトを作成
mesh_combined = trimesh.Trimesh(vertices=vertices, faces=faces)

# OBJ ファイルとして保存
#mesh_combined.export('filtered_triangles_with_cross_section.obj')

print("OBJ ファイルとして保存しました：filtered_triangles_with_cross_section.obj")

# メッシュを表示（表示するためのビューアが必要です）
mesh_combined.show()
