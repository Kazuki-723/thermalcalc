import numpy as np
import trimesh

def create_cylinder(inner_diameter, inner_bottom, outer_diameter, outer_bottom, length):
    n = 100  # Number of vertices around the circumference
    theta = np.linspace(0, 2 * np.pi, n)
    
    # Outer cylinder vertices
    x_outer = (outer_diameter / 2) * np.cos(theta)
    y_outer = (outer_diameter / 2) * np.sin(theta)
    x_outer_bottom = (outer_bottom / 2) * np.cos(theta)
    y_outer_bottom = (outer_bottom / 2) * np.sin(theta)

    z_top = np.full(n, length)
    z_bottom = np.zeros(n)
    
    # Inner cylinder vertices
    x_inner = (inner_diameter / 2) * np.cos(theta)
    y_inner = (inner_diameter / 2) * np.sin(theta)
    x_inner_bottom = (inner_bottom / 2) * np.cos(theta)
    y_inner_bottom = (inner_bottom / 2) * np.sin(theta)
    
    vertices_outer_bottom = np.column_stack([x_outer_bottom, y_outer_bottom, z_bottom])
    vertices_outer_top = np.column_stack([x_outer, y_outer, z_top])
    vertices_inner_bottom = np.column_stack([x_inner_bottom, y_inner_bottom, z_bottom])
    vertices_inner_top = np.column_stack([x_inner, y_inner, z_top])

    vertices = np.vstack([vertices_outer_bottom, vertices_outer_top, vertices_inner_bottom, vertices_inner_top])
    
    faces = []
    
    #どの頂点を選ぶかを指定している
    #iはi番目のouter_bottom
    #n + iはi番目のouter_top
    #って感じに三点を指定して三角メッシュを生産する

    #outer,topはOK
    #inner bottom, bottom, top
    #bottom outer, outer, inner

    # Outer surface
    for i in range(n - 1):
        faces.append([i, i + 1, n + i])
        faces.append([i + 1, n + i + 1, n + i])
        
    # Top surface (outer-inner)
    for i in range(n - 1):
        faces.append([n + i, n + i + 1, 3 * n + i])
        faces.append([n + i + 1, 3 * n + i + 1, 3 * n + i])

    # Inner surface
    for i in range(n - 1):
        faces.append([3 * n + i, 3 * n + i + 1, 2 * n + i + 1])
        faces.append([2 * n + i,3 * n + i,  2 * n + i + 1])

    # Bottom surface (outer-inner)
    for i in range(n - 1):
        faces.append([2 * n + i, 2 * n + i + 1, i + 1])
        faces.append([i,2 * n + i, i + 1])
        
    faces = np.array(faces)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def save_obj(mesh, filename):
    mesh.export(filename, file_type='obj')

# 内径、外径、長さを指定
inner_diameter = 18
inner_bottom = 28
outer_diameter = 20
outer_bottom = 30
length = 50

cylinder_mesh = create_cylinder(inner_diameter, inner_bottom, outer_diameter,outer_bottom, length)
save_obj(cylinder_mesh, '3Dmodel/Cylinder.obj')

print('Cylinder_hollow.objファイルが作成されました。')
