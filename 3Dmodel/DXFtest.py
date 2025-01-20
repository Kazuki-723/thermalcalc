import ezdxf

doc = ezdxf.readfile(r"3Dmodel/enginedata.dxf")

#レイヤー名抽出
# for layer in doc.layers: 
#     print(layer.dxf.name)

msp = doc.modelspace()

layer_zero = msp.query('*[layer=="1"]')
for zeros in layer_zero:
    print(zeros)