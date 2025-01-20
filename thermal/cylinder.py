#cf
#https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html
#https://yuruyuru-plantengineer.com/piping-tempreture-caluculation/

import numpy as np
import matplotlib.pyplot as plt

# 条件の設定
To = 300  # 外気温 (K)
Tin = 3000  # 流入時の温度 (K)
L_total = 0.4  # 管全体の長さ (m)
Fm = 0.1  # 質量流量 (kg/s)
Do = 0.05  # 管の外径 (m)
Di = 0.04  # 管の内径 (m)
Cp = 1005  # 空気の比熱 (J/kg*K)
k = 237  # アルミニウムの熱伝導率 (W/m*K)

#ここの係数は調査の必要あり
hi = 50  # 内側の熱伝達係数 (W/m²*K)
ho = 25  # 外側の熱伝達係数 (W/m²*K)

# 総括伝熱係数の計算
Uo = 1 / (1/hi + (Do - Di)/(k * (Do - Di)) + 1/ho)

# αの計算
alpha = Uo * np.pi * Do / (Cp * Fm)

# 距離と温度の関係を計算
L = np.linspace(0, L_total, 100)
T_L = To + (Tin - To) * np.exp(-alpha * L)

# 放出熱量の計算
Q_out = Cp * Fm * (Tin - T_L)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(L, Q_out, label='Heat Loss')
plt.xlabel('Distance from Inlet (m)')
plt.ylabel('Heat Loss (W)')
plt.title('Heat Loss vs Distance from Inlet')
plt.legend()
plt.grid(True)
plt.show()
