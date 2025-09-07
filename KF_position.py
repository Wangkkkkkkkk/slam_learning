import numpy as np
import matplotlib.pyplot as plt

# ===== data =====
car_position = [
    [301.5, -401.46],
    [298.23, -375.44],
    [297.83, -346.15],
    [300.42, -320.2],
    [301.94, -300.08],
    [299.5, -274.12],
    [305.98, -253.45],
    [301.25, -226.4],
    [299.73, -200.65],
    [299.2, -171.62],
    [298.62, -152.11],
    [301.84, -125.19],
    [299.6, -93.4],
    [295.3, -74.79],
    [299.3, -49.12],
    [301.95, -28.73],
    [296.3, 2.99],
    [295.11, 25.65],
    [295.12, 49.86],
    [289.9, 72.87],
    [283.51, 96.34],
    [276.42, 120.4],
    [264.22, 144.69],
    [250.25, 168.06],
    [236.66, 184.99],
    [217.47, 205.11],
    [199.75, 221.82],
    [179.7, 238.3],
    [160, 253.02],
    [140.92, 267.19],
    [113.53, 270.71],
    [93.68, 285.86],
    [69.71, 288.48],
    [45.71, 292.9],
    [20.87, 298.77]
]

theta_time = 1.0
theta_a = 0.2
theta_xm = 3
theta_ym = 3

x_0_0 = np.array([0, 0, 0, 0, 0, 0])

F_matrix = np.array([
    [1, 1, 0.5, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0.5],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1]
])

Q_matrix = np.array([
    [0.25, 0.5, 0.5, 0, 0, 0],
    [0.5, 1, 1, 0, 0, 0],
    [0.5, 1, 1, 0, 0, 0],
    [0, 0, 0, 0.25, 0.5, 0.5],
    [0, 0, 0, 0.5, 1, 1],
    [0, 0, 0, 0.5, 1, 1]
])
Q_matrix = Q_matrix * pow(theta_a, 2)

R_n = np.array([
    [pow(theta_xm, 2), 0],
    [0, pow(theta_ym, 2)]
])

Pe_0_0 = np.array([
    [500, 0, 0, 0, 0, 0],
    [0, 500, 0, 0, 0, 0],
    [0, 0, 500, 0, 0, 0],
    [0, 0, 0, 500, 0, 0],
    [0, 0, 0, 0, 500, 0],
    [0, 0, 0, 0, 0, 500]
])

H_matrix = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0]
])

x_list = []
pe_list = []
kn_list = []

x_list.append(x_0_0)
pe_list.append(Pe_0_0)

for i in range(len(car_position)):
    print("========== n: ", i+1)
    z_n = np.array([car_position[i][0], car_position[i][1]])
    print("z_n: ", z_n)
    
    x_n_n_1 = F_matrix @ x_list[i]
    print("x_n_n_1: ", x_n_n_1)
    Pe_n_n_1 = F_matrix @ pe_list[i] @ F_matrix.T + Q_matrix
    print("Pe_n_n_1: ", Pe_n_n_1)
    Kn_n = Pe_n_n_1 @ H_matrix.T @ np.linalg.inv(H_matrix @ Pe_n_n_1 @ H_matrix.T + R_n)
    print("Kn_n: ", Kn_n)
    x_n_n = x_n_n_1 + Kn_n @ (z_n - H_matrix @ x_n_n_1)
    print("x_n_n: ", x_n_n)
    Pe_n_n = (np.eye(6) - Kn_n @ H_matrix) @ Pe_n_n_1 @ (np.eye(6) - Kn_n @ H_matrix).T + Kn_n @ R_n @ Kn_n.T
    print("Pe_n_n: ", Pe_n_n)

    x_list.append(x_n_n)
    pe_list.append(Pe_n_n)

# ===== show result =====

# 直线段
N1 = 100
x_line = np.full(N1, 300.0)
y_line = np.linspace(-400.0, 0.0, N1)

# 圆弧段（以 (0,0) 为圆心，半径 300，从 (300,0) 到 (0,300)）
N2 = 100
R = 300.0
theta = np.linspace(0.0, np.pi / 2, N2)
x_arc = R * np.cos(theta)
y_arc = R * np.sin(theta)

# 拼接（去掉圆弧第一个点避免与直线端点重复）
gt_x = np.concatenate([x_line, x_arc[1:]])
gt_y = np.concatenate([y_line, y_arc[1:]])

car = np.array(car_position)
est = np.array(x_list[1:])
plt.figure()
plt.plot(car[:, 0], car[:, 1], '-', label='measure', color='red')
plt.plot(est[:, 0], est[:, 3], '-', label='estimate', color='blue')
plt.plot(gt_x, gt_y, '-', label='true trajectory', color='green', linewidth=2)
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('measure vs estimate position trajectory')
plt.show()