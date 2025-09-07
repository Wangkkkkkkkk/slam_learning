import numpy as np
import matplotlib.pyplot as plt

# ===== data =====
car_position = [
    [502.55, -0.9316],
    [477.34, -0.8977],
    [457.21, -0.8512],
    [442.94, -0.8114],
    [427.27, -0.7853],
    [406.05, -0.7392],
    [400.73, -0.7052],
    [377.32, -0.6478],
    [360.27, -0.59],
    [345.93, -0.5183],
    [333.34, -0.4698],
    [328.07, -0.3952],
    [315.48, -0.3026],
    [301.41, -0.2445],
    [302.87, -0.1626],
    [304.25, -0.0937],
    [294.46, 0.0085],
    [294.29, 0.0856],
    [299.38, 0.1675],
    [299.37, 0.2467],
    [300.68, 0.329],
    [304.1, 0.4149],
    [301.96, 0.504],
    [300.3, 0.5934],
    [301.9, 0.667],
    [296.7, 0.7537],
    [297.07, 0.8354],
    [295.29, 0.9195],
    [296.31, 1.0039],
    [300.62, 1.0923],
    [292.3, 1.1546],
    [298.11, 1.2564],
    [298.07, 1.3274],
    [298.92, 1.409],
    [298.04, 1.5011]
]

N = 6
sigma_number = 2 * N + 1
k = 3 - N

wi = []
for i in range(sigma_number):
    if i == 0:
        wi.append(k / (N + k))
    else:
        wi.append(1 / (2 * (N + k)))

wi = np.array(wi)
diag_wi = np.diag(wi)

def update_sigma_points(x, L):
    X_n_n = np.zeros((N, sigma_number))
    X_n_n[:, 0] = x
    for i in range(N):
        X_n_n[:, i + 1] = x + L[:, i]
        X_n_n[:, i + 1 + N] = x - L[:, i]
    return X_n_n

def H_matrix(X_n_n_1):
    dist = np.sqrt(X_n_n_1[0]**2 + X_n_n_1[3]**2)
    tan = np.arctan2(X_n_n_1[3], X_n_n_1[0])
    return np.array([dist, tan])

theta_a = 0.2
theta_rm = 5
theta_ym = 0.0087  # rad

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
    [pow(theta_rm, 2), 0],
    [0, pow(theta_ym, 2)]
])

x_0_0 = np.array([400, 0, 0, -300, 0, 0])

Pe_0_0 = np.array([
    [500, 0, 0, 0, 0, 0],
    [0, 500, 0, 0, 0, 0],
    [0, 0, 500, 0, 0, 0],
    [0, 0, 0, 500, 0, 0],
    [0, 0, 0, 0, 500, 0],
    [0, 0, 0, 0, 0, 500]
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

    matrix = (N + k) * pe_list[i]
    L = np.linalg.cholesky(matrix)
    print("L: ", L)
    
    X_n_n = update_sigma_points(x_list[i], L)
    print("X_n_n: ", X_n_n)

    X_n_n_1 = F_matrix @ X_n_n
    print("X_n_n_1: ", X_n_n_1)

    x_n_n_1 = X_n_n_1 @ wi
    print("x_n_n_1: ", x_n_n_1)

    Pe_n_n_1 = (X_n_n_1 - x_n_n_1[:, None]) @ diag_wi @ \
        (X_n_n_1 - x_n_n_1[:, None]).T + Q_matrix
    print("Pe_n_n_1: ", Pe_n_n_1)

    H_matrix_n = H_matrix(X_n_n_1)
    print("H_matrix_n: ", H_matrix_n)

    z_measure = H_matrix_n @ wi
    print("z_measure: ", z_measure)

    Pz1 = (H_matrix_n - z_measure[:, None]) @ diag_wi @ \
        (H_matrix_n - z_measure[:, None]).T + R_n
    print("Pz1: ", Pz1)

    Pxz1 = (X_n_n_1 - x_n_n_1[:, None]) @ diag_wi @ \
        (H_matrix_n - z_measure[:, None]).T
    print("Pxz1: ", Pxz1)

    Kn = Pxz1 @ np.linalg.inv(Pz1)
    print("Kn: ", Kn)

    x_n_n = x_n_n_1 + Kn @ (z_n - z_measure)
    print("x_n_n: ", x_n_n)

    Pe_n_n = Pe_n_n_1 - Kn @ Pz1 @ Kn.T
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

