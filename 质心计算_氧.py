import numpy as np

def centroid_truncated_cylinder(R, h, theta, tol=1e-10, max_iter=80):
    """
    计算当 M < M_crit 时贮箱内液体的质心。
    - 自动处理两种几何情形：
      情况 1：平面只切到底面（+侧面）；
      情况 2：平面同时切到底面和顶面（你图里的 C = A - B）。
    R      : 圆柱半径
    h      : 等效液柱高度（由体积给出, V = pi * R^2 * h）
    theta  : 倾斜角（弧度）
    返回值 : (xc, zc)，相对于圆柱坐标系底面的质心坐标
    """

    a = np.tan(theta)
    V_target = np.pi * R**2 * h         # 目标体积
    if V_target <= 0:
        return 0.0, 0.0

    # -----------------------------
    # 底面几何量：右侧区域 D(x0) = {x >= x0} 上的面积/矩
    # -----------------------------
    def area_D(x0):
        # A(x0) = 圆盘右侧的面积
        seg = R**2 * np.arccos(-x0 / R) + x0 * np.sqrt(R**2 - x0**2)
        return np.pi * R**2 - seg

    def I1_D(x0):
        # I1(x0) = ∬_D x dA
        return -(2.0 / 3.0) * (x0**2 - R**2) * np.sqrt(R**2 - x0**2)

    def I2_D(x0):
        # I2(x0) = ∬_D x^2 dA
        t = np.arcsin(x0 / R)
        return R**4 * (np.pi / 8.0 - t / 4.0 + np.sin(4.0 * t) / 16.0)

    global H  # 使用外部定义的罐高 H

    # --------------------------------------------------
    # 特殊情况：theta = 90°，液面为竖直平面（规则柱体，底面为圆缺）
    # --------------------------------------------------
    if np.isclose(theta, 0.5 * np.pi, atol=1e-6):
        # V = H * A(x0) = V_target → A_target = V_target / H
        A_target = V_target / H

        # 在 [-R, R] 上二分求解 A(x0) = A_target
        x_lo, x_hi = -R + 1e-9, R - 1e-9
        for _ in range(max_iter):
            x_mid = 0.5 * (x_lo + x_hi)
            A_mid = area_D(x_mid)
            if A_mid > A_target:
                # 面积偏大 → 平面向右移动（增大 x0）
                x_lo = x_mid
            else:
                x_hi = x_mid
            if x_hi - x_lo < tol:
                break

        x0 = 0.5 * (x_lo + x_hi)

        A = area_D(x0)
        I1 = I1_D(x0)

        # 沿 z 方向是完整高度 H，质心在中点
        xc = I1 / A
        zc = H / 2.0
        return xc, zc

    # -----------------------------
    # 情况 1：只考虑底面截断时的体积函数 V_A(x0)
    # -----------------------------
    def volume_case1(x0):
        h_plane = -a * x0
        return h_plane * area_D(x0) + a * I1_D(x0)

    # 先在 [-R, R] 上用二分法求一个 “忽略顶面” 的 x0 解
    x_lo, x_hi = -R + 1e-9, R - 1e-9
    for _ in range(max_iter):
        x_mid = 0.5 * (x_lo + x_hi)
        if volume_case1(x_mid) > V_target:
            x_lo = x_mid
        else:
            x_hi = x_mid
        if x_hi - x_lo < tol:
            break
    x0_guess = 0.5 * (x_lo + x_hi)

    # -----------------------------
    # 判断是否会切到顶面
    # -----------------------------


    # theta = 0 时，无倾斜，体是圆柱的一段
    if a == 0:
        xc = 0.0
        zc = h / 2.0
        return xc, zc

    # 平面与顶面 z = H 的交线 x_t = x0 + H/a
    x_t_guess = x0_guess + H / a

    # 若交点在圆外（x_t >= R），说明只切到底面 → 情况 1
    if x_t_guess >= R - 1e-9:
        x0 = x0_guess
        h_plane = -a * x0
        A0 = area_D(x0)
        I1_0 = I1_D(x0)
        I2_0 = I2_D(x0)
        V = V_target  # 与 volume_case1(x0) 数值相等

        # 质心公式：x_c = (h I1 + a I2)/V
        xc = (a * I2_0 + h_plane * I1_0) / V
        # z_c = (1/(2V)) ∫(h+ax)^2 dA = [...]
        zc = (h_plane**2 * A0 + 2.0 * a * h_plane * I1_0 + a**2 * I2_0) / (2.0 * V)
        return xc, zc

    # -----------------------------
    # 情况 2：同时切到底面 + 顶面（C = A - B）
    # -----------------------------
    # 体积 C(x0) = V_A(x0) - V_B(x0)
    def volume_case2(x0):
        x_t = x0 + H / a           # 顶面交线
        hA = -a * x0               # 区域 A 平面的常数项
        hB = hA - H                # 区域 B 在 z' = z - H 下的常数项

        A0 = area_D(x0)
        I1_0 = I1_D(x0)
        At = area_D(x_t)
        I1_t = I1_D(x_t)

        V_A = hA * A0 + a * I1_0
        V_B = hB * At + a * I1_t
        return V_A - V_B

    # x0 的几何可行区间：[-R, x0_crit)，其中 x0_crit 对应 x_t = R
    x0_crit = R - H / a
    x_L = -R + 1e-9
    x_H = x0_crit - 1e-9

    # 如果几何上根本没有“同时切顶底”的可能，就退回情况 1
    if x_H <= x_L:
        x0 = x0_guess
        h_plane = -a * x0
        A0 = area_D(x0)
        I1_0 = I1_D(x0)
        I2_0 = I2_D(x0)
        V = V_target
        xc = (a * I2_0 + h_plane * I1_0) / V
        zc = (h_plane**2 * A0 + 2.0 * a * h_plane * I1_0 + a**2 * I2_0) / (2.0 * V)
        return xc, zc

    V_L = volume_case2(x_L)
    V_H = volume_case2(x_H)

    # 若目标体积不在 C(x0) 的值域内，同样退回情况 1（数值安全网）
    if not (V_H <= V_target <= V_L):
        x0 = x0_guess
        h_plane = -a * x0
        A0 = area_D(x0)
        I1_0 = I1_D(x0)
        I2_0 = I2_D(x0)
        V = V_target
        xc = (a * I2_0 + h_plane * I1_0) / V
        zc = (h_plane**2 * A0 + 2.0 * a * h_plane * I1_0 + a**2 * I2_0) / (2.0 * V)
        return xc, zc

    # 在 [-R, x0_crit) 上对 C(x0) = V_target 二分求解
    lo, hi = x_L, x_H
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        V_mid = volume_case2(mid)
        if V_mid > V_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    x0 = 0.5 * (lo + hi)
    x_t = x0 + H / a

    # -----------------------------
    # 计算 A 区和 B 区的体积与质心，再用叠加得到 C
    # -----------------------------
    hA = -a * x0
    hB = hA - H

    A0 = area_D(x0)
    I1_0 = I1_D(x0)
    I2_0 = I2_D(x0)

    At = area_D(x_t)
    I1_t = I1_D(x_t)
    I2_t = I2_D(x_t)

    V_A = hA * A0 + a * I1_0
    V_B = hB * At + a * I1_t
    V_C = V_A - V_B    # 应当与 V_target 数值相等

    # A 的质心
    x_A = (hA * I1_0 + a * I2_0) / V_A
    z_A = (hA**2 * A0 + 2.0 * a * hA * I1_0 + a**2 * I2_0) / (2.0 * V_A)

    # B 的质心（在 z' = z - H 局部坐标下先算，再整体上移 H）
    x_B_local = (hB * I1_t + a * I2_t) / V_B
    z_B_local = (hB**2 * At + 2.0 * a * hB * I1_t + a**2 * I2_t) / (2.0 * V_B)

    x_B = x_B_local
    z_B = H + z_B_local

    # 叠加公式：V_A r_A = V_B r_B + V_C r_C
    xc = (V_A * x_A - V_B * x_B) / V_C
    zc = (V_A * z_A - V_B * z_B) / V_C

    return xc, zc

# -------------------------
# Parameters
# -------------------------
R = 3.66 / 2          # tank radius (m)
H = 25.0              # tank height (m)
fuel_mass_5percent = 14400  # kg
fuel_mass = 14400 * 20  # kg
rho = fuel_mass_5percent / (0.05 * np.pi * R**2 * H)  # density inferred
g = 9.81

# Fuel levels (mass fraction from 5% -> 0%)
fuel_fraction_grid = np.linspace(0.05, 0.0, 200)  # 200 sample points
theta_grid = np.linspace(0, 0.5*np.pi, 180)         # angle sampling

# Result tables
LUT_xc = np.zeros((len(theta_grid), len(fuel_fraction_grid)))
LUT_zc = np.zeros((len(theta_grid), len(fuel_fraction_grid)))

# -------------------------
# Helper: convert mass fraction -> liquid height h
# -------------------------
def fuel_fraction_to_height(f):
    V = f * np.pi * R**2 * H
    h = V / (np.pi * R**2)
    return h

# -------------------------
# Main loop
# -------------------------
for i, theta in enumerate(theta_grid):

    a = np.tan(theta)  # slope of the inclined liquid surface

    for j, frac in enumerate(fuel_fraction_grid):

        h = fuel_fraction_to_height(frac)  # liquid height

        M = frac * fuel_mass
        M_crit = rho * np.pi * R ** 3 * np.tan(theta)

        if M < M_crit:
            xc, zc = centroid_truncated_cylinder(R, h, theta)
            LUT_xc[i, j] = xc
            LUT_zc[i, j] = zc
            continue

        # if i % 20 == 0 and i != 0:
        #     print('')

        # simplified slanted-cylinder centroid model (not handling plane-through-bottom case)
        xc = (a * R**2) / (4 * h)
        zc = (h / 2) + (a*a * R**2) / (8 * h)

        LUT_xc[i, j] = xc
        LUT_zc[i, j] = zc

print("Finished generating LUT.")
print("LUT_xc shape:", LUT_xc.shape)
print("LUT_zc shape:", LUT_zc.shape)

import os

# 保存路径
save_dir = "./LUT_output_OX"
os.makedirs(save_dir, exist_ok=True)

# 保存两个重心表
np.save(os.path.join(save_dir, "OX_LUT_xc.npy"), LUT_xc)
np.save(os.path.join(save_dir, "OX_LUT_zc.npy"), LUT_zc)

np.savetxt(os.path.join(save_dir, "OX_LUT_xc.csv"), LUT_xc, delimiter=",")
np.savetxt(os.path.join(save_dir, "OX_LUT_zc.csv"), LUT_zc, delimiter=",")
