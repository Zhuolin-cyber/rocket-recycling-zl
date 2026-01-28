# -*- coding: utf-8 -*-
"""
按你给定的对称规则：把 0~90° LUT 扩展到 [-pi, pi]（即 [-180°, 180°]）

你当前已有：./LUT_output/ 下的 0~90° LUT
  LUT_xc.npy, LUT_zc.npy
  (可选) theta_grid.npy               # 0~90°（deg 或 rad）
  (可选) fuel_fraction_grid.npy

要生成：[-pi, pi] 的完整 LUT（角度维扩展，比例维不变）

对称规则（你给定的逻辑，已按度数说明）：
1) 0 ~ 90: 直接用原表（xc, zc）
2) 90 ~ 180: 角度 a 对应 alpha = 180 - a（例如 91↔89, 135↔45）
   - xc 不变
   - zc 关于 H/2 对称：zc' = H - zc
3) -90 ~ 0: 角度 a 对应 alpha = -a（例如 -30↔30）
   - xc 取反：xc' = -xc
   - zc 不变
4) -180 ~ -90: 角度 a 对应 alpha = 180 - |a|（例如 -91↔89, -135↔45）
   - xc 取反
   - zc 关于 H/2 对称：zc' = H - zc

输出目录：./LUT_output_pm_pi/
  LUT_xc.npy, LUT_zc.npy, LUT_xc.csv, LUT_zc.csv
  theta_grid_deg.npy, theta_grid_rad.npy
  fuel_fraction_grid.npy

注意：
- 你必须在下面设置圆柱高度 H（单位 m），因为 zc 的对称用到 H。
"""

import os
import numpy as np

# ========= 你只需要改这些 =========
base_dir = "./LUT_output_OX"         # 0~90° LUT 所在目录
save_dir = "./LUT_output_OX_pm_pi"  # 输出目录
H = 25.0                         # 圆柱高度（用于 zc' = H - zc）
os.makedirs(save_dir, exist_ok=True)

# ========= 读取 0~90° LUT =========
LUT_xc_0_90 = np.load(os.path.join(base_dir, "OX_LUT_xc.npy")).astype(float)
LUT_zc_0_90 = np.load(os.path.join(base_dir, "OX_LUT_zc.npy")).astype(float)

if LUT_xc_0_90.shape != LUT_zc_0_90.shape:
    raise ValueError(f"LUT_xc 与 LUT_zc shape 不一致：{LUT_xc_0_90.shape} vs {LUT_zc_0_90.shape}")

N_theta, N_frac = LUT_xc_0_90.shape

# theta_grid（可选：没有就按等间隔 0~90 deg 推断）
theta_path = os.path.join(base_dir, "theta_grid.npy")
if os.path.exists(theta_path):
    theta_grid = np.load(theta_path).astype(float)
    # 判断单位：若最大值约 1.57 视为弧度
    if theta_grid.max() <= 2.0 and theta_grid.max() > 1.0:
        theta_deg = np.rad2deg(theta_grid)
    else:
        theta_deg = theta_grid
else:
    theta_deg = np.linspace(0.0, 90.0, N_theta)

# fuel_fraction_grid（可选：没有就按 0~1 推断）
frac_path = os.path.join(base_dir, "fuel_fraction_grid.npy")
if os.path.exists(frac_path):
    fuel_fraction_grid = np.load(frac_path).astype(float)
else:
    fuel_fraction_grid = np.linspace(0.0, 1.0, N_frac)

# 基本检查
if not (np.isclose(theta_deg[0], 0.0, atol=1e-6) and np.isclose(theta_deg[-1], 90.0, atol=1e-3)):
    raise ValueError(f"theta_grid 需要覆盖 0~90°。当前范围：{theta_deg[0]}~{theta_deg[-1]} deg")

# ========= 构造 [-180, 180] 角度网格（严格单调递增，避免边界重复） =========
# theta_deg = [0, ..., 90]
theta_tail = theta_deg[1:]  # 去掉 0，长度 N_theta-1

# 1) [-180] + (-179 .. -90) 来自 -(180 - theta_tail)
seg1 = np.concatenate([
    np.array([-180.0]),
    -(180.0 - theta_tail)    # theta=1..90 -> -179..-90
])

# 2) (-89 .. -1) 来自 -theta_tail[::-1] 去掉 -90
seg2_full = -theta_tail[::-1]   # -90..-1
seg2 = seg2_full[1:]            # -89..-1

# 3) 0..90
seg3 = theta_deg

# 4) 91..180 来自 90 + theta_tail
seg4 = 90.0 + theta_tail

angle_deg_full = np.concatenate([seg1, seg2, seg3, seg4])  # 单调递增
K = len(angle_deg_full)

# ========= 为每个角度生成映射到 alpha∈[0,90] 的规则，以及符号/对称标志 =========
alpha_deg = np.empty(K, dtype=float)
sign_x = np.ones(K, dtype=float)
reflect_z = np.zeros(K, dtype=bool)

for i, a in enumerate(angle_deg_full):
    if 0.0 <= a <= 90.0 + 1e-12:
        # 0..90
        alpha_deg[i] = a
        sign_x[i] = 1.0
        reflect_z[i] = False
    elif 90.0 < a <= 180.0 + 1e-12:
        # 90..180
        alpha_deg[i] = 180.0 - a
        sign_x[i] = 1.0
        reflect_z[i] = True
    elif -90.0 <= a < 0.0:
        # -90..0
        alpha_deg[i] = -a
        sign_x[i] = -1.0
        reflect_z[i] = False
    else:
        # -180..-90
        aa = abs(a)
        alpha_deg[i] = 180.0 - aa
        sign_x[i] = -1.0
        reflect_z[i] = True

# 数值安全：裁剪到 [0,90]
alpha_deg = np.clip(alpha_deg, 0.0, 90.0)

# ========= 线性插值到 alpha 上（兼容非等间隔 theta_grid） =========
def interp_LUT(LUT_0_90, alpha_query):
    out = np.empty((len(alpha_query), N_frac), dtype=float)
    for j in range(N_frac):
        out[:, j] = np.interp(alpha_query, theta_deg, LUT_0_90[:, j])
    return out

base_x = interp_LUT(LUT_xc_0_90, alpha_deg)  # (K, N_frac)
base_z = interp_LUT(LUT_zc_0_90, alpha_deg)  # (K, N_frac)

LUT_xc_full = sign_x[:, None] * base_x
LUT_zc_full = base_z.copy()
LUT_zc_full[reflect_z, :] = H - base_z[reflect_z, :]

# ========= 保存（风格按你截图：npy + csv；额外保存网格） =========
np.save(os.path.join(save_dir, "OX_LUT_xc.npy"), LUT_xc_full)
np.save(os.path.join(save_dir, "OX_LUT_zc.npy"), LUT_zc_full)

np.savetxt(os.path.join(save_dir, "OX_LUT_xc.csv"), LUT_xc_full, delimiter=",")
np.savetxt(os.path.join(save_dir, "OX_LUT_zc.csv"), LUT_zc_full, delimiter=",")

np.save(os.path.join(save_dir, "theta_grid_deg.npy"), angle_deg_full)
np.save(os.path.join(save_dir, "theta_grid_rad.npy"), np.deg2rad(angle_deg_full))
np.save(os.path.join(save_dir, "fuel_fraction_grid.npy"), fuel_fraction_grid)

print("Done.")
print("Input :", os.path.abspath(base_dir))
print("Output:", os.path.abspath(save_dir))
print("LUT shape:", LUT_xc_full.shape, "(angle, fraction)")
print("Angle range:", angle_deg_full[0], "~", angle_deg_full[-1], "deg (包含 -180 与 180)")
print("H =", H)
