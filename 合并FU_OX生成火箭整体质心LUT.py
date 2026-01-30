# -*- coding: utf-8 -*-
"""
合并燃料(FU)与液氧(OX)两张质心 LUT，生成“火箭整体质心” LUT（随 theta 与总质量变化）。

你已拥有：
- FU: LUT_xc.npy, LUT_zc.npy, (可选) fuel_fraction_grid.npy, theta_grid_rad.npy/theta_grid_deg.npy
- OX: LUT_xc.npy, LUT_zc.npy, (可选) fuel_fraction_grid.npy, theta_grid_rad.npy/theta_grid_deg.npy

本脚本会：
1) 以“燃料剩余比例”作为第二维（列维度），根据 O/F 比推算对应的“氧剩余比例”。
2) 从 FU / OX LUT 中读取对应 (theta, fraction) 的 (xc, zc)。
   - FU: 使用表格的燃料 fraction 网格点（不插值 fuel 维）
   - OX: 对推算出来的氧 fraction 做 1D 线性插值（在 OX fraction 维）
3) 用质心叠加公式得到火箭整体 (xc, zc)。
4) 输出两张新表：
   - LUT_total_xc.npy / LUT_total_zc.npy        shape = (N_theta, N_fu_frac)
   同时输出：
   - total_mass_grid.npy                        对应每一列的火箭总质量（kg）
   - theta_grid_rad.npy / theta_grid_deg.npy    对应每一行角度网格
   - CSV 版本便于快速查看

坐标系：火箭底部中心为原点，z 向上。
注意：如果你的 FU/OX LUT 的 zc 是“各自油箱局部坐标”（油箱底=0），你需要在下面设置 z_offset_fu / z_offset_ox 把它们平移到火箭坐标。
如果你的 LUT 已经是在火箭全局坐标下计算的（常见做法），把 offset 保持为 0 即可。
"""

import os
import numpy as np

# ======================
# 你只需要改这几个参数
# ======================
# LUT 路径（目录内包含 LUT_xc.npy, LUT_zc.npy 等）
FU_DIR = "./LUT_output_FU_pm_pi"   # 例如 "./LUT_FU_output" 或你自己的目录名
OX_DIR = "./LUT_output_OX_pm_pi"   # 例如 "./LUT_OX_output"

# 输出目录
OUT_DIR = "./LUT_total_output_check"
os.makedirs(OUT_DIR, exist_ok=True)

# 火箭结构参数（kg, m）
M_DRY = 25600.0 - 4410.0     # 21190 kg
Z_DRY = 21.5                 # m（均匀分布质心）
M_ENG = 4410.0               # kg
Z_ENG = 1.5                  # m

# 推进剂总质量（kg）
M_OX0 = 14400.0              # 5% 液氧质量（你给的数）
M_FU0 = 5400.0               # 5% 煤油质量（你给的数）

# 消耗比（质量比）
OF_RATIO = 2.56              # O/F

# 若 FU/OX 的 LUT_zc 是局部坐标（油箱底=0），这里填“油箱底部在火箭坐标中的 z 值”
# 如果你的 LUT_zc 已经是全局坐标，保持 0 即可。
Z_OFFSET_FU = 3.0
Z_OFFSET_OX = 18.0

# ======================
# 工具函数：加载 LUT
# ======================
def _load_grid_any(dir_path: str):
    th_rad = os.path.join(dir_path, "theta_grid_rad.npy")
    th_deg = os.path.join(dir_path, "theta_grid_deg.npy")
    frac_p = os.path.join(dir_path, "fuel_fraction_grid.npy")

    theta_rad = None
    theta_deg = None

    if os.path.exists(th_rad):
        theta_rad = np.load(th_rad).astype(float)
        theta_deg = np.rad2deg(theta_rad)
    elif os.path.exists(th_deg):
        theta_deg = np.load(th_deg).astype(float)
        theta_rad = np.deg2rad(theta_deg)

    frac = np.load(frac_p).astype(float) if os.path.exists(frac_p) else None
    return theta_rad, theta_deg, frac

def load_lut(dir_path: str):
    x_path = os.path.join(dir_path, "LUT_xc.npy")
    z_path = os.path.join(dir_path, "LUT_zc.npy")
    # 兼容你的命名：FU_LUT_xc.npy / OX_LUT_xc.npy
    if not (os.path.exists(x_path) and os.path.exists(z_path)):
        # 在目录里找 *LUT_xc.npy / *LUT_zc.npy
        cand_x = [p for p in os.listdir(dir_path) if p.endswith("LUT_xc.npy")]
        cand_z = [p for p in os.listdir(dir_path) if p.endswith("LUT_zc.npy")]
        if len(cand_x) != 1 or len(cand_z) != 1:
            raise FileNotFoundError(f"缺少 xc/zc 文件或存在多个候选：{dir_path}")
        x_path = os.path.join(dir_path, cand_x[0])
        z_path = os.path.join(dir_path, cand_z[0])

    LUT_x = np.load(x_path).astype(float)
    LUT_z = np.load(z_path).astype(float)
    if LUT_x.shape != LUT_z.shape:
        raise ValueError(f"{dir_path}：LUT_xc 与 LUT_zc shape 不一致：{LUT_x.shape} vs {LUT_z.shape}")

    theta_rad, theta_deg, frac = _load_grid_any(dir_path)

    # 兜底推断
    if theta_deg is None:
        theta_deg = np.linspace(0.0, 90.0, LUT_x.shape[0])
        theta_rad = np.deg2rad(theta_deg)
    if frac is None:
        frac = np.linspace(0.0, 1.0, LUT_x.shape[1])

    # --------- 关键修复：对齐 LUT 列顺序与 frac_grid 含义 ----------
    # 你的实际 LUT 文件：第 0 列对应“最大剩余”（满），最后一列对应“最小剩余”（空）
    # 但 fuel_fraction_grid.npy 是 0->1 递增（0=空，1=满）
    # 因此必须把 LUT 的列翻转，使得：第 j 列 <-> frac[j] 的语义一致
    if frac[0] < frac[-1]:  # frac 是递增 0->1（你的情况）
        LUT_x = LUT_x[:, ::-1]
        LUT_z = LUT_z[:, ::-1]

    # 单调检查（便于插值）
    if np.any(np.diff(theta_rad) <= 0):
        raise ValueError(f"{dir_path}：theta_grid 必须严格递增")
    if np.any(np.diff(frac) <= 0):
        raise ValueError(f"{dir_path}：fuel_fraction_grid 必须严格递增")

    return LUT_x, LUT_z, theta_rad, theta_deg, frac

def interp_frac(LUT_theta_frac: np.ndarray, frac_grid: np.ndarray, frac_q: np.ndarray):
    frac_q = np.asarray(frac_q).astype(float)
    N_theta = LUT_theta_frac.shape[0]
    out = np.empty((N_theta, len(frac_q)), dtype=float)
    for i in range(N_theta):
        out[i, :] = np.interp(frac_q, frac_grid, LUT_theta_frac[i, :])
    return out

# ======================
# 主流程：合并生成总质心 LUT
# ======================
def main():
    FU_x, FU_z, th_fu_rad, th_fu_deg, fu_frac_grid = load_lut(FU_DIR)
    OX_x, OX_z, th_ox_rad, th_ox_deg, ox_frac_grid = load_lut(OX_DIR)

    # 角度网格一致性
    if len(th_fu_rad) != len(th_ox_rad) or np.max(np.abs(th_fu_rad - th_ox_rad)) > 1e-8:
        raise ValueError(
            "FU 与 OX 的角度网格不一致。请先确保两套 LUT 使用同一个 theta_grid。\n"
            f"FU theta size={len(th_fu_rad)}, OX theta size={len(th_ox_rad)}"
        )

    theta_rad = th_fu_rad
    theta_deg = th_fu_deg

    # 输出列维：燃料剩余比例（来自 FU LUT）
    fu_frac = fu_frac_grid
    m_fu = M_FU0 * fu_frac

    # 推算氧剩余
    m_fu_consumed = M_FU0 - m_fu
    m_ox = M_OX0 - OF_RATIO * m_fu_consumed
    m_ox = np.clip(m_ox, 0.0, M_OX0)
    ox_frac = m_ox / M_OX0

    # 读取/插值推进剂质心（加 z offset）
    FU_x_use = FU_x
    FU_z_use = FU_z + Z_OFFSET_FU

    OX_x_i = interp_frac(OX_x, ox_frac_grid, ox_frac)
    OX_z_i = interp_frac(OX_z, ox_frac_grid, ox_frac) + Z_OFFSET_OX

    # 合并质心
    M_total = M_DRY + M_ENG + m_fu + m_ox  # (N_fu,)

    LUT_total_xc = (m_fu[None, :] * FU_x_use + m_ox[None, :] * OX_x_i) / M_total[None, :]
    z_const = M_DRY * Z_DRY + M_ENG * Z_ENG
    LUT_total_zc = (z_const + m_fu[None, :] * FU_z_use + m_ox[None, :] * OX_z_i) / M_total[None, :]

    # 保存
    np.save(os.path.join(OUT_DIR, "LUT_total_xc.npy"), LUT_total_xc)
    np.save(os.path.join(OUT_DIR, "LUT_total_zc.npy"), LUT_total_zc)

    np.save(os.path.join(OUT_DIR, "theta_grid_rad.npy"), theta_rad)
    np.save(os.path.join(OUT_DIR, "theta_grid_deg.npy"), theta_deg)

    np.save(os.path.join(OUT_DIR, "fuel_fraction_grid_fu.npy"), fu_frac)
    np.save(os.path.join(OUT_DIR, "ox_fraction_mapped.npy"), ox_frac)

    np.save(os.path.join(OUT_DIR, "total_mass_grid.npy"), M_total)
    np.save(os.path.join(OUT_DIR, "fu_mass_grid.npy"), m_fu)
    np.save(os.path.join(OUT_DIR, "ox_mass_grid.npy"), m_ox)

    header = ",".join(["theta_rad"] + [f"{f:.8f}" for f in fu_frac])
    np.savetxt(os.path.join(OUT_DIR, "LUT_total_xc.csv"),
               np.column_stack([theta_rad, LUT_total_xc]),
               delimiter=",", header=header, comments="")
    np.savetxt(os.path.join(OUT_DIR, "LUT_total_zc.csv"),
               np.column_stack([theta_rad, LUT_total_zc]),
               delimiter=",", header=header, comments="")

    print("Done.")
    print("OUT_DIR:", os.path.abspath(OUT_DIR))
    print("LUT_total shape:", LUT_total_xc.shape, "(theta, fuel_fraction)")
    print("theta range(rad):", theta_rad[0], "~", theta_rad[-1])
    print("total mass range(kg):", float(M_total.min()), "~", float(M_total.max()))
    print("columns indexed by fuel fraction; OX fraction mapped by O/F.")

if __name__ == "__main__":
    main()
