# -*- coding: utf-8 -*-
"""
LUT 质心可视化 UI（角度 + 剩余比例 -> 2D 坐标系中的质心点）

依赖：仅标准库 + numpy + matplotlib（通常你已有）
运行：python LUT_UI_viewer.py

默认读取：
  ./LUT_output_full/
      LUT_xc.npy
      LUT_zc.npy
      theta_grid_deg.npy
      fuel_fraction_grid.npy

说明：
- 角度与剩余比例使用滑块连续调节。
- 内部对 LUT 做二维双线性插值（angle / fraction 都可连续）。
- 2D 坐标系展示点 (xc, zc)，并显示数值。
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# 配置：路径
# =========================
LUT_DIR = "./LUT_output_OX_pm_pi"
PATH_X = os.path.join(LUT_DIR, "OX_LUT_xc.npy")
PATH_Z = os.path.join(LUT_DIR, "OX_LUT_zc.npy")
PATH_TH = os.path.join(LUT_DIR, "theta_grid_deg.npy")
PATH_FR = os.path.join(LUT_DIR, "fuel_fraction_grid.npy")


def load_lut():
    for p in (PATH_X, PATH_Z, PATH_TH, PATH_FR):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文件：{p}")

    LUT_x = np.load(PATH_X).astype(float)
    LUT_z = np.load(PATH_Z).astype(float)
    theta_deg = np.load(PATH_TH).astype(float)
    frac = np.load(PATH_FR).astype(float)

    if LUT_x.shape != LUT_z.shape:
        raise ValueError(f"LUT_x 与 LUT_z 形状不一致：{LUT_x.shape} vs {LUT_z.shape}")
    if LUT_x.shape[0] != theta_deg.shape[0]:
        raise ValueError(f"角度网格长度不匹配：LUT行={LUT_x.shape[0]} vs theta={theta_deg.shape[0]}")
    if LUT_x.shape[1] != frac.shape[0]:
        raise ValueError(f"比例网格长度不匹配：LUT列={LUT_x.shape[1]} vs frac={frac.shape[0]}")

    # 保证单调递增
    if np.any(np.diff(theta_deg) <= 0):
        raise ValueError("theta_grid_deg.npy 必须严格递增（例如 0~360 不含 360 端点）")
    if np.any(np.diff(frac) <= 0):
        raise ValueError("fuel_fraction_grid.npy 必须严格递增")

    return LUT_x, LUT_z, theta_deg, frac


def bilinear_interp(grid_x, grid_y, values, xq, yq):
    """
    对二维规则网格做双线性插值。
    grid_x: shape (Nx,) 单调递增
    grid_y: shape (Ny,) 单调递增
    values: shape (Nx, Ny)
    xq, yq: 查询点（标量）
    """
    x = float(np.clip(xq, grid_x[0], grid_x[-1]))
    y = float(np.clip(yq, grid_y[0], grid_y[-1]))

    ix = int(np.searchsorted(grid_x, x, side="right") - 1)
    iy = int(np.searchsorted(grid_y, y, side="right") - 1)

    ix = max(0, min(ix, len(grid_x) - 2))
    iy = max(0, min(iy, len(grid_y) - 2))

    x0, x1 = grid_x[ix], grid_x[ix + 1]
    y0, y1 = grid_y[iy], grid_y[iy + 1]

    # 避免除 0（理论上不会发生，因为严格递增）
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]

    v0 = (1 - tx) * v00 + tx * v10
    v1 = (1 - tx) * v01 + tx * v11
    return (1 - ty) * v0 + ty * v1


class LUTViewer(tk.Tk):
    def __init__(self, LUT_x, LUT_z, theta_deg, frac):
        super().__init__()
        self.title("LUT 质心可视化")
        self.geometry("980x640")

        self.LUT_x = LUT_x
        self.LUT_z = LUT_z
        self.theta_deg = theta_deg
        self.frac = frac

        # 初值
        self.angle_var = tk.DoubleVar(value=float(theta_deg[0]))
        self.frac_var = tk.DoubleVar(value=float(frac[-1]))  # 默认满液或最大比例

        # 计算全局坐标范围（用于稳定视图）
        self.x_min = float(np.nanmin(LUT_x))
        self.x_max = float(np.nanmax(LUT_x))
        self.z_min = float(np.nanmin(LUT_z))
        self.z_max = float(np.nanmax(LUT_z))

        # UI
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        # 左侧控制区
        ctl = ttk.Frame(root)
        ctl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(ctl, text="角度 (deg)").pack(anchor="w")
        self.angle_slider = ttk.Scale(
            ctl, from_=float(self.theta_deg[0]), to=float(self.theta_deg[-1]),
            orient=tk.HORIZONTAL, variable=self.angle_var, command=lambda _=None: self._update_plot()
        )
        self.angle_slider.pack(fill=tk.X, pady=(0, 10))

        self.angle_entry = ttk.Entry(ctl, textvariable=self.angle_var, width=12)
        self.angle_entry.pack(anchor="w", pady=(0, 15))
        self.angle_entry.bind("<Return>", lambda _=None: self._clamp_and_update())

        ttk.Label(ctl, text="剩余比例 (fuel fraction)").pack(anchor="w")
        self.frac_slider = ttk.Scale(
            ctl, from_=float(self.frac[0]), to=float(self.frac[-1]),
            orient=tk.HORIZONTAL, variable=self.frac_var, command=lambda _=None: self._update_plot()
        )
        self.frac_slider.pack(fill=tk.X, pady=(0, 10))

        self.frac_entry = ttk.Entry(ctl, textvariable=self.frac_var, width=12)
        self.frac_entry.pack(anchor="w", pady=(0, 15))
        self.frac_entry.bind("<Return>", lambda _=None: self._clamp_and_update())

        ttk.Separator(ctl).pack(fill=tk.X, pady=10)

        self.value_label = ttk.Label(ctl, text="(xc, zc) = ")
        self.value_label.pack(anchor="w", pady=(0, 8))

        ttk.Button(ctl, text="重置视图", command=self._reset_view).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(ctl, text="退出", command=self.destroy).pack(fill=tk.X, pady=(6, 0))

        ttk.Label(
            ctl,
            text=("说明：\n"
                  "- 拖动滑块可连续调节。\n"
                  "- 使用双线性插值读取 LUT。\n"
                  "- 点坐标为 (xc, zc)。"),
            justify="left"
        ).pack(anchor="w", pady=(12, 0))

        # 右侧绘图区
        fig = plt.Figure(figsize=(7.0, 5.0), dpi=100)
        # 左：质心视图
        self.ax = fig.add_subplot(1, 2, 1)

        # 右：theta 对齐刚体视图
        self.ax_pose = fig.add_subplot(1, 2, 2)

        self.ax.set_xlabel("xc")
        self.ax.set_ylabel("zc")
        self.ax.axhline(0.0, linewidth=1)
        self.ax.axvline(0.0, linewidth=1)
        self.ax.grid(True, linewidth=0.5)


        self.ax_pose.set_aspect("equal")
        self.ax_pose.set_title("Theta-aligned body view")
        self.ax_pose.grid(True, linewidth=0.5)
        self.ax_pose.set_xlim(-8.5, 8.5)
        self.ax_pose.set_ylim(-8.5, 8.5)

        self.point = self.ax.scatter([0.0], [0.0], s=80)
        self.ann = self.ax.annotate(
            "", xy=(0.0, 0.0), xytext=(10, 10), textcoords="offset points"
        )

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._reset_view()

    def _draw_rotating_rect(self, theta_rad):
        ax = self.ax_pose
        ax.cla()

        # 基本设置
        ax.set_aspect("equal")
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-8.5, 8.5)
        ax.grid(True, linewidth=0.5)
        ax.set_title("Theta-aligned body view")

        # 矩形比例 8:3
        H = 8.0
        W = 3.0

        # 未旋转的矩形（下底中点在原点）
        rect = np.array([
            [-W / 2, 0],
            [W / 2, 0],
            [W / 2, H],
            [-W / 2, H],
            [-W / 2, 0]
        ])

        # 旋转矩阵
        c, s = np.cos(-theta_rad), np.sin(-theta_rad)
        R = np.array([[c, -s],
                      [s, c]])

        rect_r = rect @ R.T

        # 画矩形
        ax.plot(rect_r[:, 0], rect_r[:, 1], "k", linewidth=2)

        # 画随体坐标轴
        axis_len = 2.0
        ex = np.array([axis_len, 0]) @ R.T
        ey = np.array([0, axis_len]) @ R.T

        ax.arrow(0, 0, ex[0], ex[1], head_width=0.15, length_includes_head=True)
        ax.arrow(0, 0, ey[0], ey[1], head_width=0.15, length_includes_head=True)

        ax.text(ex[0], ex[1], "x'", fontsize=10)
        ax.text(ey[0], ey[1], "z'", fontsize=10)

        # 原点
        ax.plot(0, 0, "ko")


    def _reset_view(self):
        # 视野稍微加点边距
        dx = (self.x_max - self.x_min) * 0.08 if self.x_max > self.x_min else 1.0
        dz = (self.z_max - self.z_min) * 0.08 if self.z_max > self.z_min else 1.0
        self.ax.set_xlim(self.x_min - dx, self.x_max + dx)
        self.ax.set_ylim(self.z_min - dz, self.z_max + dz)
        self.canvas.draw_idle()

    def _clamp_and_update(self):
        # 输入框回车时，做裁剪并刷新
        a = float(self.angle_var.get())
        f = float(self.frac_var.get())
        a = float(np.clip(a, self.theta_deg[0], self.theta_deg[-1]))
        f = float(np.clip(f, self.frac[0], self.frac[-1]))
        self.angle_var.set(a)
        self.frac_var.set(f)
        self._update_plot()

    def _update_plot(self):
        a = float(self.angle_var.get())
        f = float(self.frac_var.get())

        # 插值（分别对 xc / zc）
        xc = bilinear_interp(self.theta_deg, self.frac, self.LUT_x, a, f)
        zc = bilinear_interp(self.theta_deg, self.frac, self.LUT_z, a, f)

        # 更新点
        self.point.set_offsets(np.array([[xc, zc]]))
        self.ann.set_text(f"({xc:.6f}, {zc:.6f})")
        self.ann.xy = (xc, zc)

        self.value_label.configure(text=f"(xc, zc) = ({xc:.6f}, {zc:.6f})")

        theta_rad = np.deg2rad(a)
        self._draw_rotating_rect(theta_rad)

        self.canvas.draw_idle()


def main():
    try:
        LUT_x, LUT_z, theta_deg, frac = load_lut()
    except Exception as e:
        tk.Tk().withdraw()
        messagebox.showerror("加载失败", str(e))
        return

    app = LUTViewer(LUT_x, LUT_z, theta_deg, frac)
    app.mainloop()


if __name__ == "__main__":
    main()
