import numpy as np
import random
import cv2
import utils
import os

class Rocket(object):
    """
    Rocekt and environment.
    The rocket is simplified into a rigid body model with a thin rod,
    considering acceleration and angular acceleration and air resistance
    proportional to velocity.

    There are two tasks: hover and landing
    Their reward functions are straight forward and simple.

    For the hover tasks: the step-reward is given based on two factors
    1) the distance between the rocket and the predefined target point
    2) the angle of the rocket body (the rocket should stay as upright as possible)

    For the landing task: the step-reward is given based on three factors:
    1) the distance between the rocket and the predefined landing point.
    2) the angle of the rocket body (the rocket should stay as upright as possible)
    3) Speed and angle at the moment of contact with the ground, when the touching-speed
    are smaller than a safe threshold and the angle is close to 90 degrees (upright),
    we see it as a successful landing.

    """

    def __init__(self, max_steps, task='hover', rocket_type='falcon',
                 viewport_h=768, path_to_bg_img=None):

        self.task = task
        self.rocket_type = rocket_type

        self.g = 9.8
        self.H = 50  # rocket height (meters)
        self.I = 1/12*self.H*self.H  # Moment of inertia
        self.dt = 0.05

        # --- LUT geometry scaling ---
        self.H_lut = 43.0  # 你的 LUT 对应的火箭高度（米）
        self.com_scale = self.H / self.H_lut

        # ==============================
        # CoM LUT (xc, zc) from lookup
        # Folder structure:
        #   rocket.py
        #   LUT_total_output/
        #       LUT_total_xc.npy
        #       LUT_total_zc.npy
        #       theta_grid_rad.npy
        #       total_mass_grid.npy
        # ==============================
        self.use_com_lut = True

        self._com_dir = os.path.join(os.path.dirname(__file__), "LUT_total_output_check")

        self.com_xc = None  # (Ntheta, Nm)
        self.com_zc = None  # (Ntheta, Nm)
        self.theta_grid = None  # rad, (Ntheta,)
        self.mass_grid = None   # kg, (Nm,)
        self.M0 = None          # initial total mass (kg), used to map m_ratio -> kg

        try:
            xc_path = os.path.join(self._com_dir, "LUT_total_xc.npy")
            zc_path = os.path.join(self._com_dir, "LUT_total_zc.npy")
            tg_path = os.path.join(self._com_dir, "theta_grid_rad.npy")
            mg_path = os.path.join(self._com_dir, "total_mass_grid.npy")

            self.com_xc = np.load(xc_path)
            self.com_zc = np.load(zc_path)
            self.theta_grid = np.load(tg_path).astype(float)
            self.mass_grid = np.load(mg_path).astype(float)

            # 用 mass_grid 的最大值当作初始总质量（kg），避免你在这里硬编码 45400
            self.M0 = float(np.max(self.mass_grid))

            # 基本一致性检查：表维度应为 (Ntheta, Nm)
            if self.com_xc.shape != (len(self.theta_grid), len(self.mass_grid)):
                # 如果你导出时转置了，就在这里修正一次
                if self.com_xc.T.shape == (len(self.theta_grid), len(self.mass_grid)):
                    self.com_xc = self.com_xc.T
                else:
                    raise ValueError("LUT_total_xc.npy shape mismatch with grids")

            if self.com_zc.shape != (len(self.theta_grid), len(self.mass_grid)):
                if self.com_zc.T.shape == (len(self.theta_grid), len(self.mass_grid)):
                    self.com_zc = self.com_zc.T
                else:
                    raise ValueError("LUT_total_zc.npy shape mismatch with grids")

        except Exception as e:
            # 兜底：找不到表或不匹配就回到原来的 H/2 刚体模型
            self.use_com_lut = False
            self.com_xc = None
            self.com_zc = None
            self.theta_grid = None
            self.mass_grid = None
            self.M0 = None


        # 不同推力->消耗燃料->质量变化对应值
        self.mass_consumption = {
            0.2 * self.g: 0.00021608,
            1.0 * self.g: 0.00107996,
            2.0 * self.g: 0.00215991
        }

        self.world_x_min = -300  # meters
        self.world_x_max = 300
        self.world_y_min = -30
        self.world_y_max = 570

        # target point
        if self.task == 'hover':
            self.target_x, self.target_y, self.target_r = 0, 200, 50
        elif self.task == 'landing':
            self.target_x, self.target_y, self.target_r = 0, self.H/2.0, 50

        self.already_landing = False
        self.already_crash = False
        self.already_fuel_empty = False
        self.max_steps = max_steps

        # viewport height x width (pixels)
        self.viewport_h = int(viewport_h)
        self.viewport_w = int(viewport_h * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        # 由于新增质量状态 m，state维度 8->9->11
        self.state_dims = 11
        self.action_dims = len(self.action_table)

        if path_to_bg_img is None:
            path_to_bg_img = task+'.jpg'
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)

        self.state_buffer = []


    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_random_state()
        else:
            self.state = state_dict

        self.state_buffer = []
        self.step_id = 0
        self.already_landing = False
        cv2.destroyAllWindows()
        return self.flatten(self.state)

    def create_action_table(self):
        f0 = 0.2 * self.g  # thrust
        f1 = 1.0 * self.g
        f2 = 2 * self.g
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi

        action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2],
                        [f1, vphi0], [f1, vphi1], [f1, vphi2],
                        [f2, vphi0], [f2, vphi1], [f2, vphi2]
                        ]
        return action_table

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_random_state(self):

        # predefined locations
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        xc = (self.world_x_max + self.world_x_min) / 2.0
        yc = (self.world_y_max + self.world_y_min) / 2.0

        if self.task == 'landing':
            x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
            y = yc + 0.4*y_range
            if x <= 0:
                theta = -85 / 180 * np.pi
            else:
                theta = 85 / 180 * np.pi
            vy = -50

        if self.task == 'hover':
            x = xc
            y = yc + 0.2 * y_range
            theta = random.uniform(-45, 45) / 180 * np.pi
            vy = -10

        state = {
            'x': x, 'y': y, 'vx': 0, 'vy': vy,
            'theta': theta, 'vtheta': 0,
            'phi': 0, 'f': 0,
            't': 0, 'a_': 0,
            # 新增rocket质量状态 m
            'm': 1.0
        }

        return state

    def check_crash(self, state):
        if self.task == 'hover':
            x, y = state['x'], state['y']
            theta = state['theta']
            crash = False
            if y <= self.H / 2.0:
                crash = True
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            return crash

        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5

            crash = False
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            if y <= 0 + self.H / 2.0 and v >= 15.0:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(x) >= self.target_r:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(theta) >= 10/180*np.pi:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(vtheta) >= 10/180*np.pi:
                crash = True
            return crash

    def check_landing_success(self, state):
        if self.task == 'hover':
            return False
        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5
            return True if y <= 0 + self.H / 2.0 and v < 15.0 and abs(x) < self.target_r \
                           and abs(theta) < 10/180*np.pi and abs(vtheta) < 10/180*np.pi else False

    def check_fuel_empty(self, state):
        if state['m'] <= 0.5770925:
            return True
        else:
            return False

    def calculate_reward(self, state):

        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range

        dist_reward = 0.1*(1.0 - dist_norm)

        if abs(state['theta']) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = abs(state['theta']) / (0.5*np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)

        reward = dist_reward + pose_reward

        # 离目标点越近、越竖直，奖励越高。
        if self.task == 'hover' and (dist_x**2 + dist_y**2)**0.5 <= 2*self.target_r:  # hit target
            reward = 0.25
        if self.task == 'hover' and (dist_x**2 + dist_y**2)**0.5 <= 1*self.target_r:  # hit target
            reward = 0.5
        # 当倾斜角 > 90° 时，奖励清零（代表“倒了”）
        if self.task == 'hover' and abs(state['theta']) > 90 / 180 * np.pi:
            reward = 0

        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5

        # 建议：整体降尺度，避免回报动辄上千导致更新爆炸
        TERM_SCALE = 0.3  # 可从 0.1 ~ 0.5 试
        if self.task == 'landing' and self.already_crash:
            reward = (reward + 5*np.exp(-1*v/10.)) * (self.max_steps - self.step_id) * TERM_SCALE
        elif self.task == 'landing' and self.already_landing:
            reward = (1.0 + 5*np.exp(-1*v/10.))*(self.max_steps - self.step_id) * TERM_SCALE
        elif self.task == 'landing' and self.already_fuel_empty:
            # 燃料耗尽给惩罚，阻止“躺平到没油”
            reward = -(self.max_steps - self.step_id) * TERM_SCALE

        return reward

    def step(self, action):

        x, y, vx, vy = self.state['x'], self.state['y'], self.state['vx'], self.state['vy']
        theta, vtheta = self.state['theta'], self.state['vtheta']
        phi = self.state['phi']
        m = self.state['m']

        f, vphi = self.action_table[action]

        # 计算推力方向与力矩
        ft, fr = -f*np.sin(phi), f*np.cos(phi)
        fx = ft*np.cos(theta) - fr*np.sin(theta)
        fy = ft*np.sin(theta) + fr*np.cos(theta)

        rho = 1 / (125/(self.g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
        # 添加质量m变化，影响动力学模型
        # ax, ay = fx-rho*vx, fy-self.g-rho*vy
        ax = (fx - rho * vx) / m
        ay = (fy - m * self.g - rho * vy) / m

        # 由于质量变化引起转动惯量I变化，缩放因数恰好就是m，因为m初始值为1，当前值<1
        mass_ratio = m

        # --- CoM from LUT (theta: -pi..pi supported) ---
        xc, zc = self._get_com_from_lut(theta, mass_ratio)

        # --- torque around CoM in 2D ---
        # thrust components in body frame already computed: ft (side), fr (axial)
        tau = zc * ft - xc * fr

        # angular acceleration
        atheta = tau / (self.I * mass_ratio)


        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0
            action = 0

        self.step_id += 1
        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt
        phi = phi + self.dt*vphi

        phi = max(phi, -20/180*3.1415926)
        phi = min(phi, 20/180*3.1415926)

        m_new = m - self.mass_consumption[f]

        self.state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': f,
            't': self.step_id, 'action_': action,
            'm': m_new
        }
        self.state_buffer.append(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)
        self.already_fuel_empty = self.check_fuel_empty(self.state)
        reward = self.calculate_reward(self.state)

        if self.already_crash or self.already_landing or self.already_fuel_empty:
            done = True
        else:
            done = False

        v = (self.state['vx'] ** 2 + self.state['vy'] ** 2) ** 0.5
        info = {
            "crash": bool(self.already_crash),
            "landing": bool(self.already_landing),
            "fuel_empty": bool(self.already_fuel_empty),
            "step_id": int(self.step_id),

            # 你要求新增的两项
            "last_reward": float(reward),  # 本 step 的 reward（终止时就是终止奖励）
            "last_v": float(v),  # 本 step 速度模长
            "last_vx": float(self.state['vx']),
            "last_vy": float(self.state['vy']),
        }

        return self.flatten(self.state), reward, done, info

    def flatten(self, state):
        xc, zc = self._get_com_from_lut(state['theta'], state['m'])
        x = [state['x'], state['y'], state['vx'], state['vy'],
             state['theta'], state['vtheta'], state['t'],
             state['phi'],
             state['m'],
             xc, zc]
        return np.array(x, dtype=np.float32)/100.


    def _wrap_to_pi(self, ang):
        # wrap any angle to [-pi, pi]
        return (ang + np.pi) % (2 * np.pi) - np.pi

    def _bilinear_lookup(self, table, theta, mass):
        """
        table: (Ntheta, Nm)
        theta: rad, will be wrapped to [-pi, pi]
        mass:  kg, will be clipped into [mass_grid[0], mass_grid[-1]]
        """
        theta = self._wrap_to_pi(theta)

        tg = self.theta_grid
        mg = self.mass_grid

        # clip to grid range
        theta = float(np.clip(theta, tg[0], tg[-1]))
        mass = float(np.clip(mass, mg[0], mg[-1]))

        # locate theta interval
        i1 = int(np.searchsorted(tg, theta, side="right"))
        i1 = min(max(i1, 1), len(tg) - 1)
        i0 = i1 - 1

        # locate mass interval
        j1 = int(np.searchsorted(mg, mass, side="right"))
        j1 = min(max(j1, 1), len(mg) - 1)
        j0 = j1 - 1

        t0, t1 = tg[i0], tg[i1]
        m0, m1 = mg[j0], mg[j1]

        wt = 0.0 if (t1 == t0) else (theta - t0) / (t1 - t0)
        wm = 0.0 if (m1 == m0) else (mass - m0) / (m1 - m0)

        v00 = table[i0, j0]
        v10 = table[i1, j0]
        v01 = table[i0, j1]
        v11 = table[i1, j1]

        v0 = v00 * (1 - wt) + v10 * wt
        v1 = v01 * (1 - wt) + v11 * wt
        return float(v0 * (1 - wm) + v1 * wm)

    def _get_com_from_lut(self, theta, m_ratio):
        """
        theta: rad (your state['theta'])
        m_ratio: your state['m'], assumed in [0,1], where 1 means initial mass
        return: (xc, zc) in meters, rocket body frame with origin at bottom center
        """
        if (not self.use_com_lut) or (self.com_xc is None) or (self.com_zc is None):
            return 0.0, self.H / 2.0

        # map m_ratio -> total mass (kg) using M0 inferred from total_mass_grid.npy
        m_ratio = float(np.clip(m_ratio, 0.0, 1.0))
        total_mass = m_ratio * self.M0

        # IMPORTANT:
        # Your LUT theta axis already covers [-180, 180], so:
        # - do NOT use abs(theta)
        # - do NOT mirror xc with sign(theta)
        xc = self._bilinear_lookup(self.com_xc, theta, total_mass)
        zc = self._bilinear_lookup(self.com_zc, theta, total_mass)

        # scale from LUT rocket (43m) to simulation rocket (50m)
        xc *= self.com_scale
        zc *= self.com_scale
        return xc, zc

    def render(self, window_name='env', wait_time=1,
               with_trajectory=True, with_camera_tracking=True,
               crop_scale=0.4):

        canvas = np.copy(self.bg_img)
        polys = self.create_polygons()

        # draw target region
        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly)
        # draw rocket
        for poly in polys['rocket']:
            self.draw_a_polygon(canvas, poly)
        frame_0 = canvas.copy()

        # draw engine work
        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly)
        frame_1 = canvas.copy()

        if with_camera_tracking:
            frame_0 = self.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
            frame_1 = self.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        # draw trajectory
        if with_trajectory:
            self.draw_trajectory(frame_0)
            self.draw_trajectory(frame_1)

        # draw text
        self.draw_text(frame_0, color=(0, 0, 0))
        self.draw_text(frame_1, color=(0, 0, 0))

        cv2.imshow(window_name, frame_0[:,:,::-1])
        cv2.waitKey(wait_time)
        cv2.imshow(window_name, frame_1[:,:,::-1])
        cv2.waitKey(wait_time)
        return frame_0, frame_1

    def create_polygons(self):

        polys = {'rocket': [], 'engine_work': [], 'target_region': []}

        if self.rocket_type == 'falcon':

            H, W = self.H, self.H/10
            dl = self.H / 30

            # rocket main body
            pts = [[-W/2, H/2], [W/2, H/2], [W/2, -H/2], [-W/2, -H/2]]
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})
            # rocket paint
            pts = utils.create_rectangle_poly(center=(0, -0.35*H), w=W, h=0.1*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            pts = utils.create_rectangle_poly(center=(0, -0.46*H), w=W, h=0.02*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            # rocket landing rack
            pts = [[-W/2, -H/2], [-W/2-H/10, -H/2-H/20], [-W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
            pts = [[W/2, -H/2], [W/2+H/10, -H/2-H/20], [W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})

        elif self.rocket_type == 'starship':

            H, W = self.H, self.H / 2.6
            dl = self.H / 30

            # rocket main body (right half)
            pts = np.array([[ 0.        ,  0.5006878 ],
                           [ 0.03125   ,  0.49243465],
                           [ 0.0625    ,  0.48143053],
                           [ 0.11458334,  0.43878955],
                           [ 0.15277778,  0.3933975 ],
                           [ 0.2326389 ,  0.23796424],
                           [ 0.2326389 , -0.49931225],
                           [ 0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

            # rocket main body (left half)
            pts = np.array([[-0.        ,  0.5006878 ],
                           [-0.03125   ,  0.49243465],
                           [-0.0625    ,  0.48143053],
                           [-0.11458334,  0.43878955],
                           [-0.15277778,  0.3933975 ],
                           [-0.2326389 ,  0.23796424],
                           [-0.2326389 , -0.49931225],
                           [-0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (212, 212, 232), 'edge_color': None})

            # upper wing (right)
            pts = np.array([[0.15972222, 0.3933975 ],
                           [0.3784722 , 0.303989  ],
                           [0.3784722 , 0.2352132 ],
                           [0.22916667, 0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # upper wing (left)
            pts = np.array([[-0.15972222,  0.3933975 ],
                           [-0.3784722 ,  0.303989  ],
                           [-0.3784722 ,  0.2352132 ],
                           [-0.22916667,  0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # lower wing (right)
            pts = np.array([[ 0.2326389 , -0.16368638],
                           [ 0.4548611 , -0.33562586],
                           [ 0.4548611 , -0.48555708],
                           [ 0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

            # lower wing (left)
            pts = np.array([[-0.2326389 , -0.16368638],
                           [-0.4548611 , -0.33562586],
                           [-0.4548611 , -0.48555708],
                           [-0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        else:
            raise NotImplementedError('rocket type [%s] is not found, please choose one '
                                      'from (falcon, starship)' % self.rocket_type)

        # engine work
        f, phi = self.state['f'], self.state['phi']
        c, s = np.cos(phi), np.sin(phi)

        if f > 0 and f < 0.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 0.5 * self.g and f < 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts4, 'face_color': (255, 255, 255), 'edge_color': None})
        # target region
        if self.task == 'hover':
            pts1 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=0, h=self.target_r/3.0)
            pts2 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=self.target_r/3.0, h=0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
        else:
            pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r/4.0)
            pts2 = utils.create_rectangle_poly(center=(0, 0), w=self.target_r/3.0, h=0)
            pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=self.target_r/6.0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

        # apply transformation
        for poly in polys['rocket'] + polys['engine_work']:
            M = utils.create_pose_matrix(tx=self.state['x'], ty=self.state['y'], rz=self.state['theta'])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys


    def draw_a_polygon(self, canvas, poly):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(pts)
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True):

        pts_px = np.zeros_like(pts)

        scale = self.viewport_w / (self.world_x_max - self.world_x_min)
        for i in range(len(pts)):
            pt = pts[i]
            x_p = (pt[0] - self.world_x_min) * scale
            y_p = (pt[1] - self.world_y_min) * scale
            y_p = self.viewport_h - y_p
            pts_px[i] = [x_p, y_p]

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(255, 255, 0)):

        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        text = "x: %.2f m, y: %.2f m" % \
               (self.state['x'], self.state['y'])
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "vx: %.2f m/s, vy: %.2f m/s" % \
               (self.state['vx'], self.state['vy'])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "a: %.2f degree, va: %.2f degree/s" % \
               (self.state['theta'] * 180 / np.pi, self.state['vtheta'] * 180 / np.pi)
        put_text(canvas, text, pt)


    def draw_trajectory(self, canvas, color=(255, 0, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        # draw horizon line
        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.H/2], [self.world_x_max - range_x/3, self.H/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw vertical line
        pts = [[0, self.H/2], [0, self.H/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        # draw traj
        pts = []
        for state in self.state_buffer:
            pts.append([state['x'], state['y']])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel



    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state['x'], self.state['y']
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        # check boundary
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis

