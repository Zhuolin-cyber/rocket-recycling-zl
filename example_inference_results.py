import torch
from rocket import Rocket
from policy import ActorCritic
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    task = 'landing'
    max_steps = 800
    # 调用最新训练的模型
    # ckpt_dir = glob.glob(os.path.join(task + '_ckpt', '*.pt'))[-1]

    ckpt_name = "ckpt_00043001.pt"  # 指定要加载的pt文件名
    ckpt_dir = os.path.join(task + '_ckpt', ckpt_name)

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, map_location=device)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    n_trials = 200
    distances = []
    crash_count = 0
    success_steps = []  # 新增：记录每次成功所用的step数
    # 统计每次成功落地的各档位推力使用量
    success_thrust_counts = []  # 存储多次试验的统计结果

    for i in range(n_trials):  # 多次执行
        thrust_count = defaultdict(int)  # 每个试验都要重新初始化
        state = env.reset()
        for step_id in range(max_steps):
            action, _, _ = net.get_action(state)

            # 新增：记录当前推力档位
            thrust_level = int(action)  # 若 action 是 one-hot 或 float，请改成 argmax(action)
            thrust_count[thrust_level] += 1


            state, reward, done, _ = env.step(action)
            if done or env.already_crash:
                final_step = step_id + 1  # 新增：记录完成所用step
                break

        # 记录落点位置
        x = env.state['x']
        y = env.state['y']
        center_x = env.target_x
        center_y = env.target_y
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        distances.append(dist)
        # 判断是否crash
        if env.already_crash or dist > 50:
            crash_count += 1
            outcome = "crash"
        else:
            outcome = "success"
            success_steps.append(final_step)  # 新增：记录成功所需的step数

            # 新增：把本次成功的推力计数保存下来
            success_thrust_counts.append(thrust_count.copy())

        print(f"Trial {i + 1:3d}: x={x:7.2f}, y={y:7.2f}, distance={dist:7.2f}, outcome={outcome}")

    # 转为NumPy数组方便筛选
    distances = np.array(distances)
    success_dist = distances[distances <= 50]
    crash_dist = distances[distances > 50]

    # 打印统计结果
    print(f"\n总次数: {n_trials}")
    print(f"成功: {len(success_dist)} ({len(success_dist)/n_trials:.1%})")
    print(f"失败: {crash_count} ({crash_count/n_trials:.1%})")
    print(f"平均距离: {np.mean(distances):.3f} m")
    print(f"标准差: {np.std(distances):.3f} m")

    # 绘制双颜色柱状图
    bins = np.linspace(0, max(distances) + 10, 20)

    plt.figure(figsize=(9, 5))
    # 不手动指定颜色，仅用不同数据区分（规范要求）
    plt.hist(success_dist, bins=bins, alpha=0.8, label='Success (≤50 m)')
    plt.hist(crash_dist, bins=bins, alpha=0.8, label='Crash (>50 m)')
    plt.axvline(50, linestyle='--', linewidth=2, label='Crash threshold (50 m)')

    plt.xlabel('Distance from Target Center (m)')
    plt.ylabel('Frequency')
    plt.title(f'Landing Distance Distribution of {43000} episodes ({n_trials} trials)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -----------------------
    # 绘制成功着陆所需 step 的分布
    # -----------------------
    if len(success_steps) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(success_steps, bins=20, alpha=0.8)
        plt.xlabel("Steps used in successful landing")
        plt.ylabel("Frequency")
        plt.title("Distribution of steps for successful landings")
        plt.grid(alpha=0.3)
        plt.show()
    else:
        print("No successful landings, cannot plot step distribution.")


    # -----------------------
    # 统计所有成功落地的推力占比
    # -----------------------
    if len(success_thrust_counts) > 0:
        # 成功落地中动作 0-8 的平均使用次数
        num_actions = 9  # 如果你的动作表长度是 9

        action_avg = []
        for a in range(num_actions):
            arr = np.array([d[a] for d in success_thrust_counts])  # 每次成功中动作 a 的次数
            action_avg.append(arr.mean())

        print("\n平均动作使用情况（成功落地）:")
        for a in range(num_actions):
            print(f"动作 {a} 平均 step: {action_avg[a]:.1f}")

        # 画图
        plt.figure(figsize=(7, 5))
        plt.bar([f"Action {i}" for i in range(num_actions)], action_avg)
        plt.ylabel("Average Steps used")
        plt.title("Average thrust usage per successful landing")
        plt.grid(alpha=0.3)
        plt.show()

    else:
        print("没有成功案例，无法绘图")