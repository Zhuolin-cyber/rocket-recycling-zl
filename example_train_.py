import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob

from datetime import datetime
import json

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

task = 'landing'  # 'hover' or 'landing'

## 创建训练目录根路径
TRAIN_ROOT = "./training_runs"
os.makedirs(TRAIN_ROOT, exist_ok=True)

## 选择是继续训练还是新建训练
# 获取所有训练目录
existing_runs = sorted([d for d in os.listdir(TRAIN_ROOT)
                        if os.path.isdir(os.path.join(TRAIN_ROOT, d))])

print(f"\n=== 训练模式选择 当前任务: {task} ===")
print("已存在的训练目录：")
for idx, run in enumerate(existing_runs):
    print(f"{idx}: {run}")

print("\n输入数字选择继续训练，或输入 N 创建新训练：")
choice = input("你的选择: ").strip()

if choice.lower() == "n" or len(existing_runs) == 0:
    run_name = f"{task}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_folder = os.path.join(TRAIN_ROOT, run_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    last_ckpt = None
else:
    idx = int(choice)
    run_name = existing_runs[idx]
    ckpt_folder = os.path.join(TRAIN_ROOT, run_name)
    ckpts = sorted(glob.glob(os.path.join(ckpt_folder, "*.pt")))
    last_ckpt = ckpts[-1] if ckpts else None

# 新增：子文件夹，用于保存对比分析需要的数据
action_folder = os.path.join(ckpt_folder, "actions")
mass_folder = os.path.join(ckpt_folder, "mass")
os.makedirs(action_folder, exist_ok=True)
os.makedirs(mass_folder, exist_ok=True)


if __name__ == '__main__':

    max_m_episode = 800000
    max_steps = 800

    env = Rocket(task=task, max_steps=max_steps)
    # ckpt_folder = os.path.join('./', task + '_ckpt')
    # if not os.path.exists(ckpt_folder):
    #     os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    # if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
    #     # load the last ckpt
    #     checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
    #     net.load_state_dict(checkpoint['model_G_state_dict'])
    #     last_episode_id = checkpoint['episode_id']
    #     REWARDS = checkpoint['REWARDS']
    if last_ckpt is not None:
        checkpoint = torch.load(last_ckpt)
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']
    else:
        print("未找到模型，将从零开始训练。")

    for episode_id in range(last_episode_id, max_m_episode):

        # training loop
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []

        action_log = []
        mass_log = []

        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(state)
            state, reward, done, _ = env.step(action)

            # ★ 新增：记录 action 和质量
            action_log.append(int(action))
            mass_log.append(state[8]*100)

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            if episode_id % 100 == 1:
                env.render()

            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

            # print(f"step_id: {step_id}, state: {state}")

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        # ★ 保存动作分布、质量变化曲线
        np.save(os.path.join(action_folder, f"actions_{episode_id:08d}.npy"), np.array(action_log))
        np.save(os.path.join(mass_folder, f"mass_{episode_id:08d}.npy"), np.array(mass_log))

        # ★ Episode summary 用于未来对比三种动力学版本
        episode_summary = {
            "episode_id": episode_id,
            "reward": float(np.sum(rewards)),
            "steps": len(action_log),
            "mass_consumed": float(mass_log[0] - mass_log[-1]),
            "action_hist": {str(i): int(action_log.count(i)) for i in set(action_log)}
        }

        with open(os.path.join(ckpt_folder, "episode_summary.jsonl"), "a") as f:
            f.write(json.dumps(episode_summary) + "\n")

        if episode_id % 100 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



