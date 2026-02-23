import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob
import random

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
traj_folder = os.path.join(ckpt_folder, "trajectories")   # 每回合完整轨迹
metrics_folder = os.path.join(ckpt_folder, "metrics")     # update点/统计指标
eval_folder = os.path.join(ckpt_folder, "eval")           # 固定频率评估

os.makedirs(action_folder, exist_ok=True)
os.makedirs(mass_folder, exist_ok=True)
os.makedirs(traj_folder, exist_ok=True)
os.makedirs(metrics_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)


if __name__ == '__main__':

    max_m_episode = 800000
    max_steps = 800

    SAVE_TRAJ_EVERY = 100  # 每10回合保存一次完整轨迹，避免磁盘爆炸
    SAVE_ACTION_MASS_EVERY = 100

    env = Rocket(task=task, max_steps=max_steps)
    # ckpt_folder = os.path.join('./', task + '_ckpt')
    # if not os.path.exists(ckpt_folder):
    #     os.mkdir(ckpt_folder)

    # ===== Reproducibility (论文对比必须) =====
    SEED = 12345
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ===== Run metadata snapshot =====
    run_meta = {
        "run_name": run_name,
        "task": task,
        "time": datetime.now().isoformat(),
        "device": str(device),
        "seed": SEED,
        "max_steps": max_steps,
        "gamma": 0.999,  # 与质心版对齐
        "env": {
            "state_dims": getattr(env, "state_dims", None),
            "action_dims": getattr(env, "action_dims", None),
            "H_sim": getattr(env, "H", None),
            # 无质心版通常没有以下属性，getattr 会给 None，不影响保存
            "use_com_lut": getattr(env, "use_com_lut", None),
            "H_lut": getattr(env, "H_lut", None),
            "com_scale": getattr(env, "com_scale", None),
            "com_dir": getattr(env, "_com_dir", None),
            "theta_grid_range": [
                float(env.theta_grid[0]), float(env.theta_grid[-1])
            ] if getattr(env, "theta_grid", None) is not None else None,
            "mass_grid_range": [
                float(env.mass_grid[0]), float(env.mass_grid[-1])
            ] if getattr(env, "mass_grid", None) is not None else None,
            "M0_from_grid": getattr(env, "M0", None),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    }
    with open(os.path.join(ckpt_folder, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

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

        state_log = []
        reward_log = []
        done_log = []
        value_log = []
        logprob_log = []
        last_info = None


        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(state)
            state, reward, done, _ = env.step(action)

            # ===== 无质心版没有 info，这里用 env 属性 + 当前 state/reward 手动构造 =====
            vx = float(state[2] * 100.0)
            vy = float(state[3] * 100.0)
            v = float((vx * vx + vy * vy) ** 0.5)

            info = {
                "crash": int(bool(getattr(env, "already_crash", False))),
                "landing": int(bool(getattr(env, "already_landing", False))),
                "fuel_empty": int(bool(getattr(env, "already_fuel_empty", False))),
                "step_id": int(getattr(env, "step_id", step_id)),
                "last_reward": float(reward),
                "last_v": v,
                "last_vx": vx,
                "last_vy": vy,
            }
            last_info = info

            # ★ 新增：记录 action 和质量
            action_log.append(int(action))
            mass_log.append(state[8]*100)

            # ★ 记录完整轨迹（用于论文复盘）
            state_log.append(state.copy())
            reward_log.append(float(reward))
            done_log.append(int(done))
            value_log.append(float(value.detach().cpu().item()))
            logprob_log.append(float(log_prob.detach().cpu().item()))

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            if episode_id % 100 == 1:
                env.render()

            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)

                update_meta = {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "done": int(done),
                    "Qval": float(Qval.detach().cpu().item()),
                    "traj_len": int(len(rewards)),
                }

                if last_info is not None:
                    update_meta.update({
                        "crash": int(bool(last_info.get("crash", False))),
                        "landing": int(bool(last_info.get("landing", False))),
                        "fuel_empty": int(bool(last_info.get("fuel_empty", False))),
                        "env_step_id": int(last_info.get("step_id", -1)),
                        "last_reward": float(last_info.get("last_reward", 0.0)),
                        "last_v": float(last_info.get("last_v", 0.0)),
                        "last_vx": float(last_info.get("last_vx", 0.0)),
                        "last_vy": float(last_info.get("last_vy", 0.0)),
                    })

                with open(os.path.join(metrics_folder, "update_points.jsonl"), "a") as f:
                    f.write(json.dumps(update_meta) + "\n")

                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

            # print(f"step_id: {step_id}, state: {state}")

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        # ★ 保存动作分布、质量变化曲线
        np.save(os.path.join(action_folder, f"actions_{episode_id:08d}.npy"), np.array(action_log))
        np.save(os.path.join(mass_folder, f"mass_{episode_id:08d}.npy"), np.array(mass_log))

        # ★ 保存每回合完整轨迹（强烈建议用于论文）
        if episode_id % SAVE_TRAJ_EVERY == 0:
            np.savez_compressed(
                os.path.join(traj_folder, f"traj_{episode_id:08d}.npz"),
                states=np.array(state_log, dtype=np.float32),
                actions=np.array(action_log, dtype=np.int64),
                rewards=np.array(reward_log, dtype=np.float32),
                dones=np.array(done_log, dtype=np.int8),
                values=np.array(value_log, dtype=np.float32),
                log_probs=np.array(logprob_log, dtype=np.float32),
            )

        # ★ Episode summary 用于未来对比三种动力学版本
        episode_summary = {
            "episode_id": episode_id,
            "reward": float(np.sum(rewards)),
            "steps": len(action_log),
            "mass_consumed": float(mass_log[0] - mass_log[-1]),
            "action_hist": {str(i): int(action_log.count(i)) for i in set(action_log)}
        }

        # ===== derived metrics for paper =====
        states_arr = np.array(state_log, dtype=np.float32)

        theta_series = states_arr[:, 4] if len(states_arr) else np.array([])
        m_series = states_arr[:, 8] if (len(states_arr) and states_arr.shape[1] > 8) else np.array([])

        action_switches = int(np.sum(np.array(action_log[1:]) != np.array(action_log[:-1]))) if len(
            action_log) > 1 else 0
        theta_max_abs = float(np.max(np.abs(theta_series))) if len(theta_series) else 0.0
        theta_rms = float(np.sqrt(np.mean(theta_series ** 2))) if len(theta_series) else 0.0

        # 无质心版一般 state 维度 < 11，因此这里会保持 None
        xc_range = None
        zc_range = None
        if len(states_arr) and states_arr.shape[1] >= 11:
            xc_series = states_arr[:, 9]
            zc_series = states_arr[:, 10]
            xc_range = float(np.max(xc_series) - np.min(xc_series))
            zc_range = float(np.max(zc_series) - np.min(zc_series))

        episode_summary.update({
            "reward_per_step": float(np.sum(reward_log) / max(1, len(reward_log))) if len(reward_log) else None,
            "action_switches": action_switches,
            "theta_max_abs": theta_max_abs,
            "theta_rms": theta_rms,
            "m_min": float(np.min(m_series)) if len(m_series) else None,
            "m_final": float(m_series[-1]) if len(m_series) else None,
            "value_mean": float(np.mean(value_log)) if len(value_log) else None,
            "value_std": float(np.std(value_log)) if len(value_log) else None,
            "xc_range": xc_range,
            "zc_range": zc_range,
        })

        if last_info is not None:
            episode_summary.update({
                "crash": int(bool(last_info.get("crash", False))),
                "landing": int(bool(last_info.get("landing", False))),
                "fuel_empty": int(bool(last_info.get("fuel_empty", False))),
                "terminal_step_id_env": int(last_info.get("step_id", -1)),
                "terminal_reward": float(last_info.get("last_reward", 0.0)),
                "terminal_v": float(last_info.get("last_v", 0.0)),
                "terminal_vx": float(last_info.get("last_vx", 0.0)),
                "terminal_vy": float(last_info.get("last_vy", 0.0)),
            })

        with open(os.path.join(ckpt_folder, "episode_summary.jsonl"), "a") as f:
            f.write(json.dumps(episode_summary) + "\n")

        # ===== Evaluation episodes (paper-grade comparison) =====
        EVAL_EVERY = 2000
        N_EVAL = 5

        if episode_id % EVAL_EVERY == 0 and episode_id > last_episode_id:
            eval_rewards = []

            # 1) 保存训练用的 RNG 状态（评估结束后恢复，保证不影响训练）
            py_state = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            try:
                # 2) 为评估设置独立 seed（可复现）
                eval_seed = SEED + 999
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)

                # 3) 使用独立 eval_env，避免动训练 env 的内部状态
                #    这里懒创建：第一次评估时创建一次，后续复用
                if "eval_env" not in locals():
                    eval_env = Rocket(task=task, max_steps=max_steps)

                net.eval()
                with torch.no_grad():
                    for k in range(N_EVAL):
                        s = eval_env.reset()
                        er = 0.0
                        for t in range(max_steps):
                            # 如果 get_action 返回 tensor，确保转换为 Python 标量，避免隐式同步
                            a, _, _ = net.get_action(s)
                            if isinstance(a, torch.Tensor):
                                a = int(a.item())
                            s, r, d, _ = eval_env.step(a)
                            er += float(r)
                            if d:
                                break
                        eval_rewards.append(er)

                net.train()

            finally:
                # 4) 恢复 RNG 状态：评估不会改变后续训练的随机轨迹
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                if torch.cuda.is_available() and cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

            eval_summary = {
                "episode_id": episode_id,
                "eval_n": N_EVAL,
                "eval_reward_mean": float(np.mean(eval_rewards)),
                "eval_reward_std": float(np.std(eval_rewards)),
                "eval_rewards": [float(x) for x in eval_rewards],
            }
            with open(os.path.join(eval_folder, "eval_summary.jsonl"), "a") as f:
                f.write(json.dumps(eval_summary) + "\n")

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



