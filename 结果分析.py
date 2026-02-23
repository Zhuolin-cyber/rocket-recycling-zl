import torch
from rocket import Rocket
from policy import ActorCritic
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time


# test for git checkout: mass-update
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_one_ckpt(env, net, ckpt_path, tag, n_trials=100, max_steps=800, dist_thresh=50.0, base_seed=0):
    """
    评估单个checkpoint，返回统计结果dict，并返回每条trial的记录（便于后处理）。
    env/net 在外部创建以避免重复构造开销（最小改动）。
    """
    # load checkpoint
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=device)

    # ---------- infer model state dims from checkpoint ----------
    sd = checkpoint['model_G_state_dict']
    # actor.linear1.weight: [hidden, input_dim * 15]
    w_in = sd['actor.linear1.weight'].shape[1]
    assert w_in % 15 == 0, f"Unexpected input width={w_in}, cannot infer state_dims"
    model_state_dims = w_in // 15

    print(f"\n=== START EVAL [{tag}] ===")
    print(f"ckpt: {ckpt_path}")
    print(f"env.state_dims={getattr(env, 'state_dims', None)}, model_state_dims={model_state_dims}, w_in={w_in}")

    # IMPORTANT: create a matching net for this checkpoint (avoid shape mismatch)
    net = ActorCritic(input_dim=model_state_dims, output_dim=env.action_dims).to(device)
    net.load_state_dict(sd)
    net.eval()
    # ------------------------------------------------------------

    distances = []
    crash_count = 0
    success_steps = []
    success_thrust_counts = []
    trials = []

    for i in range(n_trials):
        # -------- optional: make evaluation more comparable --------
        seed_i = base_seed + i
        np.random.seed(seed_i)
        torch.manual_seed(seed_i)
        # ----------------------------------------------------------

        thrust_count = defaultdict(int)
        state = env.reset()
        state = state[:model_state_dims]  # env may output 11-dim; A/B may need 8/9

        final_step = None
        for step_id in range(max_steps):
            action, _, _ = net.get_action(state)

            thrust_level = int(action)
            thrust_count[thrust_level] += 1

            state, reward, done, _ = env.step(action)
            state = state[:model_state_dims]
            if done or env.already_crash:
                final_step = step_id + 1
                break

        # record landing distance
        x = env.state['x']
        y = env.state['y']
        center_x = env.target_x
        center_y = env.target_y
        dist = float(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
        distances.append(dist)

        # outcome
        if env.already_crash or dist > dist_thresh:
            crash_count += 1
            outcome = "crash"
        else:
            outcome = "success"
            if final_step is None:
                final_step = max_steps
            success_steps.append(final_step)
            success_thrust_counts.append(thrust_count.copy())

        trials.append({
            "trial_id": i + 1,
            "seed": seed_i,
            "x": float(x),
            "y": float(y),
            "distance": dist,
            "outcome": outcome,
            "final_step": int(final_step) if final_step is not None else None,
            "thrust_count": dict(thrust_count),
        })

        # progress print: print every 10 trials, or always print crashes
        if (i + 1) % 10 == 0 or outcome == "crash":
            cur_success = (i + 1) - crash_count
            print(f"[{tag}] Trial {i + 1:3d}/{n_trials}: dist={dist:7.2f}, outcome={outcome}, "
                  f"success_so_far={cur_success}, crash_so_far={crash_count}")

    distances_np = np.array(distances, dtype=np.float32)
    success_mask = distances_np <= dist_thresh
    success_dist = distances_np[success_mask]
    crash_dist = distances_np[~success_mask]

    # avg thrust usage for successful trials
    action_avg = None
    if len(success_thrust_counts) > 0:
        num_actions = 9
        action_avg = []
        for a in range(num_actions):
            arr = np.array([d.get(a, 0) for d in success_thrust_counts], dtype=np.float32)
            action_avg.append(float(arr.mean()))

    result = {
        "tag": tag,
        "ckpt_path": ckpt_path,
        "n_trials": n_trials,
        "max_steps": max_steps,
        "dist_thresh": dist_thresh,
        "success": int(success_dist.shape[0]),
        "crash": int(crash_count),
        "success_rate": float(success_dist.shape[0] / n_trials),
        "distance_mean": float(distances_np.mean()),
        "distance_std": float(distances_np.std()),
        "success_steps_mean": float(np.mean(success_steps)) if len(success_steps) > 0 else None,
        "success_steps_std": float(np.std(success_steps)) if len(success_steps) > 0 else None,
        "action_avg_success": action_avg,  # length=9 or None
    }
    return result, trials


if __name__ == '__main__':
    task = 'landing'
    max_steps = 800
    n_trials = 100
    dist_thresh = 50.0

    # --- 1) env + net create once ---
    env = Rocket(task=task, max_steps=max_steps)
    s0 = env.reset()
    print("[debug] env.state_dims =", env.state_dims)
    print("[debug] len(env.reset() state) =", len(s0))

    net = None

    # --- 2) define 3 policies checkpoints (edit filenames here) ---
    ckpt_root = task + "_ckpt_评估"
    ckpts = [
        ("A", os.path.join(ckpt_root, "ckpt_00037801_A.pt")),
        ("B", os.path.join(ckpt_root, "ckpt_00039001_B.pt")),
        ("C", os.path.join(ckpt_root, "ckpt_00040001_C.pt")),
    ]

    # --- 3) results output dir ---
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("results_eval", f"{task}_{run_id}")
    os.makedirs(out_root, exist_ok=True)

    summary = {"task": task, "max_steps": max_steps, "n_trials": n_trials, "dist_thresh": dist_thresh, "models": []}

    for tag, ckpt_path in ckpts:
        tag_dir = os.path.join(out_root, f"policy_{tag}")
        os.makedirs(tag_dir, exist_ok=True)

        result, trials = eval_one_ckpt(
            env=env,
            net=net,
            ckpt_path=ckpt_path,
            tag=tag,
            n_trials=n_trials,
            max_steps=max_steps,
            dist_thresh=dist_thresh,
            base_seed=12345,  # 你可以换成任意固定值
        )

        # save json
        with open(os.path.join(tag_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(os.path.join(tag_dir, "trials.jsonl"), "w", encoding="utf-8") as f:
            for row in trials:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary["models"].append(result)

    # save global summary
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    for r in summary["models"]:
        print(f"Policy {r['tag']}: success_rate={r['success_rate']:.1%}, "
              f"dist_mean={r['distance_mean']:.2f}, dist_std={r['distance_std']:.2f}, "
              f"success_steps_mean={r['success_steps_mean']}")