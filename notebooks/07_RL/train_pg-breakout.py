#!/usr/bin/env python
# coding: utf-8

"""Train an Actor-Critic agent on ALE/Breakout with wandb logging and Adam optimizer (NumPy)."""

import os, time, pickle, secrets
import numpy as np
import gymnasium as gym, ale_py  # registers ALE environments
import wandb

# --------------------------
# Hyperparameters & settings
# --------------------------
num_actions = 4
H = int(os.environ.get("BREAKOUT_H", "512"))
batch_size = int(os.environ.get("BREAKOUT_BATCH_SIZE", "10"))
learning_rate = float(os.environ.get("BREAKOUT_LEARNING_RATE", "1e-3"))
gamma = float(os.environ.get("BREAKOUT_GAMMA", "0.99"))
seed = int(os.environ.get("BREAKOUT_SEED", "42"))
max_grad_norm = float(os.environ.get("BREAKOUT_MAX_GRAD_NORM", "200"))
LOG_INTERVAL = int(os.environ.get("BREAKOUT_LOG_INTERVAL", "50"))
K = int(os.environ.get("BREAKOUT_STEPS", "50000000"))  # environment steps

# === GAE(λ) hyperparam (added) ===
lam = float(os.environ.get("BREAKOUT_LAMBDA", "0.95"))

# ----------------
# Output directory
# ----------------
RUNStamp = time.strftime("%Y%m%d-%H%M%S")
RUN_TAG = os.environ.get(
    "BREAKOUT_RUN_TAG",
    f"br-H{H}-bs{batch_size}-lr{learning_rate:.0e}-lam{lam:g}-gclip{int(max_grad_norm)}",
)
RUN_NUMBER = secrets.randbelow(1_000_000)
OUTPUT_DIR = os.path.join("runs", f"{RUN_TAG}-{RUN_NUMBER:06d}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------
# Weights & Biases
# ----------------
wandb.init(
    project="breakout-actor-critic-numpy-v1",
    name=RUN_TAG,
    config={
        "H": H,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "lambda": lam,
        "seed": seed,
        "max_grad_norm": max_grad_norm,
        "steps": K,
    },
)

# ----------------
# Model definition
# ----------------
D = 80 * 80
model = {
    "W1": (np.random.randn(H, D) / np.sqrt(D)).astype(np.float32),
    "W2": (np.random.randn(num_actions, H) / np.sqrt(H)).astype(np.float32),  # actor head
    "Wv": (np.random.randn(1, H) / np.sqrt(H)).astype(np.float32),            # critic head
}

# Adam optimizer state
grad_buffer = {k: np.zeros_like(v, dtype=np.float32) for k, v in model.items()}
adam_m      = {k: np.zeros_like(v, dtype=np.float32) for k, v in model.items()}
adam_v      = {k: np.zeros_like(v, dtype=np.float32) for k, v in model.items()}
adam_eps = 1e-8
adam_t = 0
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

# ----------------
# Utility functions
# ----------------
def onehot_encoder(action):
    y = np.zeros((num_actions,), dtype=np.float32)
    y[action] = 1.0
    return y

def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)

def prepro(I):
    """
    Preprocess raw Atari frame.

    Input:
        I : np.ndarray, shape (210, 160, 3), dtype=uint8
            Raw RGB frame from the environment.
    Output:
        np.ndarray, shape (6400,), dtype=float32
            Flattened binary image (0.0 or 1.0).
    """
    I = I[35:195, :, 0]        # crop & take R channel
    I = I[::2, ::2]            # downsample by 2x and keep red channel → 80x80
    # in-place binary mask (works because we no longer cast until the end)
    M = (I != 0) # erase background
    out = np.zeros_like(I, dtype=np.float32)
    out[M] = 1.0
    return out.ravel()

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0 # reset of the return at every non-zero reward, don't want that for Breakout (only for Pong)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# === GAE(λ) helper (added) ===
def compute_gae(rewards, values, gamma, lam, done=True, last_v=0.0):
    """
    rewards: (T,), values: (T,)
    returns: (T,), advantages: (T,)
    If `done` is False, bootstrap with last_v for the final step; otherwise use 0.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = values[t+1] if t+1 < T else (0.0 if done else last_v)
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return returns, adv

def policy_forward(x):
    """
    Returns:
      p: action probabilities (A,)
      h: hidden activations (H,)
      v: scalar state-value estimate (float)
    """
    h = np.dot(model["W1"], x)          # (H,)
    h[h < 0] = 0                        # ReLU
    logp = np.dot(model["W2"], h)       # (A,)
    p = softmax(logp)                   # (A,)
    v = float(np.dot(model["Wv"], h))   # scalar
    return p, h, v

def policy_backward(eph, epdlogp, epx):
    """
    Backprop for the policy (actor) part only:
    - eph: (T, H), epdlogp: (T, A), epx: (T, D)
    Returns dict of grads for W1, W2 (actor).
    """
    dW2 = np.dot(epdlogp.T, eph)            # (A,H)
    dh = np.dot(epdlogp, model["W2"])       # (T,H)
    dh[eph <= 0] = 0                        # ReLU backprop
    dW1 = np.dot(dh.T, epx)                 # (H,D)
    return {"W1": dW1, "W2": dW2}

# ----------------
# Environment setup
# ----------------
env = gym.make("ALE/Breakout-v5", render_mode=None)
observation, info = env.reset(seed=seed)
prev_x = None

xs, hs, dlogps, drs, ps, vs = [], [], [], [], [], []
acts = [] # store taken actions for correct loss logging
reward_sum = 0.0
episode_number = 0
running_reward = None
lives = info["lives"] if "lives" in info else 5

# ----------------
# Training loop
# ----------------
for _ in range(K):
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
    prev_x = cur_x

    # Choose an action
    aprob, h, v_est = policy_forward(x)
    action = np.random.choice(range(num_actions), p=aprob)

    xs.append(x)
    hs.append(h)
    y = onehot_encoder(action)
    dlogps.append(y - aprob) # we need this for the gradient calculation since $(y_t - p_t) = \nabla log \pi(a_t|s_t)$
    ps.append(aprob)
    vs.append(v_est)
    acts.append(action)

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # life loss shaping (keep as you had it)
    if "lives" in info and lives - info["lives"] == 1:
        lives = info["lives"]
        reward = -1.0

    drs.append(reward)
    reward_sum += reward

    if done:
        episode_number += 1

        # Stack episode trajectories
        epx = np.asarray(xs, dtype=np.float32)          # (T,D)
        eph = np.asarray(hs, dtype=np.float32)          # (T,H)
        epdlogp = np.asarray(dlogps, dtype=np.float32)  # (T,A)
        epr = np.asarray(drs, dtype=np.float32)         # (T,)
        epp = np.vstack(ps).astype(np.float32)          # (T,A)
        epv = np.asarray(vs, dtype=np.float32)          # (T,)

        # === GAE(λ) returns & advantages (replaces MC returns) ===
        returns, advantages = compute_gae(epr, epv, gamma, lam, done=True)

        # Optional variance reduction (common in A2C):
        adv_std = advantages.std() + 1e-8
        advantages_norm = (advantages - advantages.mean()) / adv_std

        # ---- Losses (for logging only) ----
        T = len(epr)
        acts_arr = np.asarray(acts, dtype=int)
        logp_taken = np.log(np.clip(epp[np.arange(T), acts_arr], 1e-8, 1.0))
        policy_loss = -np.mean(logp_taken * advantages_norm)

        # --- Huber value loss (delta=1.0) ---
        delta = 1.0
        err = returns - epv  # <-- bootstrapped target via GAE
        abs_err = np.abs(err)
        huber = np.where(abs_err <= delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
        value_loss = float(np.mean(huber))

        # ---- Gradients ----
        # Actor: scale log-prob grads by *normalized* advantages
        epdlogp *= advantages_norm[:, None]
        grad_actor = policy_backward(eph, epdlogp, epx)  # dW1_actor, dW2

        # Critic (Huber):
        # dL/dV = -huber_grad, where huber_grad = clip(err, -delta, delta)
        huber_grad = np.where(abs_err <= delta, err, delta * np.sign(err)).astype(np.float32)

        # Ascent-compatible grads: g_ascent = -∂L/∂θ
        # For Wv: ∂L/∂Wv = -(huber_grad) * h^T  ⇒ g_ascent = huber_grad * h^T
        gWv = np.dot(huber_grad.reshape(1, -1), eph)  # (1,T) @ (T,H) -> (1,H)

        # Contribution to W1 through the value head:
        # ascent grad wrt h: huber_grad * Wv
        dh_critic = huber_grad[:, None] * model["Wv"]  # (T,1)*(1,H)->(T,H)
        dh_critic[eph <= 0] = 0                        # ReLU backprop
        dW1_critic = np.dot(dh_critic.T, epx)          # (H,T) @ (T,D) -> (H,D)

        # Accumulate into buffers
        grad_buffer["W1"] += grad_actor["W1"] + dW1_critic
        grad_buffer["W2"] += grad_actor["W2"]
        grad_buffer["Wv"] += gWv.astype(np.float32)

        # Clear episode buffers
        xs, hs, dlogps, drs, ps, vs, acts = [], [], [], [], [], [], []

        # Adam step every batch
        if episode_number % batch_size == 0:
            total_sq = sum(np.sum(g * g) for g in grad_buffer.values())
            total_norm = float(np.sqrt(total_sq))
            clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-8))

            adam_t += 1
            for k in model:
                g = grad_buffer[k] * clip_coef
                adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * g
                adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * (g * g)
                mhat = adam_m[k] / (1 - beta1 ** adam_t)
                vhat = adam_v[k] / (1 - beta2 ** adam_t)
                model[k] += learning_rate * mhat / (np.sqrt(vhat) + adam_eps)
                grad_buffer[k].fill(0.0)

            if episode_number % (batch_size * LOG_INTERVAL) == 0:
                wandb.log(
                    {"grad_norm": total_norm, "clip_coef": clip_coef, "optimizer_step": adam_t},
                    step=episode_number,
                )

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if episode_number % LOG_INTERVAL == 0:
            wandb.log(
                {
                    "episode": episode_number,
                    "ep_reward": reward_sum,
                    "running_mean_reward": running_reward,
                    "policy_loss": float(policy_loss),
                    "value_loss": float(value_loss),
                },
                step=episode_number,
            )

        # reset for next episode
        observation, info = env.reset()
        prev_x = None
        reward_sum = 0.0
        lives = info["lives"] if "lives" in info else 5

        if episode_number % 1000 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"weights_ep{episode_number}.pkl")
            with open(ckpt_path, "wb") as checkpoint_file:
                pickle.dump(model, checkpoint_file)
            print(f"[Checkpoint] Saved locally: {ckpt_path}")

# ----------------
# Cleanup
# ----------------
env.close()
wandb.finish()
print("Training complete.")
