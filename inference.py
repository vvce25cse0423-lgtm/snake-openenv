from environment import SnakeEnv


def get_env():
    return SnakeEnv(grid_size=10, max_steps=500)


def reset(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs, info


def step(env, action):
    obs, reward, terminated, truncated, info = env.step(int(action))
    return obs, reward, terminated, truncated, info


def render(env):
    env._get_obs()
    return env._render_ansi()


if __name__ == "__main__":
    env = get_env()
    obs, info = reset(env, seed=42)
    print("[START] task=snake_game", flush=True)

    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = step(env, action)
        total_reward += reward
        print(f"[STEP] step={i+1} reward={reward:.2f} score={info['score']} length={info['snake_length']}", flush=True)
        if terminated or truncated:
            break

    print(f"[END] task=snake_game score={total_reward:.2f} steps={i+1}", flush=True)