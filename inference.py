import os
import json
import urllib.request
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


def get_llm_action(env, obs, info):
    try:
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("API_KEY", "dummy-key")

        prompt = f"""You are playing Snake game.
Head position: {info['head_pos']}
Food position: {info['food_pos']}
Snake length: {info['snake_length']}
Choose action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
Reply with ONLY a single digit."""

        payload = json.dumps({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            action = int(result["choices"][0]["message"]["content"].strip())
            if action not in [0, 1, 2, 3]:
                action = env.action_space.sample()
            return action
    except Exception:
        return env.action_space.sample()


if __name__ == "__main__":
    env = get_env()
    obs, info = reset(env, seed=42)
    print("[START] task=snake_game", flush=True)

    total_reward = 0
    for i in range(10):
        action = get_llm_action(env, obs, info)
        obs, reward, terminated, truncated, info = step(env, action)
        total_reward += reward
        print(f"[STEP] step={i+1} reward={reward:.2f} score={info['score']} length={info['snake_length']}", flush=True)
        if terminated or truncated:
            break

    print(f"[END] task=snake_game score={total_reward:.2f} steps={i+1}", flush=True)