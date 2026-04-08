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


def run_task(task_name, seed, steps):
    """Run one task and return score strictly between 0 and 1."""
    env = get_env()
    obs, info = reset(env, seed=seed)
    total_reward = 0
    max_possible = steps * 10.0
    completed_steps = 0

    for i in range(steps):
        action = get_llm_action(env, obs, info)
        obs, reward, terminated, truncated, info = step(env, action)
        total_reward += reward
        completed_steps = i + 1
        print(f"[STEP] step={i+1} reward={reward:.2f} score={info['score']} length={info['snake_length']}", flush=True)
        if terminated or truncated:
            break

    # Score strictly between 0 and 1
    raw_score = (total_reward + max_possible) / (2 * max_possible)
    # Clamp to strictly (0, 1)
    score = max(0.01, min(0.99, raw_score))
    return score, completed_steps


if __name__ == "__main__":
    print("[START] task=survival_task", flush=True)
    score1, steps1 = run_task("survival_task", seed=42, steps=10)
    print(f"[END] task=survival_task score={score1:.4f} steps={steps1}", flush=True)

    print("[START] task=food_collection_task", flush=True)
    score2, steps2 = run_task("food_collection_task", seed=99, steps=10)
    print(f"[END] task=food_collection_task score={score2:.4f} steps={steps2}", flush=True)

    print("[START] task=navigation_task", flush=True)
    score3, steps3 = run_task("navigation_task", seed=7, steps=10)
    print(f"[END] task=navigation_task score={score3:.4f} steps={steps3}", flush=True)