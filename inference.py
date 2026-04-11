import os
import json
import urllib.request
from environment import SnakeEnv


def get_env(difficulty="normal"):
    return SnakeEnv(grid_size=10, max_steps=500, difficulty=difficulty)


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
    """Use LLM proxy to decide next action intelligently."""
    try:
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        api_key  = os.environ.get("API_KEY", "dummy-key")

        prompt = f"""You are an expert Snake game AI agent.

Current game state:
- Head position (row, col): {info['head_pos']}
- Food position (row, col): {info['food_pos']}
- Distance to food: {info['distance_to_food']} steps
- Snake length: {info['snake_length']}
- Steps taken: {info['steps']}
- Difficulty: {info['difficulty']}
- Obstacles on board: {info['obstacles']}

Rules:
- Grid is 10x10 (rows 0-9, cols 0-9)
- Avoid walls (outside grid), your own body, and obstacles (X)
- Move toward food to score points

Actions: 0=UP (row-1), 1=DOWN (row+1), 2=LEFT (col-1), 3=RIGHT (col+1)

Think step by step: which action moves closest to food while staying safe?
Reply with ONLY a single digit (0, 1, 2, or 3)."""

        payload = json.dumps({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            action = int(result["choices"][0]["message"]["content"].strip())
            if action not in [0,1,2,3]:
                action = env.action_space.sample()
            return action
    except Exception:
        return env.action_space.sample()


def compute_score(total_reward, foods_eaten, steps_survived, max_steps):
    """Compute score strictly between 0 and 1 using weighted metrics."""
    max_possible_reward = max_steps * 10.0
    reward_norm   = (total_reward + max_possible_reward) / (2 * max_possible_reward)
    food_norm     = min(foods_eaten / 5.0, 1.0)
    survival_norm = steps_survived / max(max_steps, 1)
    raw = 0.5 * reward_norm + 0.3 * food_norm + 0.2 * survival_norm
    return round(max(0.01, min(0.99, raw)), 4)


def run_task(task_name, difficulty, seed, num_steps):
    """Run one full task episode and return (score, steps)."""
    env = get_env(difficulty=difficulty)
    obs, info = reset(env, seed=seed)
    total_reward, steps_survived, foods_eaten = 0.0, 0, 0

    for i in range(num_steps):
        action = get_llm_action(env, obs, info)
        obs, reward, terminated, truncated, info = step(env, action)
        total_reward  += reward
        steps_survived = i + 1
        foods_eaten    = info['score']
        print(f"[STEP] step={i+1} reward={reward:.2f} score={info['score']} "
              f"length={info['snake_length']} distance={info['distance_to_food']}", flush=True)
        if terminated or truncated:
            break

    score = compute_score(total_reward, foods_eaten, steps_survived, num_steps)
    return score, steps_survived


if __name__ == "__main__":

    # ── Task 1: Survival Challenge (Easy) ─────────────────────────────
    # Agent must stay alive as long as possible on a clear board
    print("[START] task=survival_task", flush=True)
    score1, steps1 = run_task("survival_task", difficulty="easy", seed=42, num_steps=30)
    print(f"[END] task=survival_task score={score1} steps={steps1}", flush=True)

    # ── Task 2: Food Collection (Normal) ──────────────────────────────
    # Agent must collect as much food as possible on normal difficulty
    print("[START] task=food_collection_task", flush=True)
    score2, steps2 = run_task("food_collection_task", difficulty="normal", seed=99, num_steps=30)
    print(f"[END] task=food_collection_task score={score2} steps={steps2}", flush=True)

    # ── Task 3: Obstacle Navigation (Hard) ────────────────────────────
    # Agent must navigate around obstacles on a hard board
    print("[START] task=obstacle_navigation_task", flush=True)
    score3, steps3 = run_task("obstacle_navigation_task", difficulty="hard", seed=7, num_steps=30)
    print(f"[END] task=obstacle_navigation_task score={score3} steps={steps3}", flush=True)
