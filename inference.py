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
    """Use LLM proxy to decide next action."""
    try:
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        api_key  = os.environ.get("API_KEY", "dummy-key")

        prompt = f"""You are an expert Snake game AI agent.
Current game state:
- Head position: {info['head_pos']}
- Food position: {info['food_pos']}
- Snake length: {info['snake_length']}
- Distance to food: {info['distance_to_food']}
- Steps taken: {info['steps']}
- Difficulty: {info['difficulty']}

Strategy: Move toward food while avoiding walls and self-collision.
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

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


def compute_score(total_reward, foods_eaten, steps_survived, max_steps, max_reward):
    """
    Compute a score strictly between 0 and 1.
    Combines reward, food eaten, and survival into one metric.
    """
    reward_score   = (total_reward + max_reward) / (2 * max_reward)
    food_score     = min(foods_eaten / 5.0, 1.0)   # max out at 5 foods
    survival_score = steps_survived / max_steps

    # Weighted combination
    raw = 0.5 * reward_score + 0.3 * food_score + 0.2 * survival_score

    # Strictly clamp to (0, 1)
    return max(0.01, min(0.99, raw))


def run_task(task_name, difficulty, seed, num_steps):
    """Run one task episode and return score strictly between 0 and 1."""
    env = get_env(difficulty=difficulty)
    obs, info = reset(env, seed=seed)

    total_reward    = 0.0
    steps_survived  = 0
    foods_eaten     = 0

    for i in range(num_steps):
        action = get_llm_action(env, obs, info)
        obs, reward, terminated, truncated, info = step(env, action)
        total_reward   += reward
        steps_survived  = i + 1
        foods_eaten     = info['score']

        print(f"[STEP] step={i+1} reward={reward:.2f} score={info['score']} length={info['snake_length']} distance={info['distance_to_food']}", flush=True)

        if terminated or truncated:
            break

    score = compute_score(
        total_reward   = total_reward,
        foods_eaten    = foods_eaten,
        steps_survived = steps_survived,
        max_steps      = num_steps,
        max_reward     = num_steps * 10.0
    )
    return score, steps_survived


if __name__ == "__main__":

    # ── Task 1: Survival Task (Easy) ──────────────────────────────
    # Goal: Stay alive as long as possible on easy difficulty
    print("[START] task=survival_task", flush=True)
    score1, steps1 = run_task(
        task_name  = "survival_task",
        difficulty = "easy",
        seed       = 42,
        num_steps  = 20
    )
    print(f"[END] task=survival_task score={score1:.4f} steps={steps1}", flush=True)

    # ── Task 2: Food Collection Task (Normal) ─────────────────────
    # Goal: Collect as much food as possible on normal difficulty
    print("[START] task=food_collection_task", flush=True)
    score2, steps2 = run_task(
        task_name  = "food_collection_task",
        difficulty = "normal",
        seed       = 99,
        num_steps  = 20
    )
    print(f"[END] task=food_collection_task score={score2:.4f} steps={steps2}", flush=True)

    # ── Task 3: Speed Challenge (Hard) ────────────────────────────
    # Goal: Score points quickly on hard difficulty (fewer steps allowed)
    print("[START] task=speed_challenge_task", flush=True)
    score3, steps3 = run_task(
        task_name  = "speed_challenge_task",
        difficulty = "hard",
        seed       = 7,
        num_steps  = 20
    )
    print(f"[END] task=speed_challenge_task score={score3:.4f} steps={steps3}", flush=True)
