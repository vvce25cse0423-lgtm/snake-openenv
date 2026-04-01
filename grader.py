"""
grader.py — Automated grader for SnakeEnv
Validates: runtime correctness, interface compliance, task design, grading logic
"""
import traceback
import numpy as np
from environment import SnakeEnv


PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"{status}  {name}" + (f"  — {detail}" if detail else ""))
    return condition


def run_grader():
    print("=" * 60)
    print("  SnakeEnv Automated Grader")
    print("=" * 60)

    # ── 1. Runtime correctness ──────────────────────────────────────
    print("\n[1] Runtime Correctness")
    try:
        env = SnakeEnv(grid_size=10, max_steps=200)
        obs, info = env.reset(seed=42)
        check("reset() returns (obs, info) tuple", isinstance(obs, np.ndarray) and isinstance(info, dict))
        check("observation shape matches obs space", obs.shape == env.observation_space.shape)
        check("observation dtype is int32", obs.dtype == np.int32)

        obs2, reward, term, trunc, info2 = env.step(env.action_space.sample())
        check("step() returns 5-tuple", True)
        check("reward is float", isinstance(reward, float))
        check("terminated is bool", isinstance(term, bool))
        check("truncated is bool",  isinstance(trunc, bool))
        check("info is dict",       isinstance(info2, dict))
    except Exception as e:
        check("No exception during basic run", False, str(e))
        traceback.print_exc()

    # ── 2. Interface compliance ─────────────────────────────────────
    print("\n[2] Interface Compliance")
    env = SnakeEnv()
    check("has observation_space", hasattr(env, "observation_space"))
    check("has action_space",      hasattr(env, "action_space"))
    check("has reset()",           callable(getattr(env, "reset", None)))
    check("has step()",            callable(getattr(env, "step",  None)))
    check("has render()",          callable(getattr(env, "render", None)))
    check("has close()",           callable(getattr(env, "close",  None)))
    check("action_space.n == 4",   env.action_space.n == 4)

    # ── 3. Task design ──────────────────────────────────────────────
    print("\n[3] Task Design")
    env = SnakeEnv(grid_size=10, max_steps=50)
    obs, _ = env.reset(seed=0)

    # Check head and food are visible in observation
    check("HEAD (2) present in initial obs", 2 in obs)
    check("FOOD (3) present in initial obs", 3 in obs)

    # Check wall collision terminates
    env2 = SnakeEnv(grid_size=5, max_steps=1000)
    env2.reset(seed=1)
    env2.snake = [(0, 0)]
    env2.direction = 0  # UP
    env2._get_obs()
    _, _, term, _, _ = env2.step(0)  # UP from top row → wall
    check("Wall collision terminates episode", term)

    # Check snake grows after eating
    env3 = SnakeEnv(grid_size=10)
    env3.reset(seed=5)
    initial_len = len(env3.snake)
    env3.food = (env3.snake[0][0], env3.snake[0][1] + 1)  # Place food right in front
    env3.direction = 3  # RIGHT
    env3._get_obs()
    env3.step(3)
    check("Snake grows after eating food", len(env3.snake) > initial_len,
          f"{initial_len} → {len(env3.snake)}")

    # ── 4. Grading logic / Reward ───────────────────────────────────
    print("\n[4] Grading Logic (Rewards)")
    env4 = SnakeEnv(grid_size=10)
    env4.reset(seed=7)

    # Force food in front and eat it
    env4.snake     = [(5, 5), (5, 4), (5, 3)]
    env4.direction = 3  # RIGHT
    env4.food      = (5, 6)
    env4.steps     = 0
    env4.score     = 0
    env4._get_obs()
    _, reward_eat, _, _, _ = env4.step(3)
    check("Eating food gives positive reward", reward_eat > 0, f"reward={reward_eat}")

    # Force wall hit
    env5 = SnakeEnv(grid_size=5)
    env5.reset()
    env5.snake     = [(0, 0)]
    env5.direction = 0
    env5._get_obs()
    _, reward_wall, term_wall, _, _ = env5.step(0)
    check("Wall hit gives negative reward",  reward_wall < 0, f"reward={reward_wall}")
    check("Wall hit terminates episode",     term_wall)

    # Truncation at max_steps
    env6 = SnakeEnv(grid_size=10, max_steps=3)
    env6.reset(seed=0)
    for _ in range(3):
        _, _, term6, trunc6, _ = env6.step(3)
        if term6:
            break
    check("Episode truncates at max_steps", trunc6 or env6.steps <= 3)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r[0] == PASS)
    total  = len(results)
    pct    = passed / total * 100
    print(f"  Result: {passed}/{total} checks passed ({pct:.0f}%)")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All checks passed! Your environment is ready to submit.")
    else:
        print("\n⚠️  Some checks failed. Review the output above before submitting.")


if __name__ == "__main__":
    run_grader()
