# 🐍 SnakeEnv — OpenEnv Hackathon Submission

A classic **Snake Game** implemented as a fully compliant Gymnasium RL environment.

## Overview

| Property | Value |
|---|---|
| Observation Space | `Box(0, 3, shape=(10,10), dtype=int32)` |
| Action Space | `Discrete(4)` — UP / DOWN / LEFT / RIGHT |
| Reward Range | `[-10, +10]` |
| Max Steps | 500 per episode |

### Observation Encoding
| Value | Meaning |
|---|---|
| 0 | Empty cell |
| 1 | Snake body |
| 2 | Snake head |
| 3 | Food |

### Actions
| Action | Direction |
|---|---|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |

### Rewards
| Event | Reward |
|---|---|
| Eat food | +10.0 |
| Hit wall / self | -10.0 |
| Survive a step | +0.09 |

## Quick Start

```bash
# Install dependencies
pip install gymnasium numpy

# Run grader (validates all checks)
python grader.py

# Start server
python server.py
```

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/info` | Environment metadata |
| POST | `/reset` | Reset environment `{"seed": 42}` |
| POST | `/step` | Take action `{"action": 3}` |
| POST | `/render` | Render current state |

## Example Usage

```python
from environment import SnakeEnv

env = SnakeEnv(grid_size=10, max_steps=500)
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()   # Replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## File Structure
```
snake_env/
├── environment.py   # Core SnakeEnv class
├── server.py        # HTTP server (OpenEnv interface)
├── grader.py        # Automated grader / validator
├── pyproject.toml   # Dependencies
└── README.md        # This file
```
