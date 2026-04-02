---
title: Snake OpenEnv
emoji: 🐍
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---

# 🐍 SnakeEnv — OpenEnv Hackathon Submission

A classic Snake Game implemented as a fully compliant Gymnasium RL environment.

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/info` | Environment metadata |
| POST | `/reset` | Reset environment |
| POST | `/step` | Take action (0=UP 1=DOWN 2=LEFT 3=RIGHT) |
| POST | `/render` | Render current state |

## Quick Start
```python
from environment import SnakeEnv
env = SnakeEnv(grid_size=10, max_steps=500)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(3)
```