import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import random
import math


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    DIRECTION_MAP = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
    EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3

    def __init__(self, grid_size=10, max_steps=500, render_mode=None, difficulty="normal"):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.difficulty = difficulty

        if difficulty == "easy":
            self.max_steps = max_steps * 2
            self.reward_scale = 1.2
        elif difficulty == "hard":
            self.max_steps = max_steps // 2
            self.reward_scale = 1.5
        else:
            self.reward_scale = 1.0

        self.observation_space = spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.snake, self.food, self.direction = [], (0,0), self.RIGHT
        self.steps, self.score, self.prev_distance = 0, 0, 0.0
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid-1), (mid, mid-2)]
        self.direction, self.steps, self.score = self.RIGHT, 0, 0
        self._place_food()
        self.prev_distance = self._manhattan(self.snake[0], self.food)
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action == self.OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action
        dr, dc = self.DIRECTION_MAP[action]
        new_head = (self.snake[0][0] + dr, self.snake[0][1] + dc)

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._get_obs(), -10.0 * self.reward_scale, True, False, self._get_info()
        if new_head in self.snake[:-1]:
            return self._get_obs(), -10.0 * self.reward_scale, True, False, self._get_info()

        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10.0 * self.reward_scale
            if len(self.snake) % 5 == 0:
                reward += 2.0
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
            curr = self._manhattan(self.snake[0], self.food)
            reward = 0.5 if curr < self.prev_distance else -0.4
            reward += 0.1 - 0.01

        self.prev_distance = self._manhattan(self.snake[0], self.food)
        self.steps += 1
        return self._get_obs(), reward, False, self.steps >= self.max_steps, self._get_info()

    def render(self):
        if self.render_mode == "ansi": return self._render_ansi()
        elif self.render_mode == "rgb_array": return self._render_rgb()
        elif self.render_mode == "human": print(self._render_ansi())

    def close(self): pass

    def _manhattan(self, a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _place_food(self):
        empty = [(r,c) for r in range(self.grid_size) for c in range(self.grid_size) if (r,c) not in self.snake]
        if empty: self.food = random.choice(empty)

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for r,c in self.snake[1:]: grid[r][c] = self.BODY
        grid[self.snake[0][0]][self.snake[0][1]] = self.HEAD
        grid[self.food[0]][self.food[1]] = self.FOOD
        self.grid = grid
        return grid.copy()

    def _get_info(self):
        return {"score": self.score, "snake_length": len(self.snake), "steps": self.steps,
                "food_pos": self.food, "head_pos": self.snake[0] if self.snake else None,
                "difficulty": self.difficulty,
                "distance_to_food": self._manhattan(self.snake[0], self.food) if self.snake else 0}

    def _render_ansi(self):
        symbols = {0:".", 1:"O", 2:"H", 3:"F"}
        border = "+" + "-"*(self.grid_size*2-1) + "+"
        rows = [border] + ["|"+" ".join(symbols[self.grid[r][c]] for c in range(self.grid_size))+"|" for r in range(self.grid_size)] + [border]
        rows.append(f"Score:{self.score} Steps:{self.steps} Length:{len(self.snake)} Difficulty:{self.difficulty}")
        return "\n".join(rows)

    def _render_rgb(self):
        cell=20; size=self.grid_size*cell
        img=np.zeros((size,size,3),dtype=np.uint8)
        colors={0:(30,30,30),1:(0,200,0),2:(0,255,100),3:(255,50,50)}
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                img[r*cell:(r+1)*cell,c*cell:(c+1)*cell]=colors[self.grid[r][c]]
        return img
