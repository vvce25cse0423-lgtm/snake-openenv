import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import random

class SnakeEnv(gym.Env):
    """
    Snake Game OpenEnv Environment
    ================================
    A classic Snake game implemented as a Gymnasium-compatible RL environment.

    Observation Space:
        Box(0, 3, shape=(grid_size, grid_size), dtype=int)
        0 = empty, 1 = snake body, 2 = snake head, 3 = food

    Action Space:
        Discrete(4) -> 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

    Rewards:
        +10  for eating food
        -10  for hitting wall or self
        +0.1 for surviving each step (encourages longer play)
        -0.01 small time penalty to avoid infinite loops
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    DIRECTION_MAP = {
        0: (-1, 0),   # UP    → row decreases
        1: (1,  0),   # DOWN  → row increases
        2: (0, -1),   # LEFT  → col decreases
        3: (0,  1),   # RIGHT → col increases
    }
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    EMPTY = 0
    BODY  = 1
    HEAD  = 2
    FOOD  = 3

    def __init__(self, grid_size: int = 10, max_steps: int = 500, render_mode: Optional[str] = None):
        super().__init__()
        assert 5 <= grid_size <= 20, "grid_size must be between 5 and 20"
        self.grid_size  = grid_size
        self.max_steps  = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(self.grid_size, self.grid_size),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)

        # State
        self.snake: list  = []
        self.food: tuple  = (0, 0)
        self.direction: int = self.RIGHT
        self.steps: int   = 0
        self.score: int   = 0
        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        mid = self.grid_size // 2
        # Start with length-3 snake in the middle, facing RIGHT
        self.snake     = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = self.RIGHT
        self.steps     = 0
        self.score     = 0

        self._place_food()
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Prevent 180° reversal
        if action == self.OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action

        dr, dc = self.DIRECTION_MAP[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # --- Collision check ---
        terminated = False
        reward     = 0.0

        # Wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            terminated = True
            reward = -10.0
            return self._get_obs(), reward, terminated, False, self._get_info()

        # Self collision (ignore tail since it will move)
        if new_head in self.snake[:-1]:
            terminated = True
            reward = -10.0
            return self._get_obs(), reward, terminated, False, self._get_info()

        # --- Move snake ---
        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 10.0
            self.score += 1
            self._place_food()
            # Don't remove tail — snake grows
        else:
            self.snake.pop()    # Remove tail
            reward = 0.1 - 0.01  # survive bonus minus time penalty

        self.steps += 1
        truncated = self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            print(self._render_ansi())

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _place_food(self):
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.snake
        ]
        if empty_cells:
            self.food = random.choice(empty_cells)

    def _get_obs(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for r, c in self.snake[1:]:
            grid[r][c] = self.BODY
        hr, hc = self.snake[0]
        grid[hr][hc] = self.HEAD
        grid[self.food[0]][self.food[1]] = self.FOOD
        self.grid = grid
        return grid.copy()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "score":       self.score,
            "snake_length": len(self.snake),
            "steps":       self.steps,
            "food_pos":    self.food,
            "head_pos":    self.snake[0] if self.snake else None,
        }

    def _render_ansi(self) -> str:
        symbols = {self.EMPTY: ".", self.BODY: "O", self.HEAD: "H", self.FOOD: "F"}
        border  = "+" + "-" * (self.grid_size * 2 - 1) + "+"
        rows    = [border]
        for r in range(self.grid_size):
            row = "|" + " ".join(symbols[self.grid[r][c]] for c in range(self.grid_size)) + "|"
            rows.append(row)
        rows.append(border)
        rows.append(f"Score: {self.score}  Steps: {self.steps}  Length: {len(self.snake)}")
        return "\n".join(rows)

    def _render_rgb(self) -> np.ndarray:
        cell = 20
        size = self.grid_size * cell
        img  = np.zeros((size, size, 3), dtype=np.uint8)
        colors = {
            self.EMPTY: (30,  30,  30),
            self.BODY:  (0,   200, 0),
            self.HEAD:  (0,   255, 100),
            self.FOOD:  (255, 50,  50),
        }
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = colors[self.grid[r][c]]
                img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = color
        return img
