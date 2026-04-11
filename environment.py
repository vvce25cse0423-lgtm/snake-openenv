import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import random


class SnakeEnv(gym.Env):
    """
    Snake Game OpenEnv Environment — Competition Grade v2.0
    Enhanced with difficulty levels, obstacles, distance shaping, RGB rendering.
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    DIRECTION_MAP = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
    OPPOSITE = {0:1, 1:0, 2:3, 3:2}
    EMPTY, BODY, HEAD, FOOD, OBSTACLE = 0, 1, 2, 3, 4

    def __init__(self, grid_size=10, max_steps=500, render_mode=None, difficulty="normal"):
        super().__init__()
        assert 5 <= grid_size <= 20
        assert difficulty in ["easy","normal","hard"]
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.difficulty = difficulty
        cfg = {"easy":{"ms":max_steps*2,"rs":1.0,"obs":False},
               "normal":{"ms":max_steps,"rs":1.2,"obs":False},
               "hard":{"ms":max_steps//2,"rs":1.5,"obs":True}}[difficulty]
        self.max_steps = cfg["ms"]
        self.reward_scale = cfg["rs"]
        self.use_obstacles = cfg["obs"]
        self.observation_space = spaces.Box(low=0,high=4,shape=(grid_size,grid_size),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.snake,self.food,self.obstacles = [],[],[]
        self.direction,self.steps,self.score,self.prev_dist = self.RIGHT,0,0,0.0
        self.grid = np.zeros((grid_size,grid_size),dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: random.seed(seed); np.random.seed(seed)
        mid = self.grid_size//2
        self.snake = [(mid,mid),(mid,mid-1),(mid,mid-2)]
        self.direction,self.steps,self.score = self.RIGHT,0,0
        self.obstacles = []
        if self.use_obstacles:
            forbidden = set(self.snake)
            n = self.grid_size//2
            while len(self.obstacles)<n:
                r,c = random.randint(0,self.grid_size-1),random.randint(0,self.grid_size-1)
                if (r,c) not in forbidden:
                    self.obstacles.append((r,c)); forbidden.add((r,c))
        self._place_food()
        self.prev_dist = self._manhattan(self.snake[0],self.food)
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action==self.OPPOSITE[self.direction]: action=self.direction
        self.direction=action
        dr,dc = self.DIRECTION_MAP[action]
        nh = (self.snake[0][0]+dr, self.snake[0][1]+dc)
        if not(0<=nh[0]<self.grid_size and 0<=nh[1]<self.grid_size) or nh in self.snake[:-1] or nh in self.obstacles:
            return self._get_obs(),-10.0*self.reward_scale,True,False,self._get_info()
        self.snake.insert(0,nh)
        if nh==self.food:
            reward=10.0*self.reward_scale+(2.0 if len(self.snake)%5==0 else 0)
            self.score+=1; self._place_food()
        else:
            self.snake.pop()
            cd=self._manhattan(self.snake[0],self.food)
            reward=(0.5 if cd<self.prev_dist else -0.4)+0.1-0.01
        self.prev_dist=self._manhattan(self.snake[0],self.food)
        self.steps+=1
        return self._get_obs(),reward,False,self.steps>=self.max_steps,self._get_info()

    def render(self):
        if self.render_mode=="ansi": return self._render_ansi()
        elif self.render_mode=="rgb_array": return self._render_rgb()
        elif self.render_mode=="human": print(self._render_ansi())

    def close(self): pass

    def _manhattan(self,a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

    def _place_food(self):
        forbidden=set(self.snake)|set(self.obstacles)
        empty=[(r,c) for r in range(self.grid_size) for c in range(self.grid_size) if (r,c) not in forbidden]
        if empty: self.food=random.choice(empty)

    def _get_obs(self):
        g=np.zeros((self.grid_size,self.grid_size),dtype=np.int32)
        for r,c in self.obstacles: g[r][c]=self.OBSTACLE
        for r,c in self.snake[1:]: g[r][c]=self.BODY
        g[self.snake[0][0]][self.snake[0][1]]=self.HEAD
        g[self.food[0]][self.food[1]]=self.FOOD
        self.grid=g; return g.copy()

    def _get_info(self):
        return {"score":self.score,"snake_length":len(self.snake),"steps":self.steps,
                "food_pos":self.food,"head_pos":self.snake[0] if self.snake else None,
                "difficulty":self.difficulty,
                "distance_to_food":int(self._manhattan(self.snake[0],self.food)) if self.snake else 0,
                "obstacles":len(self.obstacles)}

    def _render_ansi(self):
        sym={0:".",1:"O",2:"H",3:"F",4:"X"}
        b="+"+ "-"*(self.grid_size*2-1)+"+"
        rows=[b]+["|"+" ".join(sym[self.grid[r][c]] for c in range(self.grid_size))+"|" for r in range(self.grid_size)]+[b]
        rows.append(f"Score:{self.score} Steps:{self.steps} Length:{len(self.snake)} Difficulty:{self.difficulty} Obstacles:{len(self.obstacles)}")
        return "\n".join(rows)

    def _render_rgb(self):
        cell=24; size=self.grid_size*cell
        img=np.zeros((size,size,3),dtype=np.uint8)
        colors={0:(20,20,20),1:(0,180,0),2:(0,255,120),3:(255,60,60),4:(150,150,150)}
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                img[r*cell:(r+1)*cell,c*cell:(c+1)*cell]=colors[self.grid[r][c]]
                img[r*cell,c*cell:(c+1)*cell]=(40,40,40)
                img[r*cell:(r+1)*cell,c*cell]=(40,40,40)
        return img
