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
