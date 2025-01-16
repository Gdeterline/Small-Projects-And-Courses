import gym

class LunarLandEnv:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')    
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        self.env.render()
        
    def close(self):
        return self.env.close()