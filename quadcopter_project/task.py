import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self, init_position=None, init_velocity=None, 
        init_angle_velocity=None, runtime=5., target_position=None):
        
        # Simulation
        self.sim = PhysicsSim(init_position, init_velocity, init_angle_velocity, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.target_position = target_position if target_position is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        ''' Reward'''
        reward = 1. - .1*(abs(self.sim.position[:3] - self.target_position)).sum()
        return reward 

    def step(self, rotor_speeds):
        reward = 0
        positions = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            positions.append(self.sim.position)
        next_state = np.concatenate(positions)
        return next_state, reward, done

    def reset(self):
        self.sim.reset()
        state = np.concatenate([self.sim.position] * self.action_repeat) 
        return state