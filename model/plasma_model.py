import numpy as np
import random

class PlasmaModel:
    def __init__(self):
        self.target_ip = 1.5
        self.target_beta = 2.4
        self.ip = 1.5
        self.beta = 2.4
        self.ne = 0.82
        self.te = 5.3
        self.q = 3.21
        self.tearing = 0.02
        self.rwm = 0.01
        self.z_pos = 0.0
        self.ip_cmd = 1.5
        self.beta_cmd = 2.4
        self.heating = 5.0
        self.b_vert = 2.0
        self.ip_history = [1.5] * 60
        
    def step(self, action=None):
        if action is not None:
            self._apply_action(action)
        self._evolve_parameters()
        self.ip_history.pop(0)
        self.ip_history.append(self.ip)
        return self.get_state()
    
    def _apply_action(self, action):
        if action == 0:
            self.ip_cmd = min(2.5, self.ip_cmd + 0.05)
        elif action == 1:
            self.ip_cmd = max(0.8, self.ip_cmd - 0.05)
        elif action == 2:
            self.beta_cmd = min(4.2, self.beta_cmd + 0.05)
        elif action == 3:
            self.beta_cmd = max(0.8, self.beta_cmd - 0.05)
        elif action == 4:
            self.heating = min(15.0, self.heating + 0.3)
    
    def _evolve_parameters(self):
        self.ip += (self.ip_cmd - self.ip) * 0.05 + (random.random() - 0.5) * 0.02
        beta_delta = (self.beta_cmd - self.beta) * 0.03
        heat_delta = (self.heating - 5.0) * 0.01
        self.beta += beta_delta + heat_delta + (random.random() - 0.5) * 0.03
        self.te += heat_delta * 2 + (random.random() - 0.5) * 0.1
        if self.heating > 7.0:
            self.ne += 0.02
        else:
            self.ne -= 0.01
        self.ne += (random.random() - 0.5) * 0.02
        self.z_pos += (self.b_vert - 2.0) * 0.02 + (random.random() - 0.5) * 0.01
        self.q = 3.2 + (1.5 - self.ip) * 0.5 - (self.b_vert - 2.0) * 0.3
        ip_error = abs(self.ip - self.target_ip) / self.target_ip
        beta_error = abs(self.beta - self.target_beta) / self.target_beta
        inst_drive = ip_error + beta_error + abs(self.z_pos) * 0.5
        self.tearing = min(0.95, self.tearing + inst_drive * 0.02)
        self.rwm = min(0.8, self.rwm + inst_drive * 0.01)
        self.tearing *= 0.98
        self.rwm *= 0.97
        self.ip = max(0.6, min(2.8, self.ip))
        self.beta = max(0.5, min(4.5, self.beta))
        self.ne = max(0.2, min(2.0, self.ne))
        self.te = max(1.0, min(15.0, self.te))
        self.q = max(1.8, min(6.0, self.q))
        self.z_pos = max(-0.2, min(0.2, self.z_pos))
    
    def get_state(self):
        return np.array([self.ip, self.beta, self.ne, self.te, self.q, self.tearing, self.rwm, self.z_pos])
    
    def calculate_reward(self):
        ip_error = abs(self.ip - self.target_ip) / self.target_ip
        beta_error = abs(self.beta - self.target_beta) / self.target_beta
        inst_penalty = self.tearing * 0.7 + self.rwm * 0.3
        pos_penalty = abs(self.z_pos) * 5
        stability = np.exp(-3 * (ip_error + beta_error)) * (1 - inst_penalty) * 10
        return stability - pos_penalty
    
    def is_disrupted(self):
        return (self.tearing > 0.85 or self.rwm > 0.7 or abs(self.z_pos) > 0.15)
    
    def reset(self, random_init=True):
        if random_init:
            self.ip = 1.2 + random.random() * 0.6
            self.beta = 1.8 + random.random() * 1.2
            self.ne = 0.5 + random.random() * 0.8
            self.te = 3.0 + random.random() * 4.0
            self.q = 2.8 + random.random() * 1.0
            self.tearing = 0.05 + random.random() * 0.1
            self.rwm = 0.02 + random.random() * 0.1
            self.z_pos = (random.random() - 0.5) * 0.1
        else:
            self.ip = 1.5
            self.beta = 2.4
            self.ne = 0.82
            self.te = 5.3
            self.q = 3.21
            self.tearing = 0.02
            self.rwm = 0.01
            self.z_pos = 0.0
        self.ip_cmd = self.ip
        self.beta_cmd = self.beta
        self.heating = 5.0
        self.b_vert = 2.0
        self.ip_history = [self.ip] * 60
        return self.get_state()
    
    def set_controls(self, ip_cmd, beta_cmd, heating, b_vert):
        self.ip_cmd = ip_cmd
        self.beta_cmd = beta_cmd
        self.heating = heating
        self.b_vert = b_vert
