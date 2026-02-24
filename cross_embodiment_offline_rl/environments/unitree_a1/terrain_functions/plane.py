from pathlib import Path
import numpy as np


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        current_file = Path(__file__)
        robot_dir = current_file.parent.parent
        robot = robot_dir.name
        self.xml_file_name = f"{robot}.xml"
        self.center_height = 0.0
        self.nr_sampled_heights = 1
        self.current_difficulty_level = 0.0
        self.sampled_heights = np.zeros(self.nr_sampled_heights)
    
    def step(self, obs, reward, absorbing, info):
        return
    
    def sample(self):
        return
