import os
import numpy as np
import pysal as ps
from util import *

class seg_sim(object):
    '''
    Class to run the segregation simulation.
    '''
    def __init__(self, epoch=10000, iterations=100, nx=50, GA=0.425, GB=0.425, threshold=0.3, PS=0.01, PC=1.5):
        '''
        Initialize the simulation object.
        Args:
        epoch (int): number of epochs to run.
        iterations (int): number of iterations during updating.
        nx (int): the width of the sauqre grid world
        GA (float): the proportion of agents in Group A
        GB (float): the proportion of agents in Group B
        threshold (float): threshold to move
        PS (float): ID strength change probability
        PC (float): ID change constant
        '''
        self.epoch = epoch
        self.iterations = iterations
        self.nx = nx
        # The total population is the size of the grid world
        self.npop = nx * nx
        # 2D array store some properties the agents
        # 0-Group ID, 1-Type, 2-Cuurent ID strength, 3-Future ID strength
        self.population = np.zeros((nx*nx, 4))

        # 2D array store the segregation data
        # 0-Total A, 1-Total B, 2-Total E,
        # 3-Initial average ID strength, 4-Ending average ID strength,
        # 5-Threshold
        self.popagg = np.zeros((3, 6))
        
        # Proportion of different agents
        self.GA = GA
        self.GB = GB
        self.GE = 1 - GA - GB

        self.threshold = threshold
        self.PS = PS
        self.PC = PC

        # Randomly assign group ID to agents
        # 0-Group A, 1-Group B, 2-Group E (empty cell/agent)
        self.population[:, 0] = np.random.multinomial(1, [GA, GB, 1-GA-GB], nx*nx).argmax(1)

        # Get the number of empty cell/agent
        self.NGE = np.sum(self.population[:, 0] == 2)

        # Randomly assign type and ID strength to agents
        # Type: 0-Open, 1-Netural, 2-Close
        # ID strength: -1 (weak) to 1 (strong)
        self.population[:, 1] = np.random.randint(0, 3, size=nx*nx)
        self.population[:, 2] = np.random.uniform(-1, 1, size=nx*nx)

        # Initialize the World
        self.world = ps.lat2W(nx, nx, rook=False)
    
    def move_stage(self):

        plot_map(self.population[:, 0], self.nx, 0, True)

if __name__ == '__main__':
    SEG = seg_sim()
    SEG.move_stage()


