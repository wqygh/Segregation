import os
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import lat2W
from util import *

class seg_sim(object):
    '''
    Class to run the segregation simulation.
    '''
    def __init__(self, n_move=10000, n_interact=1000, nx=50, GA=0.425, GB=0.425, threshold=0.3, PS=0.01, CS=1.5):
        '''
        Initialize the simulation object.
        Args:
        n_move (int): number of moves to run
        n_interact (int): number of interactions to run
        nx (int): the width of the sauqre grid world
        GA (float): the proportion of agents in Group A
        GB (float): the proportion of agents in Group B
        threshold (float): threshold to move
        PS (float): ID strength change probability
        CS (float): ID strength change constant
        '''
        self.n_move = n_move
        self.n_interact = n_interact
        self.nx = nx
        # Initialize the count of moves and interactions
        self.c_move = 0
        self.c_interact = 0
        # The total population is the size of the grid world
        self.npop = nx * nx
        # 2D array store some properties the agents
        # 0-Group ID, 1-Type, 2-Current ID strength, 3-Future ID strength
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
        self.CS = CS

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
        self.world = lat2W(nx, nx, rook=False)

        # Initialize the average ID strength list
        self.avgS = []
    
    def move_stage(self, show_progress=False, save_figures=True):
        '''
        Moving agents according to their ID strength.
        Args:
        show_progress (bool): whether to show the updating map
        save_figures (bool): whether to save the initial and final map
        Returns:
        Nono
        '''

        pop = self.population
        W = self.world
        nx = self.nx

        _ = plot_map(pop[:, 0], nx, 0, saving=save_figures)
        img = plot_heatmap(pop, nx, 0, saving=save_figures)

        for t in range(self.n_move):
            # Randomly pick an agent
            loc = np.random.randint(0, self.npop)
            # For a not empty agent
            if pop[loc, 0] < 2:
                # Get the within group ID and outside group ID
                in_g = pop[loc, 0]
                out_g = 1 - in_g
                # Count the numbher of within group and outside group neighbors
                n_in_g = np.sum(pop[W.neighbors[loc], 0] == in_g)
                n_out_g = np.sum(pop[W.neighbors[loc], 0] == out_g)
                # When ther ratio of within group and outside group neighbors us lower than the threshold
                if n_out_g > 0 and n_in_g / n_out_g < self.threshold:
                    # Randomly pick an empty cell/agent
                    mv = np.random.randint(0, self.NGE)
                    des = np.where(pop[:,0] == 2)[0][mv]
                    # Move the current agent
                    pop[des, :] = pop[loc, :]
                    pop[loc, 0] = 2
                    
                    if show_progress:
                        scaled_pop_strength = heatmap_scaler(pop)
                        img.set_data(scaled_pop_strength[:, 1].reshape(nx, nx))
                        plt.draw(), plt.pause(0.05)
    
        self.c_move += 1
        _ = plot_map(pop[:, 0], nx, 'after_{0}_moves_{1}'.format(self.n_move, self.c_move), saving=save_figures)
        _ = plot_heatmap(pop, nx, 'after_{0}_moves_{1}'.format(self.n_move, self.c_move), saving=save_figures)

    def interact_stage(self):
        ''''''
        
        pop = self.population
        p = self.PS
        c = self.CS
        W = self.world

        # Run the interaction n_interact times
        for interaction in range(self.n_interact):
            # 
            pass
if __name__ == '__main__':
    SEG = seg_sim()
    SEG.move_stage(show_progress=True)


