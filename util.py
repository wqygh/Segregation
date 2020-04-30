import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

OUTPUT = 'output'

def plot_map(pop, nx, st, saving=False):
    '''
    Plotting the map.
    Args:
    pop (1D array): the population of group ID to plot
    nx (int): the width of the square grid world
    st (int): the run iteration
    saving (bool): save the figure or not
    Returns:
    img (AxisImage): the map image object
    '''
    # choosing the colors
    cmap = colors.ListedColormap(['red', 'blue', 'grey'])
    # boundaries of color bar
    bounds = [-0.5, 0.5, 1.5, 2.5]
    # reshape the population to the shape of the grid world
    girdmap = pop.reshape((nx, nx))
    # initiate the plot and add the color bar legend
    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()
    img = ax.imshow(girdmap, cmap=cmap)
    plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=np.arange(np.min(girdmap),np.max(girdmap)+1), shrink=0.5)
    # save the plot
    if saving:
        filename = 'map_time_{0}.png'.format(st)
        filepath = os.path.join(OUTPUT, filename)
        plt.savefig(filepath, dpi=200, facecolor='white')
    
    return img

def heatmap_scaler(pop):
    '''
    Helper function to put the ID strength of different group to one scale: -1 to 5.
    Args:
    pop(2D array): the population to plot, 0-Group ID, 1-Type, 2-Current ID strength, 3-Future ID strength
    Returns:
    scaled_pop(2D array): 0-Group, 2-Current ID strength, 2-Future ID strength, ID strength scale: -1 to 5
    '''
    scaled_pop = np.copy(pop[:,[0, 2, 3]])
    scaled_pop[:, 1] = scaled_pop[:, 0] * 4 + scaled_pop[:, 1]
    scaled_pop[:, 2] = scaled_pop[:, 0] * 4 + scaled_pop[:, 2]
    scaled_pop[scaled_pop[:, 0]==2] = 2

    return scaled_pop

def plot_heatmap(pop, nx, st, saving=False):
    '''
    Plotting the heatmap of ID strength.
    Args:
    pop (2D array): the population to plot, 0-Group ID, 1-Type, 2-Current ID strength, 3-Future ID strength
    nx (int): the width of the square grid world
    st (int): the run iteration
    saving (bool): save the figure or not
    Returns:
    img (AxisImage): the map image object
    '''
    # put the ID strength of different group to one scale: -1 to 5
    pop_strength = heatmap_scaler(pop)
    # reshape the sclaed current ID strength
    current_ID = pop_strength[:, 1].reshape((nx, nx))
    # initiate the plot and add the color bar legend
    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()
    img = ax.imshow(current_ID, cmap='RdBu')

    if saving:
        filename = 'heatmap_time_{0}.png'.format(st)
        filepath = os.path.join(OUTPUT, filename)
        plt.savefig(filepath, dpi=200, facecolor='white')

    return img