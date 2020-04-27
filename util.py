import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

OUTPUT = 'output'

def plot_map(pop, nx, st, saving=False):
    '''
    Plotting the map.
    Args:
    pop (np array): the population to plot
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
    