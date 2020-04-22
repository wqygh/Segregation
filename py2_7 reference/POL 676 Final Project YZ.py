import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pysal as ps 
import os


os.chdir("C:\\Simulation") # change the directory later

def plotit(pop, nx, t):
    cmap = colors.ListedColormap(['red', 'blue', 'grey'])
    bounds=[-0.5, 0.5, 1.5, 2.5]
    mymap = pop.reshape(nx,nx)
    img = plt.matshow(mymap, cmap=cmap)
    plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=np.arange(np.min(mymap),np.max(mymap)+1), shrink=0.5)       
    plt.show()
#     filename = "SSM_map_time_%s.png" %t            
#     plt.savefig(filename, dpi = 300, facecolor = 'w', edgecolor = 'w', 
#         orientation = 'portrait', papertype = None, format = None, transparent = False, 
#         bbox_inches = None, pad_inches = 0.01, frameon = None)
# 


nsteps = 10000  # number of interactions in moving
ninter =100 # number of interactions in updating
nx = 50  ## pop size
npop = nx*nx

pop = np.zeros((npop, 4)) # 0-Group ID, 1-Disposition, 2-current ID strength
                          # 3-future ID strength,  
                          
popagg = np.zeros((3,6)) # Matrix for agg. data:0-total A, 1-total B, 2-total E
                         # 3-avg ID strength at beginning,4-avg ID strength at end
                         # 5 -threshold
                         
threshold = 0.3# 0.9 for high, 0.6 for moderate, and 0.3 for low
GA = 0.425 
GB = 0.425  # proportion of Group A member and proportion of Group B member
GE = 1 - GA - GB
x = 0.01 # ID strength updating probability
k = 1.5 # ID strength updating constant



## asign group id to all agnets: 0-GA, 1-GB, 2-Empty
pop[:,0] = np.random.multinomial(1,[GA,GB,GE], npop).argmax(1)  

## total empty cells:
NGE = np.sum(pop[:,0] == 2)



## assign distributive values
for i in range(npop):
    pop[i,1] = np.random.randint(0,3) # typ: 0-open, 1-netural, 2-close

for i in range(npop):        
    pop[i,2] = np.random.uniform(-1,1) # ID strength: -1 = weak, 1 = strong 
    

    
#_________________________Move Stage __________________________________________                                                                  

W = ps.lat2W(nx, nx, rook = False) 


plotit(pop[:,0], nx, 0)

for t in range(nsteps):
    l = np.int(np.random.randint(0, npop, 1)) # location choosing
    if pop[l,0] < 2: # non-empty space
        in_g = pop[l,0]      #in_g as ingroup, out_g as outgroup
        out_g = 1 - in_g
        n_in_g = sum(pop[W.neighbors[l],0] == in_g)
        n_out_g = sum(pop[W.neighbors[l],0] == out_g)
        if n_out_g > 0:  # if outgroup presents
            if n_in_g/float(n_out_g) < threshold:
                move = np.int(np.random.randint(0,NGE,1))
                des = np.where(pop[:,0] == 2 )[0][move]
                pop[des,:] = pop[l,:]   ## Moving Everthing Along
                pop[l,0] = 2

plotit(pop[:,0], nx, t) 

 
    
#-----------------------------Interaction to Update ID Strength -------------------

avgid = []

for inter in range(ninter):  # interact for 1000 times

  for i in range(npop):    # go through the pop
        
    if pop[i,0] < 2: #if not empty
        p1 = i # determine player 1's index  
        nb = 0  # number of total neighbors
           
        for n in range(len(W.neighbors[i])):
            if pop[W.neighbors[i][n],0] < 2: # if not empty
                p2 = W.neighbors[i][n]  # determine player 2's index 
                d = (pop[p1,2]+pop[p2,2])/float(2) # avg. ID Strg
                nb = nb + 1 # caculate one's total neighbors (no empty)
                tid = 0 # temporary id strength updating
                
                if pop[p1,0] == pop[p2,0]:
                    tid = d + k*x # same group: follow both closed pattern
                       
                else: #pop[p1,0] != pop[p2,0]: diff. group, detm. by disp and ID strg
                    if pop[p1,1] == 0 and pop[p2,1] == 0:
                        tid = d - k*x # situation 1: open vs open
                    if pop[p1,1] == 0 and pop[p2,1] == 1:
                        tid = d - k*x # situation 2 : open vs netural 
                    if pop[p1,1] == 0 and pop[p2,0] == 2:
                        tid = d # situation 3: open vs close
                    if pop[p1,1] == 1 and pop[p2,1] == 1:
                        tid = d # situation 4: neutral vs neutral
                    if pop[p1,1] == 1 and pop[p2,1] == 2:
                        tid = d + x # situation 5: neutral vs closed
                    if pop[p1,1] == 1 and pop[p2,1] == 0:
                        tid = d - x # situation 6: neutral vs open
                    if pop[p1,1] == 2 and pop[p2,1] == 0:
                        tid = d # situation 7: closed vs open
                    if pop[p1,1] == 2 and pop[p2,1] == 1:
                        tid = d + k*x # situation 8: closed vs netural 
                    if pop[p1,1] == 2 and pop[p2,1] == 2:
                        tid = d + k*x # situation 9: closed vs closed
                #print tid
        if nb>0:
            pop[p1,2] = tid/float(nb)   ## average the id updating for one round's interaction
  avgid.append(sum(pop[:,2])/float(npop*GE))
# 
# plt.plot(np.arange(len(avgid)),avgid)
# plt.show()
    

           
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
                
                
        
        
