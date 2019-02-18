from random import seed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random
import numpy as np



def create_abeille(T):
    bees = []
    min_x = min([i[0] for i in T])
    max_x = max([i[0] for i in T])
    min_y = min([i[1] for i in T])
    max_y = max([i[1] for i in T])
    min_z = min([i[2] for i in T])
    max_z = max([i[2] for i in T])
    for i in range(len(T)-2):
        bees.append([min_x + random()*(max_x-min_x), min_y + random()*(max_y-min_y), min_z + random()*(max_z-min_z)])
    return bees

def sonar(bee, bees, T):
    nearest = []

    X = bees + T
    # calcul carre des normes
    normes= []
    for i in X:
        normes.append(abs((bee[0]-i[0])**2 + (bee[1]-i[1])**2 + (bee[2]-i[2])**2))
    for i in range(3):
        idx = normes.index(min(normes))
        normes.pop(idx)
        nearest.append(X.pop(idx))
    return nearest

def deplacement(bee, proche):
    bee = [sum([i[0] for i in proche])/3.0, sum([i[1] for i in proche])/3.0, sum([i[2] for i in proche])/3.0]
    return bee

def affichage(bees, T, links):

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter([i[0] for i in bees], [i[1] for i in bees],[i[2] for i in bees], s = 50, c = 'blue')
    ax.scatter([i[0] for i in T], [i[1] for i in T],[i[2] for i in T], s = 80, c = 'red')
    for idx, lien in enumerate(links):
        X0 = bees[idx]
        for X1 in lien:
            if X1 in T or X1 in bees :
                plt.plot([X0[0], X1[0]], [X0[1], X1[1]],[X0[2], X1[2]], c='green')
    plt.show()

if __name__ == '__main__':
    seed(1)
    #T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
    T = [[0.0,0.0,0.0], [0.0,1.0,0.0], [1.0,1.0,0.0], [1.0,0.0,0.0]]
    #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]

    #T = [[1,0]]
    #for i in range(1,5):
    #    T.append([np.cos(2*i*np.pi/5), np.sin(2*i*np.pi/5)])
    bees = create_abeille(T)
    links = [[] for i in range(len(bees))]
    for i in range(5):
        affichage(bees, T, links)
        for j, bee in enumerate(bees):
            nearest = sonar(bee, [a for a in bees if a!=bee], T)
            bees[j] = deplacement(bee, nearest)
            links[j] = nearest
