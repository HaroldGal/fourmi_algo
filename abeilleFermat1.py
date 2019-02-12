from random import seed
import matplotlib.pyplot as plt
from random import random, shuffle
import numpy as np
from math import acos, sqrt


def create_abeille(T):
    bees = []
    min_x = min([i[0] for i in T])
    max_x = max([i[0] for i in T])
    min_y = min([i[1] for i in T])
    max_y = max([i[1] for i in T])
    for i in range(len(T)-2):
        bees.append([min_x + random()*(max_x-min_x), min_y + random()*(max_y-min_y)])
    return bees



def affichage(bees, T, links):
    plt.scatter([i[0] for i in bees], [i[1] for i in bees], s = 50, c = 'red')
    plt.scatter([i[0] for i in T], [i[1] for i in T], s = 80, c = 'blue')
    for idx, lien in enumerate(links):
        X0 = bees[idx]
        for X1 in lien:
            if X1 in T or X1 in bees :
                plt.plot([X0[0], X1[0]], [X0[1], X1[1]], c='green')
    plt.show()


def distance(P1,P2):
    return np.sqrt((P1[0]-P2[0])*(P1[0]-P2[0])+(P1[1]-P2[1])*(P1[1]-P2[1]))

#definir la mediane --> 1 point au centre des autres points ( point de fermat )
def median(F,Points):
    W=x=y=0
    dist=0
    for p in Points:
        d=distance(F,p) 
        w=1.0/d
        W = W+ w
        x = x + p[0]*w
        y = y + p[1]*w
        dist=dist+d
    return x/W, y/W, dist


# lancer l'alorithme de recherche de la mediane jusqua ce que la distance au point de fermat ( somme des distances de chaque point ) soit minimal 
def geometric_median(Points,eps):
    n=float(len(Points))
    Pini=[sum(P[0] for P in Points)/n,sum(P[1] for P in T)/n]#point initial : moyenne de tous les points 
    distIni=0 
    for i in range(len(Points)):
        distIni=distIni+distance(Points[i],Pini)
    #print(distIni)
    P=Pini
    cptMAX=15
    cpt=0
    while True:
        [Qx,Qy,dist] = median(P,Points)
        P=[Qx,Qy]
        #print(P)
        if (dist < eps or cpt== cptMAX or dist > distIni):
            #print(dist)
            Fermat = [Qx,Qy]
            return Fermat 
        cpt=cpt+1


def sonar5(bee, bees, NonVisited):
    nearest = []
    X = bees + NonVisited
    # calcul carre des normes
    normes= []
    for i in X:
        normes.append(abs((bee[0]-i[0])**2 + (bee[1]-i[1])**2))
    if (len(normes) > 1):
        for i in range(3):
            idx = normes.index(min(normes))
            normes.pop(idx)
            nearest.append(X.pop(idx))
        for k in range(len(nearest)):
            if (nearest[k] in NonVisited ) :
                NonVisited.remove(nearest[k])
    else : 
        if(X.pop(normes.index(normes[0])) in NonVisited) :
            nearest.append(normes.index(normes[0]))
            NonVisited.remove(nearest[0])
    return nearest, NonVisited


if __name__ == '__main__':
    seed(1)
    #T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
    T = [[0.0,5.0], [0.0,0.], [2.5, 2.5], [2.0, 5*0.8], [3.5,3.5], [1.0,0.5], [ 1.5,3.5]]
    #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]
    bees = create_abeille(T)
    links = [[] for i in range(len(bees))]
    #au depart tous les terminaux sont non visites 
    for i in range(8):
        NonVisited=[]
        for h in range(len(T)):
            NonVisited.append(T[h])
        affichage(bees, T, links)
        for j, bee in enumerate(bees):
            nearest, NonVisited=sonar5(bee, [a for a in bees if a!=bee], NonVisited)
            if(len(nearest) > 0):
                bees[j]=geometric_median(nearest,0.1)
                links[j]=nearest

    print(T)
    affichage(bees, T, links)
    print(bees)