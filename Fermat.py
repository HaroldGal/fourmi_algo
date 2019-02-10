from random import seed
import matplotlib.pyplot as plt
from random import random
import numpy as np

#points doivent être triés 
T = [[0.0,0.0], [0.0,1.0], [1.0,0.0]]
print("Test")
#T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
#T = [[0.0,5.0], [0.0,0.], [2.5, 2.5], [2.0, 5*0.8], [3.5,3.5], [1.0,0.5], [ 1.5,3.5]]

#test code point de fermat - ON a enfaite une seule abeille
def affichage(bees, T, links):
    plt.scatter([i[0] for i in bees], [i[1] for i in bees], s = 50, c = 'blue')
    plt.scatter([i[0] for i in T], [i[1] for i in T], s = 80, c = 'red')
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
    print(distIni)
    P=Pini
    cptMAX=15
    cpt=0
    while True:
        [Qx,Qy,dist] = median(P,Points)
        P=[Qx,Qy]
        print(P)
        if (dist < eps or cpt== cptMAX or dist > distIni):
            print(dist)
            Fermat = [Qx,Qy]
            return Fermat
        cpt=cpt+1



FermatF=geometric_median(T,0.1)
affichage([FermatF], T, links)
