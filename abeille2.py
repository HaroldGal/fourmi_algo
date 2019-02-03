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

def sonar2(bee, bees, T):
    nearest = []

    X = bees + T
    # calcul carre des normes
    normes= []
    for i in X:
        normes.append(abs((bee[0]-i[0])**2 + (bee[1]-i[1])**2))
    for i in range(3):
        idx = normes.index(min(normes))
        normes.pop(idx)
        nearest.append(X.pop(idx))
    return nearest

def sonar(bee, T):
    nearest = []
    #X=bees+T
    X=T
    # calcul carre des normes
    normes= []
    for i in X:
        normes.append( sqrt((bee[0]-i[0])**2 + (bee[1]-i[1])**2) )

    for i in range(3):
        idx = normes.index(min(normes))
        normes.pop(idx)
        nearest.append(X[idx])
        X.pop(idx)
        
    return nearest

def fobj(bee,proche):
	s=0
	for p in proche :
		s+= (p[0]-bee[0])**2 + (p[1]-bee[1])**2
	return s

def gradobj(bee,proche):
	s = [0, 0]
	for p in proche :
		s[0]+=2*(bee[0]-p[0])
		s[1]+=2*(bee[1]-p[1])
	return s

def angles(proche): # p0-p1 = l2// p0-p2=l1 //p1-p2=l0

    l0=sqrt( (proche[2][0]-proche[1][0])**2 + (proche[2][1]-proche[1][1])**2 )
    l1=sqrt( (proche[0][0]-proche[2][0])**2 + (proche[0][1]-proche[2][1])**2 )
    l2=sqrt( (proche[0][0]-proche[1][0])**2 + (proche[0][1]-proche[1][1])**2 )
    eps=0.0001
    if (l0==0 or l1==0 or l2==0) :
        return -1
    if ( (abs(l0-l1) <=eps) and (abs(l1-l2)<=eps) and (abs(l0-l2)<=eps) ):
        return -2
    A=float(l1**2 + l2**2 - l0**2)/ float(2.0*l1*l2)
    B=float(l0**2 + l2**2 - l1**2)/ float(2.0*l0*l2)
    C=(l1**2 + l0**2 - l2**2)/ (2.0*l1*l0)
    Z=[A,B,C]
    for i in Z:
        if abs(float(i))<=1.0:
            if float(i)<=-1:
                i=-1.0
            else:
                i=1.0  
    a0= acos(Z[0])
    a1= acos(Z[1])
    a2= acos(Z[2])
    
    if (a0 >=2*np.pi/3):
        return 0
    if (a1 >=2*np.pi/3):
        return 1
    if (a2 >=2*np.pi/3):
        return 2
    return -1

def deplacement2(bee, proche):
    #teste angle >=120
    ind=angles(proche)
    if (ind >-1) :
        bee=proche[ind]
        return bee
    #teste triangle equilateral
    if (ind==-2) :
        return deplacement(bee,proche)
    # sinon descente gradient 
    eps=0.000001
    alpha=0.0002
    grad=gradobj(bee,proche)
    f=5000
    f2=fobj(bee,proche)
    while abs(f-f2)>=eps:
        bee[0]=bee[0] - alpha*grad[0]
        bee[1]=bee[1] - alpha*grad[1]
        grad=gradobj(bee,proche)
        f=f2
        f2=fobj(bee,proche)
    return bee

def deplacement(bee, proche):
    bee = [sum([i[0] for i in proche])/3.0, sum([i[1] for i in proche])/3.0]
    return bee

def affichage(bees, T, links):
    plt.scatter([i[0] for i in bees], [i[1] for i in bees], s = 50, c = 'red')
    plt.scatter([i[0] for i in T], [i[1] for i in T], s = 80, c = 'blue')
    for idx, lien in enumerate(links):
        X0 = bees[idx]
        for X1 in lien:
            if X1 in T or X1 in bees :
                plt.plot([X0[0], X1[0]], [X0[1], X1[1]], c='green')
    plt.show()

def longueur(links): # pas bon redondance aretes
    somme = 0
    for j in links:
        for i in j:
            somme+= sqrt(i[0]**2+i[1]**2)
    return somme

if __name__ == '__main__':
    seed(1)
    T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
    #T = [[0.0,5.0], [0.0,0.], [2.5, 2.5], [2.0, 5*0.8], [3.5,3.5], [1.0,0.5], [ 1.5,3.5]]
    #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]
    bees = create_abeille(T)
    links = [[] for i in range(len(bees))]
    for i in range(8):
        #affichage(bees, T, links)

        for j, bee in enumerate(bees):
            #T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
            #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]
            nearest = sonar2(bee, [a for a in bees if a!=bee], T)
            #nearest = sonar(bee, T)
            #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]
            #T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]            
            bees[j] = deplacement2(bee, nearest)
            links[j] = nearest
    
    #print(longueur(links))
    affichage(bees, T, links)
    print(bees)