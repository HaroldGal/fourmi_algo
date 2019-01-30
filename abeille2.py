from random import seed
import matplotlib.pyplot as plt
from random import random

def create_abeille(T):
    bees = []
    min_x = min([i[0] for i in T])
    max_x = max([i[0] for i in T])
    min_y = min([i[1] for i in T])
    max_y = max([i[1] for i in T])
    for i in range(len(T)-2):
        bees.append([min_x + random()*(max_x-min_x), min_y + random()*(max_y-min_y)])
    return bees

def sonar(bee, bees, T):
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

def deplacement2(bee, proche):
    #bee = [sum([i[0] for i in proche])/3.0, sum([i[1] for i in proche])/3.0]
    eps=0.001
    alpha=0.002
    #f=fobj(bee,proche)
    grad=gradobj(bee,proche)
    ngrad=grad[0]**2 + grad[1]**2
    while ngrad>=eps:
    	bee[0]=bee[0] - alpha*grad[0]
    	bee[1]=bee[1] - alpha*grad[1]
    	grad=gradobj(bee,proche)
    	ngrad=grad[0]**2 + grad[1]**2
    return bee

def deplacement(bee, proche):
    bee = [sum([i[0] for i in proche])/3.0, sum([i[1] for i in proche])/3.0]
    return bee

def affichage(bees, T, links):
    plt.scatter([i[0] for i in bees], [i[1] for i in bees], s = 50, c = 'blue')
    plt.scatter([i[0] for i in T], [i[1] for i in T], s = 80, c = 'red')
    for idx, lien in enumerate(links):
        X0 = bees[idx]
        for X1 in lien:
            if X1 in T or X1 in bees :
                plt.plot([X0[0], X1[0]], [X0[1], X1[1]], c='green')
    plt.show()

if __name__ == '__main__':
    seed(1)
    T = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
    #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]
    bees = create_abeille(T)
    links = [[] for i in range(len(bees))]
    for i in range(5):
        affichage(bees, T, links)
        for j, bee in enumerate(bees):
            nearest = sonar(bee, [a for a in bees if a!=bee], T)
            bees[j] = deplacement2(bee, nearest)
links[j] = nearest