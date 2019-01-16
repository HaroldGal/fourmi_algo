from numpy import meshgrid, cumsum, array
import matplotlib.pyplot as plt
from random import random, seed
size = 20 
mu = 0.2

def position(v):
	return v[0]*(size+1) + v[1] + 1

def initProba():
	proba = [mu for i in range((size+1)*(size+1))]
	for i in range(size+1):
		proba[i] = 0
		proba[i*(size+1)]= 0
		proba[(i+1)*(size)]= 0
		proba[(size+1)*size + i] = 0
	return proba


def mesh(Steiner_points):
	Ndim = len(Steiner_points[0])

	mini = [500 for i in range(Ndim)]
	maxi = [-500 for i in range(Ndim)]

	for X in Steiner_points:
		for i in range(Ndim):
			if X[i] < mini[i] :
				mini[i] = X[i]
			if X[i] > maxi[i] :
				maxi[i] = X[i]

	MeshVect = []
	for i in range(Ndim):
		h = (maxi[i] - mini[i]) / size
		MeshVect.append([mini[i] + j*h for j in range(size)])
		MeshVect[i].append(maxi[i])

	x, y = meshgrid(MeshVect[0], MeshVect[1])
	return (x,y)

def cheminFourmi(coord_fourmi, proba, term):
	seed(1)

	chemin = [coord_fourmi]

	while(v not in term):
		case = v[0]*(size+1) + v[1] + 1
		possible = [[v[0]+1, v[1]], [v[0]-1, v[1]], [v[0], v[1]+1], [v[0], v[1]-1]]
		Pro = [proba[position(i)] for i in possible]
		P_max = sum([proba[position(i)] for i in possible])
		choix = random()*P_max
		for idx, i in enumerate(cumsum(Pro)):
			if choix <= i : 
				break
		v = possible[idx]
		chemin.append(v)
	return chemin
	

if __name__ == '__main__':
	X = [[0.0,1.0], [0.0,0.],[1.0,0.0]]
	(x, y) = mesh(X)
	edge = initProba()
	print(edge)
	plt.scatter(x, y, s = 10)
	plt.scatter([i[0] for i in X],[i[1] for i in X], s = 80, c = 'red')
	plt.show()