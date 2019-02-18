from numpy import meshgrid, cumsum, array
import matplotlib.pyplot as plt
from random import random, seed, choice
from collections import Counter
from math import log

###### VAR GLOB #####

size = 10 # taille maillage
mu = 0.2 # pheromone
dissip = 0.5 # dissipation par tour
Nfourmi = 200 # nb fourmi par maj_colonie
Ntour = 20 # nb de tour avec depot de pheromone

#####################

# proba
# size**2 haut, gauche, bas, droite

def norme_proba(Proba):
	mat = [[0 for i in range(size)] for j in range(size)]
	for i in range(size):
		for j in range(size):
			mat[i][j] = sum([Proba[k*size*size + i*size+j] for k in range(4)])
	plt.matshow(mat, origin='lower')
	plt.show()
	#norm = []

def initProba():
	proba = [mu for i in range(4*size*size)] # 4 chemins possible de chaque noeud
	for i in range(size):
		proba[i] = 0 # dessus de la premiere ligne
		proba[i*(size)+ size*size]= 0 # gauche de la premiere colonne
		proba[(i+1)*(size) - 1 + 3*size*size]= 0 # droite de la derniere colonne
		proba[(size-1)*size + i +2*size*size] = 0 # bas de la derniere ligne
	return proba

# Maillage de notre domaine
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
		h = (maxi[i] - mini[i]) / (size-1)
		MeshVect.append([mini[i] + j*h for j in range(size-1)])
		MeshVect[i].append(maxi[i])

	x, y = meshgrid(MeshVect[0], MeshVect[1])

	T = []
	for X in Steiner_points:
		borne = 1000
		new_point = []
		for idx, i in enumerate(MeshVect[0]):
			if abs(X[0]-i) < borne:
				borne = abs(X[0]-i)
				j = idx
		new_point.append(j)
		borne = 1000
		for idx, i in enumerate(MeshVect[1]):
			if abs(X[1]-i) < borne:
				borne = abs(X[1]-i)
				j = idx
		new_point.append(j)
		T.append(new_point)

	return (x,y, MeshVect, T) # maillage x, maillage y, vecteur de maille, ...
	# ... position matriciel des terminaux

def double_plot(Proba, MeshVect, x, y, liste_chemin, T):

	mat = [[0 for i in range(size)] for j in range(size)]
	for i in range(size):
		for j in range(size):
			mat[i][j] = sum([Proba[k*size*size + i*size+j] for k in range(4)])



	plt.figure(2)
	plt.subplot(211)
	plt.matshow(mat, origin='lower')
	plt.subplot(212)
	plt.scatter(x, y, s = 10)
	plt.scatter([MeshVect[0][i[0]] for i in T],[MeshVect[1][i[1]] for i in T], s = 80, c = 'red')
	for chemin in liste_chemin:
		for i in range(len(chemin)-1) :
			X1, X2 = chemin[i], chemin[i+1]
			plt.plot([MeshVect[0][X1[0]], MeshVect[0][X2[0]]], [MeshVect[1][X1[1]], MeshVect[1][X2[1]]], c='g' )

	plt.show()

# affichage de la matrice proba
def print_matrix_proba(matrice, taille):
	for i in range(4*taille):
		for j in range(taille):
			print(matrice[i*taille + j], end =' ')
		print()

# simulation du trajet d'une fourmi
def cheminFourmi(coord_fourmi, proba, term):

	depart = choice(coord_fourmi) # position random dans la colonie au depart
	vertex_visite = coord_fourmi.copy() # elle ne se deplace pas dans la colonie
	chemin = [depart] # point de depart
	orientation = []
	v = chemin[-1]
	while(True):
		case = v[1]*(size) + v[0]
		proba_local = [proba[case + i*size*size] for i in range(4)]
		P_max = sum(proba_local)
		max_it = 0
		while(True):
			if max_it > 20 : # on est bloque
				chemin.pop() # On reste pas ici
				if(len(orientation)>0): # attention sur Steiner point on peut etre bloque
					orientation.pop()
				else :
					# Steiner point bloquant on annule tout on restart la fourmi
					return cheminFourmi(coord_fourmi, proba, term)
				v = chemin[-1] # On est mieux ici
				case = v[1]*(size) + v[0]
				proba_local = [proba[case + i*size*size] for i in range(4)]
				P_max = sum(proba_local)
				max_it = 0
			choix = random()*P_max
			for idx, i in enumerate(cumsum(proba_local)):
				if choix <= i :
					break
			if idx ==0 :
				w = [v[0], v[1]-1]
			elif idx ==1 :
				w = [v[0]-1, v[1]]
			elif idx ==2 :
				w = [v[0], v[1]+1]
			elif idx ==3 :
				w = [v[0]+1, v[1]]
			else :
				print("ERREUR PROBA >>>")
				return

			if(w not in vertex_visite):
				break
			max_it+=1

		v=w
		chemin.append(v)
		orientation.append(idx)
		vertex_visite.append(v)
		if(v in term):
			break
	return chemin, orientation # chemin de la fourmi + orientation (1, 2, 3 ou 4)

# affiche un chemin
def afficher_chemin(MeshVect, x, y, liste_chemin, T):

	plt.scatter(x, y, s = 10)
	plt.scatter([MeshVect[0][i[0]] for i in T],[MeshVect[1][i[1]] for i in T], s = 80, c = 'red')
	for chemin in liste_chemin:
		for i in range(len(chemin)-1) :
			X1, X2 = chemin[i], chemin[i+1]
			plt.plot([MeshVect[0][X1[0]], MeshVect[0][X2[0]]], [MeshVect[1][X1[1]], MeshVect[1][X2[1]]], c='g' )
	plt.show()
	return

# depot et dissipation de pheromone des N fourmis
def maj_proba(proba, liste_chemin, liste_orientation):
	for i in range(len(proba)):
	 	proba[i] = proba[i]*dissip
	for I, CheminI in enumerate(liste_chemin):
		concentration = float(len(CheminI))
		for J, OrientationJ in enumerate(liste_orientation[I]):
			proba[CheminI[J][1]*size + CheminI[J][0] + OrientationJ * size * size] += 5 * mu/concentration
	return

# aggrandissement de la colonie a la fin dun tour
def maj_colonie(colonie, best_path):
	colonie.extend(best_path)

if __name__ == '__main__':

	seed(1)

	X = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
	(x, y, MeshVect, T) = mesh(X)


	colonie = [T[0]]
	plot_chemin = []
	#moyenne = []
	for arete in range(len(X)-1):
		edge = initProba()
		print(colonie)
		for j in range(Ntour):
			liste_chemin = []
			liste_direction = []
			for i in range(Nfourmi):
				chemin, direction = cheminFourmi(colonie, edge, T)
				liste_chemin.append(chemin)
				liste_direction.append(direction)
			#moyenne.append(sum([len(c) for c in liste_chemin])/Nfourmi)
			maj_proba(edge, liste_chemin, liste_direction)
		#norme_proba(edge)
		longueur_chemins = [len(c) for c in liste_chemin]
		idx = longueur_chemins.index(min(longueur_chemins))
		best_path = liste_chemin[idx]
		plot_chemin.append(best_path)
		afficher_chemin(MeshVect, x, y, plot_chemin, T)
		maj_colonie(colonie, best_path)
