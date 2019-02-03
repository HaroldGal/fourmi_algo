from numpy import meshgrid, cumsum, array
import matplotlib.pyplot as plt
from random import random, seed, choice
from collections import Counter
from math import log, sqrt
import numpy as np
from math import acos
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
    if (l0==0 or l1==0 or l2==0) :#or ((abs(l0-l1) <=eps) and (abs(l1-l2)<=eps) and (abs(l0-l2)<=eps)) ):
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
#    print("long : l0,l1,l2")
#    print(l0,l1,l2)
#    print("val")
#    print (A,B,C)
#    print("arcos")
#    print(a0,a1,a2)
    
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
    
    if (ind==-2) :
        return deplacement(bee,proche)
    # sinon decente gradient 
    eps=0.000001
    alpha=0.0002
    grad=gradobj(bee,proche)
    #ngrad=np.sqrt(grad[0]**2 + grad[1]**2)
    f=5000
    f2=fobj(bee,proche)
    while abs(f-f2)>=eps:
        bee[0]=bee[0] - alpha*grad[0]
        bee[1]=bee[1] - alpha*grad[1]
        grad=gradobj(bee,proche)
        f=f2
        f2=fobj(bee,proche)
    return bee
def create_Steiner(MeshVect, X1, X2, X3):
	point_Steiner = [0,0]
	proche = []
	for i in range(2):
		point_Steiner[i] = (MeshVect[i][X1[i]] + MeshVect[i][X2[i]] + MeshVect[i][X3[i]])/3.0
	a,b,c = [], [], []
	for i in range(2):
		a.append(MeshVect[i][X1[i]])
		b.append(MeshVect[i][X2[i]])
		c.append(MeshVect[i][X3[i]])
	proche.append(a)
	proche.append(b)
	proche.append(c)

	point_Steiner = deplacement2(point_Steiner, proche)
	return point_Steiner


# aggrandissement de la colonie a la fin dun tour
def maj_colonie(colonie, best_path):
	colonie.extend(best_path)

if __name__ == '__main__':

	seed(1)

	# ON RENTRE ICI LES COORDONNEES DES TERMINAUX

	#X = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
    
	X = [[1,0]]
	for i in range(1,5):
		X.append([np.cos(2*i*np.pi/5), np.sin(2*i*np.pi/5)])

	X = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
	##################################################

	(x, y, MeshVect, T) = mesh(X)
	X_visite = [T[0]]
	Steiner_points = []
	link = []

	colonie = [T[0]]
	plot_chemin = []
	#moyenne = []
	for arete in range(len(X)-1):
		edge = initProba()
		#print(colonie)
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
		maj_colonie(colonie, best_path)

		## CREATION POINT STEINER ##

		X_visite.append(best_path[-1])
		if len(X_visite)==3:
			Steiner_points.append(create_Steiner(MeshVect, X_visite[-1],X_visite[-2],X_visite[-3]))
			link.append([X_visite[-1],X_visite[-2],X_visite[-3]])
		elif (len(X_visite)-3)%2==0 and len(X_visite)>3:
			Steiner_points.append(create_Steiner(MeshVect, X_visite[-1],X_visite[-2],X_visite[-3]))
			link.append([X_visite[-1],X_visite[-2],X_visite[-3]])
		# plt.scatter([MeshVect[0][i[0]] for i in X_visite],[MeshVect[1][i[1]] for i in X_visite], s = 100, c = 'blue')
		# plt.scatter([i[0] for i in Steiner_points],[i[1] for i in Steiner_points], s = 50, c = 'red')
		# plt.show()

	for j in range(1,(len(X_visite)-3)%2+1): #Il reste des points pas vu
		norm = [(MeshVect[0][i[0]] - MeshVect[0][X_visite[-j][0]])**2 + (MeshVect[1][i[1]] - MeshVect[1][X_visite[-j][1]])**2 for i in X_visite if i!=X_visite[-j]]
		idx = norm.index(min(norm))
		Steiner_points.append([MeshVect[0][X_visite[idx][0]], MeshVect[1][X_visite[idx][1]]])
		link.append([X_visite[-j]])

	longueur = 0
	for idx, liens in enumerate(link):
		for arete in liens:
			longueur += sqrt((Steiner_points[idx][0] - MeshVect[0][arete[0]])**2 + (Steiner_points[idx][1] - MeshVect[1][arete[1]])**2)

	print("Longueur totale : " + str(longueur))

	plt.scatter([MeshVect[0][i[0]] for i in T],[MeshVect[1][i[1]] for i in T], s = 100, c = 'blue')
	plt.scatter([i[0] for i in Steiner_points],[i[1] for i in Steiner_points], s = 50, c = 'red')
	for idx, liens in enumerate(link):
		for arete in liens:
			plt.plot([Steiner_points[idx][0], MeshVect[0][arete[0]]], [Steiner_points[idx][1], MeshVect[1][arete[1]]], c='g' )
	plt.show()



	# X = [[0.0,1.0], [0.0,0.], [0.5, 0.5], [0.4, 0.8], [0.7,0.7], [0.2,0.1], [ 0.3,0.7]]
	# (x, y, MeshVect, T) = mesh(X)
	#
	# #moyenne = []
	# save = [0 for i in range(size*size*4)]
	# for X0 in T:
	# 	for X1 in T:
	# 		if X0==X1 :
	# 			continue
	# 		edge = initProba()
	# 		print("done")
	# 		liste_chemin = []
	# 		liste_direction = []
	# 		for i in range(Nfourmi):
	# 			chemin, direction = cheminFourmi([X0], edge, [X0,X1])
	# 			liste_chemin.append(chemin)
	# 			liste_direction.append(direction)
	# 		#moyenne.append(sum([len(c) for c in liste_chemin])/Nfourmi)
	# 		maj_proba(edge, liste_chemin, liste_direction)
	# 		save = [save[i]+log(max(0.1,edge[i])) for i in range(len(edge))]
	# norme_proba(save)


	# X = [[0.0,1.0], [0.0,0.0], [1.0,1.0],[1.0,0.0]]
	# (x, y, MeshVect, T) = mesh(X)
	#
	# #moyenne = []
	# save = [0 for i in range(size*size*4)]
	# for X0 in T:
	# 	for X1 in T:
	# 		if X0==X1 :
	# 			continue
	# 		edge = initProba()
	# 		print("done")
	# 		liste_chemin = []
	# 		liste_direction = []
	# 		for i in range(Nfourmi):
	# 			chemin, direction = cheminFourmi([X0], edge, [X0,X1])
	# 			liste_chemin.append(chemin)
	# 			liste_direction.append(direction)
	# 		#moyenne.append(sum([len(c) for c in liste_chemin])/Nfourmi)
	# 		maj_proba(edge, liste_chemin, liste_direction)
	# 		save = [save[i]+log(max(0.1,edge[i])) for i in range(len(edge))]
	# norme_proba(save)

	# X= [[0.0,0.0], [0.25,0.5]]
	# (x, y, MeshVect, T) = mesh(X)
	# colonie = [T[0]]
	# plot_chemin = []
	# edge = initProba()
	# for j in range(Ntour):
	# 	plot_chemin = []
	# 	liste_chemin = []
	# 	liste_direction = []
	# 	for i in range(Nfourmi):
	# 		chemin, direction = cheminFourmi(colonie, edge, T)
	# 		liste_chemin.append(chemin)
	# 		liste_direction.append(direction)
	# 	#moyenne.append(sum([len(c) for c in liste_chemin])/Nfourmi)
	# 	maj_proba(edge, liste_chemin, liste_direction)
	# 	chemin, direction = cheminFourmi(colonie, edge, T)
	# 	plot_chemin.append(chemin)
	# 	afficher_chemin(MeshVect, x, y, plot_chemin, T)
