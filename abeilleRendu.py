from random import seed
import matplotlib.pyplot as plt
from random import random, shuffle
import numpy as np
from math import acos, sqrt



#Fonction permettant de créer (t-2 arêtes, t étant le nombre de terminaux) et de les placer dans l'espace delimités par les terminaux
def create_abeille(T):
    bees = []
    min_x = min([i[0] for i in T])
    max_x = max([i[0] for i in T])
    min_y = min([i[1] for i in T])
    max_y = max([i[1] for i in T])
    for i in range(len(T)-2):
        bees.append([min_x + random()*(max_x-min_x), min_y + random()*(max_y-min_y)])
    return bees


#Fonction affichage permettant d'afficher les terminaux, les abeilles et les liaisons entre abeilles et terminaux
def affichage(bees, T, links):
    plt.scatter([i[0] for i in bees], [i[1] for i in bees], s = 50, c = 'red')
    plt.scatter([i[0] for i in T], [i[1] for i in T], s = 80, c = 'blue')
    for idx, lien in enumerate(links):
        X0 = bees[idx]
        for X1 in lien:
            if X1 in T or X1 in bees :
                plt.plot([X0[0], X1[0]], [X0[1], X1[1]], c='green')
    plt.show()



#calcul de la distance euclidienne entre 2 points
def distance(P1,P2):
    return np.sqrt((P1[0]-P2[0])*(P1[0]-P2[0])+(P1[1]-P2[1])*(P1[1]-P2[1]))


#Fonction déplacement - définir une nouvelle position pour l'abeille 
def deplacement(bee, proche):
    bee = [sum([i[0] for i in proche])/3.0, sum([i[1] for i in proche])/3.0]
    return bee



# 2 fonction afin de définir une nouvelle position pour l'abeille en utilisant le point de Fermat 
#definir la mediane --> 1 point au centre des autres points ( point de Fermat )

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

#Lancer l'alorithme de recherche de la médiane jusqu' a ce que la distance entre deux itérations soit inférieure à epsilon 
def geometric_median(Points,eps):
    n=float(len(Points))
    Pini=[sum(P[0] for P in Points)/n,sum(P[1] for P in T)/n]#point initial : moyenne de tous les points 
    distIni=0 
    for i in range(len(Points)):
        distIni=distIni+distance(Points[i],Pini)
    P=Pini
    cptMAX=15
    cpt=0
    while True:
        [Qx,Qy,dist] = median(P,Points)
        P=[Qx,Qy]
        if (dist < eps or cpt== cptMAX or dist > distIni):
            Fermat = [Qx,Qy]
            return Fermat 
        cpt=cpt+1





#Il y'a 3 fonctions sonar dont le but est de renvoyer les voisins les plus proches de chaque abeille
#Au fur et à mesure des améliorations de la fonction sonar, des contraintes sont ajoutées afin de corriger certains "défauts"



#Fonction sonar 1 de base - renvoie les 3 voisins les plus proche de l'abeille bee en fonction de la distance euclidienne  
def sonar1(bee, bees, T):
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





#Fonction sonar 2 dont le but est de renvoyer les 3 voisins les plus proches -
#Ajout d'une contrainte visant à forcer tous les terminaux à être reliés qu'à 1 abeille
def sonar2(bee, bees, NonVisited):
    nearest = []
    X = bees + NonVisited
    # calcul carre des normes
    normes= []
    for i in X:
        normes.append(abs((bee[0]-i[0])**2 + (bee[1]-i[1])**2))
    for i in range(3):
        idx = normes.index(min(normes))
        normes.pop(idx)
        nearest.append(X.pop(idx))
    for k in range(len(nearest)):
        if (nearest[k] in NonVisited ) :
            NonVisited.remove(nearest[k])
    return nearest, NonVisited




#Fonction sonar 3 -
#Ajout d'une contraite sur le nombre de liaisons des abeilles 
def sonar3(bee, bees, beesOK, beesN, NonVisited):
    nearest = []
    X = beesOK + NonVisited
    normes= []
    for i in X:
        normes.append(sqrt(abs((bee[0]-i[0])**2 + abs(bee[1]-i[1])**2)))
    for i in range(3):
        idx = normes.index(min(normes))
        normes.pop(idx)
        nearest.append(X.pop(idx))
    for k in range(len(nearest)):
        if (nearest[k] in NonVisited ) :
            NonVisited.remove(nearest[k])
        elif nearest[k] in bees:
            beesN[bees.index(nearest[k])]= beesN[bees.index(nearest[k])] + 1
    for h in range(len(beesN)):
        if beesN[h] > 3 and bees[h] in beesOK: #Nombre de liaisons max pour une abeille = 3 
            beesOK.remove(bees[h])
    return nearest, NonVisited, beesOK, beesN







#Fonction principale 
if __name__ == '__main__':
    seed(1)
    nbIte=8 # on determine un nombre d'itérations pour definir l'arbre de steiner minimale
    T=[] # Stockage des positions des terminaux 
    cst=6
    #cas symétriques : 
    #cas du pentagone 
    for i in range(0,cst):
        T.append([np.cos(2*i*np.pi/cst), np.sin(2*i*np.pi/cst)])
    #cas du carré
    #T = [[0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]] 
    #cas non symétrique : 
    #T = [[0.0,5.0], [0.0,0.], [2.5, 2.5], [2.0, 5*0.8], [3.5,3.5], [1.0,0.5], [ 1.5,3.5]]

    bees = create_abeille(T)
    links = [[] for i in range(len(bees))]
    for i in range(nbIte): 
        beesN=[]  # un tableau contenant le nombre de liaisons pour chaque abeilles
        beesOK=[] # un tableau contenant les abeilles que l'on peut encore choisir de relier 
        for b in range(len(bees)):
            beesN.append(0)
        for b1 in range(len(bees)):
            beesOK.append(bees[b1])
        #creation du tableau contenant les terminaux déja visités 
        NonVisited=[]
        #au départ tous les terminaux sont visités 
        for h in range(len(T)):
            NonVisited.append(T[h])
        affichage(bees, T, links)
        for j, bee in enumerate(bees):
            #lancement des fonctions sonar 
            #nearest=sonar1(bee, [a for a in bees if a!=bee],T)
            #nearest, NonVisited=sonar2(bee, [a for a in bees if a!=bee], NonVisited)
            nearest, NonVisited, beesOK, beesN=sonar3(bee, bees, beesOK, beesN, NonVisited)
            bees[j]=geometric_median(nearest,0.1)
            links[j]=nearest
            
# calculer la distance dans links 
    affichage(bees, T, links)


