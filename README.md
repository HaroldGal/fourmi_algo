# Metaheuristiques pour le problème de l'arbre Steiner minimal

Le problème de l'arbre Steiner minimal est de trouver un arbre couvrant de longueur minimal sur un arbre qui à ajouter des sommets et des arrêtes a cet arbre.
Une large documentation sur l'étude de ce problème est disponible sur le net. Le logiciel **geosteiner** est optimal pour avoir des résultats en 2 dimensions et l'algorithme de Smith énumère les topologies possible en dimensions plus grande. Cette énumeration s'effectue en temps non polynomial, nous cherchons donc une métaheuristique permettant d'approximer efficacement un arbre steiner minimal.

Un rapport de projet, avec des renseignements supplémentaires sur les métaheuristiques et des résultats est disponible dans le répertoire **documentation**. Attention, la méta-heuristiques des abeilles a été améliorée depuis l'écriture de ce rapport.

## Prérequis

Python3 est utilisé.

* [numpy](http://www.numpy.org/)

* [matplotlib](https://matplotlib.org/)

Les codes sont assez commentés.

## Algorithme d'optimisation par colonie de Fourmi

Algorithme fortement ispiré de [cet article](https://www.researchgate.net/publication/221410657_Ant_colony_optimization_for_Steiner_tree_problems?fbclid=IwAR1DlPncIQS1gC0ZJkZkmKWKX1hy5mRQoB3P1wUmfHUvPVtELTC2MP6MYPc).

### Principe

Le principe de cet algorithme est de mailler notre domaine pour ensuite cherche le plus court chemin avec la norme L1 entre un terminal initial et n'importe quel autre terminal. Ensuite l'opération est effectuée de nouveau, mais avec une subtilité, le plus court chemin sera entre le terminal le plus proche et l'arête créer. Nous chercherons donc un plus court "réseau". L'algorithme de recherche du plus court chemin s'effectue avec l'algorithme d'optimisation par colonies de fourmis.

Une amélioration pour la recherche de point Steiner consiste a poser le point de Fermat de chaque réseau de trois points proches trouvé et d'en faire un point Steiner relié aux trois points proche.



### Organisation

Deux scripts sont disponibles, les paramètres sont à changer dans le script dans la première partie.

* fourmi_algo - Déroule un algorithme de recherche du plus court chemin d'arête à point. Trouve le plus court réseau.

```
python3 fourmi_algo.py
```

* ant_steiner - Ajoute les points Steiner et plot un arbre Steiner correct approximant le minimal.

```
python3 ant_steiner.py
```

### Complexité

La complexité dépend du maillage, et dépendra donc de la dimension. Algorithme non généralisable en grandes dimensions.

## Algorithme d'optimisation par essaim d'abeilles

### Principe

Le principe sera de partir avec nos points Steiner, dits abeilles, capable de communiquer entre eux pour optimiser leurs positions et se déplacer pour améliorer la fonction objectif, la longueur total de l'arbre.

### Organisation

Deux scripts sont disponibles, différents sonar peuvent être utilisés dans le main ajoutant des règles.

* abeilleRendu - Déroule notre optimisation par essaim d'abeilles

```
python3 abeilleRendu.py
```

* bee_3d - Squelette créé pour les tests en 3d, attendant une specification de règle dans son sonar.

```
python3 bee_3d.py
```

### Complexité

La complexité dépend du nombre de points seulement, seul le calcul de norme (facile) dépend de la dimension. Les règles spécifiées ne sont pas encore suffisantes mais prometteuses.
