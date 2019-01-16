#include <stdio.h>
#include <stdlib.h>
#include <utility>      
#include <string>       
#include <iostream>     
#include <vector>

struct li;

struct no
{
	float *X; // emplacement
    struct li **edges; // edges 
};
typedef struct no Node;

struct li
{
	float pherom;
    std::pair<Node*,Node*> noeuds;
};
typedef struct li Link;

int mesh(std::vector< std::vector<float> > points)
{	
	float h; // taille mesh

	int Ndim = points[0].size();
	float min[Ndim];
	float max[Ndim];
	for(int i=0; i<Ndim; i++) // Init
	{
		min[i] = -500;
		max[i] = 500;
	} 

	for(int i=0; i<Ndim; i++) // pour chaque dimension
	{
		for (std::vector< std::vector<float> >::iterator it = points.begin() ; it != points.end(); ++it)
		{
			if((*it)[i] > max[i])	max[i] = (*it)[i];
			if((*it)[i] < min[i])	min[i] = (*it)[i];
		}
	}
	// Ici on a les extremas de chaque dimension i
	// On maille par 20.

	float S = 20.;
	std::vector< Node* > grille;
	for(int i=0; i<Ndim; i++)
	{
		h = (max[i]-min[i])/S;
		Node* 
		//grille.push_back();

	}

	return 0;
}

int main(){

	return 0;
}