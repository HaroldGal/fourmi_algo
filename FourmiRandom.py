from random import choice

graphe = [
	[0,2,0,0,0,3],
	[2,0,2,0,3,0],
	[0,2,0,3,0,0],
	[0,0,1,0,3,4],
	[0,3,0,3,0,0],
	[3,0,0,4,0,0]
	]

pathAv = [[1,5],[0,2,4],[1,3],[2,4,5],[1,3],[0,3]]
print(graphe)

wayAv = []
proba = dict()
for i in range(10000):
	way = [0]
	way.append(choice(pathAv[way[-1]]))
	
	while(way[-1]!=3):
		ch = choice(pathAv[way[-1]])
		if(ch != way[-2]):
			way.append(ch)
			pos = ch
	if way in wayAv:
		proba[wayAv.index(way)] +=1
	else:
		wayAv.append(way)
		proba[wayAv.index(way)] = 1
	
print(proba)
