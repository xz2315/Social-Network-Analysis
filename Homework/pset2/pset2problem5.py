# SOLUTION BY ERIC CHEN
import random
import math

def watts_strogatz_sim(r,k,l):
	nodes = []
	edges = []
	for x in range(r*r):
		nodes.append(x)
	for x in nodes:
		for y in nodes:
			if x < y:
				if (abs(x%r - y%r)+int(abs(x/r - y/r)) <= k):
					edges.append([x,y])
	for x in range(l):
		for a in nodes:
			b = random.randint(0,r*r-2)
			if a > b:
				e = [b,a]
			else:
				b = b+1
				e = [a,b]
			if (e not in edges):
				edges.append(e)
	adjlist = []
	for x in range(r*r):
		adjlist.append([])
	for e in edges:
		a = e[0]
		b = e[1]
		adjlist[a].append(b)
		adjlist[b].append(a)
	return adjlist


def kleinberg_sim(r,k,l):
	nodes = []
	edges = []
	for x in range(r*r):
		nodes.append(x)
	for x in nodes:
		for y in nodes:
			if x < y:
				if (abs(x%r - y%r)+int(abs(x/r - y/r)) <= k):
					edges.append([x,y])
	adjlist = []
	for x in range(r*r):
		adjlist.append([])
	for e in edges:
		a = e[0]
		b = e[1]
		adjlist[a].append(b)
		adjlist[b].append(a)
	for x in range(l):
		for a in nodes:
			totdist = 0.
			for b in nodes:
				dist = abs(a%r - b%r)+int(abs(a/r - b/r))
				if dist != 0:
					totdist = 1./dist/dist + totdist
			val = random.random()
			b = -1
			while val > 0.:
				b = b+1
				dist = abs(a%r - b%r)+int(abs(a/r - b/r))
				if dist != 0:
					scale = 1./dist/dist
				else:
					scale = 0.
				val = val - scale
			if (b not in adjlist[a]):
				adjlist[a].append(b)
	return adjlist

def bfs(adjlist,start):
	queue = []
	visited = []
	parent = []
	for node in range(len(adjlist)):
		visited.append(False)
		parent.append(None)
	queue.append(start)
	visited[start] = True
	while len(queue) != 0:
		cur = queue.pop(0)
		for adj in adjlist[cur]:
			if visited[adj] == False:
				visited[adj] = True
				parent[adj] = cur
				queue.append(adj)
	dists = []
	for node in range(len(adjlist)):
		dists.append(0)
	for node in range(len(adjlist)):
		dist = 0
		cur = node
		while parent[cur] != None:
			dist = dist + 1
			cur = parent[cur]
		dists[node] = dist
	return dists

def av_dist_calc(adjlist):
	tot = 0
	for node in range(len(adjlist)):
		tot = tot + sum(bfs(adjlist,node))
		if node % 1000 == 999:
			print(node + 1)
	avgdist = float(tot)/len(adjlist)/(len(adjlist)-1.)
	return avgdist


watts = []
klein = []
for r in range(1,11):
	watts.append(0.)
	klein.append(0.)
for n in range(10):
	#note that r = 1,2 yields complete graphs all the time
	for r in range(3,11):
		watts[r-1] += av_dist_calc(watts_strogatz_sim(r,2,2))
		klein[r-1] += av_dist_calc(kleinberg_sim(r,2,2))
watts = [i/10. for i in watts]
klein = [i/10. for i in klein]
print(watts)
print(klein)



def findsize(file):
	size = 0
	for line in open(file):
		line = line.rstrip('\n')
		fields = line.split()
		a = int(fields[0])
		b = int(fields[1])
		if a > b:
			if a+1 > size:
				size = a+1
		else:
			if b+1 > size:
				size = b+1
	print("Done sizing!")
	return size

def readin(file):
	adjlist = []
	edges = []
	size = findsize(file)
	for x in range(size):
		adjlist.append([])
	for line in open(file):
		line = line.rstrip('\n')
		fields = line.split()
		a = int(fields[0])
		b = int(fields[1])
		adjlist[a].append(b)
	print("Done reading in!")
	print(len(adjlist))
	return adjlist

def calcdist(adjlist):
	avgdist = av_dist_calc(adjlist)
	return avgdist

def estdist(adjlist,n):
	print("Started estimating:")
	tot = 0 
	for i in range(n):
		dists = bfs(adjlist,random.randint(0,len(adjlist)-1))
		tot = sum(dists) + tot
	avgdist = tot/n/(len(adjlist)-1)
	return avgdist

#print(calcdist(readin("enron.txt")))
print(estdist(readin("enron.txt"),1000))
print(estdist(readin("epinions.txt"),1000))
#print(estdist(readin("livejournal.txt"),50))

