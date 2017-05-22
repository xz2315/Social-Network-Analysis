import pickle

# 5a
with open('opinions.pk', 'rb') as f:
	C = pickle.load(f)

num_cascades = len(C)
tau = len(C[0])
print num_cascades, 'cascades of length', tau

idx = 0
while not C[6][13][idx]:
	idx = idx + 1
print 'Node 13 is activated in cascade 6 at time', idx


# 5b
G = {}
e = []
with open('network1.txt', 'rb') as f2:
	for line in f2:
		u, v = ( int(v) for v in line.split())
		if u not in G:
			G[u] = [v]
		else:
			G[u].append(v)
		if u not in e:
			e.append(u)
		if v not in e:
			e.append(v)

num_nodes = len(G)
avg_deg = sum([len(G[u]) for u in G])/float(len(G))
print num_nodes, 'nodes with average out-degree', avg_deg


# 5c
def doesInfluence(n, m, t, o):
	if t > tau - 2:
		return 0
	elif C[o][n][t] and C[o][m][t+1] and not C[o][m][t]:
		return 1
	else:
		return 0

print doesInfluence(10, 4, 1, 0)


# 5d
def firstActive(v, c):
	t = 0
	while not C[c][v][t] and t != tau:
		t = t + 1
	return t

weights = {}
FA = {}
for u in G.keys():
	FA[u] = [firstActive(u, o) for o in xrange(num_cascades)]
	for v in G[u]:
		weights[(u,v)] = sum([ doesInfluence(u,v,FA[u][o],o) for o in xrange(num_cascades) ]) / float(sum([ C[o][u][FA[u][o]] for o in xrange(num_cascades) ]))

print 'MLE w(1,2):', weights[(1,2)]
print 'MLE w(26,21):', weights[(26,21)]

with open('mle_edge_weights.csv', 'wb') as e:
	for key in weights.keys():
		e.write(','.join([str(u) for u in key] + [str(weights[key])]))
		e.write('\n')


# 5e
edge_weights = {}
for u in G:
	edge_weights[u] = []
	for e in weights:
		if u in e:
			edge_weights[u].append(weights[e])
	edge_weights[u] = float(sum(edge_weights[u]))/len(edge_weights[u])
	FA[u] = float(sum(FA[u]))/len(FA[u])

print edge_weights.keys()[edge_weights.values().index(max(edge_weights.values()))], 'has highest average outgoing edge weight while', edge_weights.keys()[edge_weights.values().index(min(edge_weights.values()))], 'has the lowest.'
print FA.keys()[FA.values().index(max(FA.values()))], 'is activated last, on average, while', FA.keys()[FA.values().index(min(FA.values()))], 'is activated earliest.'


