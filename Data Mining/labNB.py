X = [[0, 1, 2, 1],
     [1, 0, 1, 1],
     [0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 2, 1],
     [1, 1, 2, 0],
     [1, 0, 2, 1],
     [1, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 1, 1]]

Y = [1, 1, 1]

"""
X = [[0, 0, 1, 0, 0],
     [1, 0, 0, 1, 1],
     [2, 0, 1, 0, 1],
     [0, 1, 0, 0, 1],
     [0, 1, 1, 0, 1],
     [0, 1, 1, 1, 0],
     [1, 0, 0, 1, 0],
     [2, 0, 0, 0, 1],
     [2, 1, 1, 0, 1],
     [0, 1, 1, 1, 0]]

Y = [2, 1, 1, 1]
"""

Q = []
for i in range(len(X[0])-1):
    Q.append([])
    r = 2
    for x in X:
        if (x[i]==2): r = 3
    for j in range(r):
        Q[i].append([])
        for k in range(2):
            Q[i][j].append(0)

for x in X:
    for i in range(len(x)-1):
        Q[i][x[i]][x[len(x)-1]] += 1

P = [0, 0]

for x in X:
    P[x[len(x)-1]] += 1

print(P, "\n")

PQ = []

for q in Q:
    Qcs = [0, 0]
    for c in range(len(q[0])):
        for r in range(len(q)):
            Qcs[c] += q[r][c]

    for c in range(len(q[0])):
        l = 0
        for v in q:
            if (v[c] == 0): 
                l = 0.1
        PQ.append([])
        for r in range(len(q)):
            PQ[len(PQ)-1].append((q[r][c]+l)/(Qcs[c]+l*len(q)))

i = 0
for pq in PQ:
    print(i, "|", pq)
    i += 1
    if i%2==0: 
        i=0
        print("\n")

R = []

for i in range(2):
    r = P[i]
    for j in range(len(Y)):
        r *= PQ[2*j+i][Y[j]]
    R.append(r)

print(R)

print("Class of Y: ", R.index(max(R)))