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

print ("     0 | 1")
for q in Q:
    i = 0
    for r in q:
        print(i, "|", r)
        i += 1
    print("\n")

Qp = []
p = 0

for q in Q:
    for i in range(len(q)):
        p += min(q[i])
    Qp.append(p)
    p = 0

for i in range(len(Qp)):
    Qp[i] = Qp[i]/len(X)

print(Qp)

cl = Qp.index(min(Qp))

print("Class of Y: ", Q[cl][Y[cl]].index(max(Q[cl][Y[cl]])))