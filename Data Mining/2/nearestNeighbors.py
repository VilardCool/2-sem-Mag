import numpy as np
import matplotlib.pyplot as plt

def nn_clustering(data, k):
    clusters = {i: [i] for i in range(len(data))}  # Initial clusters
    neighbors = {i: find_k_nearest_neighbors(data, i, k) for i in range(len(data))}
    
    for i in range(len(data)):
        for neighbor in neighbors[i]:
            if i in neighbors[neighbor]:
                for clust in clusters[neighbor]:
                    if clust not in clusters[i]:
                        clusters[i].append(clust)
                clusters[neighbor] = clusters[i]
    
    unique_clusters = []
    seen = set()
    for cluster in clusters.values():
        if tuple(sorted(cluster)) not in seen:
            unique_clusters.append(cluster)
            seen.add(tuple(sorted(cluster)))
    
    return unique_clusters

def find_k_nearest_neighbors(data, index, k):
    distances = [(j, euclidean_distance(data[index], data[j])) for j in range(len(data)) if j != index]
    distances.sort(key=lambda x: x[1])
    return [distances[i][0] for i in range(k)]

def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

data = np.array([[1,1],[1,8],[2,2],[2,5],[3,1],[4,3],[5,2],[6,1],[6,8],[8,6]])
#data = np.array([(1,1),(1,2),(1,5),(2,8),(3,7),(4,2),(7,5),(8,3),(8,7),(9,3)])
#data = np.array([(2,1),(2,4),(3,5),(3,6),(4,1),(4,9),(5,4),(5,6),(7,2),(9,8)])
#data = np.array([(1,4),(2,5),(2,8),(3,4),(3,5),(4,1),(4,7),(5,6),(7,6),(8,1)])

k = nn_clustering(data, 3)

colors = ['r','g','b','y','c','m','k','w']

for cluster_index in k:
    color = colors[k.index(cluster_index)]
    for point in cluster_index:
        plt.scatter(data[point][0], data[point][1], color = color,s = 30)

plt.show()
