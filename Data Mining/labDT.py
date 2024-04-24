import numpy as np

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

data = np.array(X)

P = Y

X, Y = data[:, :-1], data[:, -1]
Y = Y.reshape((-1, 1))

class Node:
     def __init__(self, data=None, children=None, split_on = None, pred_class=None, is_leaf=False):

        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf

class DecisionTreeClassifier:
    def __init__(self):
        self.root = Node()


    def fit(self, X, Y):
        data = np.column_stack([X, Y])
        self.root.data = data
        self.best_split(self.root)

    def meet_criteria(self, node):
        y = self.get_y(node.data)
        return True if self.calculate_entropy(y) == 0 else False

    @staticmethod
    def get_y(data):
        y = data[:, -1]
        return y

    @staticmethod
    def calculate_entropy(Y):
        _, labels_counts = np.unique(Y, return_counts=True)
        total_instances = len(Y)
        entropy = sum([label_count / total_instances * np.log2(1 / (label_count / total_instances)) for label_count in labels_counts])
        return entropy

    @staticmethod
    def get_pred_class(Y):
        labels, labels_counts = np.unique(Y, return_counts=True)
        index = np.argmax(labels_counts)
        return labels[index]

    def best_split(self, node):
        if self.meet_criteria(node):
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        index_feature_split = -1
        min_entropy = 1

        for i in range(data.shape[1] - 1):
            split_nodes, weighted_entropy = self.split_on_feature(node.data, i)
            if weighted_entropy < min_entropy:
                child_nodes, min_entropy = split_nodes, weighted_entropy
                index_feature_split = i

        node.children = child_nodes
        node.split_on = index_feature_split

        for child_node in child_nodes.values():
            self.best_split(child_node)

    def split_on_feature(self, data, feat_index):
        feature_values = data[:, feat_index]
        unique_values = np.unique(feature_values)

        split_nodes = {}
        weighted_entropy = 0
        total_instances = len(data)

        for unique_value in unique_values:
            partition = data[data[:, feat_index] == unique_value, :]
            node = Node(data=partition)
            split_nodes[unique_value] = node
            partition_y = self.get_y(partition)
            node_entropy = self.calculate_entropy(partition_y)
            weighted_entropy += (len(partition) / total_instances) * node_entropy

        return split_nodes, weighted_entropy

    def traverse_tree(self, x, node):
        print(node.data)
        if node.is_leaf:
            return node.pred_class

        feat_value = x[node.split_on]

        predicted_class = self.traverse_tree(x, node.children[feat_value])

        return predicted_class

    def predict(self, X):
        predictions = np.array([self.traverse_tree(x, self.root) for x in X])
        return predictions
    
model = DecisionTreeClassifier()
model.fit(X, Y)
res = model.predict([P])

print("Class of Y: ", res[0])