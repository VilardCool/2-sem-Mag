import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

transactions = (("a","b","c","d"), ("b","c","d"), ("a","e","f","g","h"), ("b","c","d","e","g","j"), ("b","c","d","e","f"),
            ("a","f","g"), ("a","i","j"), ("a","b","e","h"), ("f","g","h","i","j"), ("e","f","h"))

#transactions = (("a","b","c","d", "e"), ("a","c","d", "f"), ("a","b","c","d","e","g"), ("c","d","e","f"), ("c","e","f","h"),
#            ("d","e","f"), ("a","f","g"), ("d","e","g","h"), ("a","b","c","f"), ("c","d","e","h"))

def find_uniItems(transactions):
  unique_items = []
  for i in transactions:
    for j in i:
      if j not in unique_items:
        unique_items.append(j)
  return unique_items
     

unique_items = find_uniItems(transactions)

from collections import Counter

def find_frequency(lists):
    result = {}
    for sub_list in lists:
        sub_list_counter = Counter(sub_list)
        for item, count in sub_list_counter.items():
            if item in result:
                result[item] += count
            else:
                result[item] = count
    return result

frequent_item_sets = find_frequency(transactions)

min_support = (40/100)*len(transactions)

def remove_infrequent_and_sort(frequent_item_sets, min_support):
  temp_itemset = frequent_item_sets.copy()
  for key,values in frequent_item_sets.items():
    if values < min_support:
      temp_itemset.pop(key)
    elif key == '':
      temp_itemset.pop(key)
    else:
      continue
      
  frequent_item_sets = temp_itemset
  keys = list(frequent_item_sets.keys())
  values = list(frequent_item_sets.values())
  sorted_value_index = np.argsort(values)
  sorted_value_index = np.flip(sorted_value_index)
  frequent_item_sets = {keys[i]: values[i] for i in sorted_value_index}
  return frequent_item_sets

frequent_item_sets = remove_infrequent_and_sort(frequent_item_sets, min_support)

def build_ordered_itemset(transactions, frequent_item_sets):
    keys = list(frequent_item_sets.keys())
    temp_transactions = []
    for transaction in transactions:
        temp_items = []
        for item in transaction:
            if item in keys:
                temp_items.append(item)
        temp_transactions.append(temp_items)
    
    transactions = []
    for temp_transaction in temp_transactions:
        new_transaction = []
        for key in keys:
            if key in temp_transaction:
                new_transaction.append(key)
        transactions.append(new_transaction)
    
    return transactions

transactions = build_ordered_itemset(transactions, frequent_item_sets)

class Node:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next_node = None
        self.link = None

    def add_child(self, child):
        if child.item not in self.children:
            self.children[child.item] = child

    def increment_count(self, count):
        self.count += count

    def get_nodes_with_item(self, item):
        nodes = []
        if self.item == item:
            nodes.append(self)
        for child in self.children.values():
            nodes.extend(child.get_nodes_with_item(item))
        return nodes

class FPTree:
    def __init__(self):
        self.root = Node("*", 0, None)
        self.header_table = {}

    def add_transaction(self, transaction):
        current_node = self.root
        for item in transaction:
            child_node = current_node.children.get(item)
            if child_node is None:
                child_node = Node(item, 0, current_node)
                current_node.children[item] = child_node
                if item in self.header_table:
                    last_node = self.header_table[item]
                    while last_node.link is not None:
                        last_node = last_node.link
                    last_node.link = child_node
                else:
                    self.header_table[item] = child_node
            child_node.increment_count(1)
            current_node = child_node

    def get_frequent_items(self, min_support):
        frequent_items = {}
        for item in self.header_table:
            support = 0
            node = self.header_table[item]
            while node is not None:
                support += node.count
                node = node.link
            if support >= min_support:
                frequent_items[item] = support
        return frequent_items
    
    def get_nodes_with_item(self, item):
        return self.header_table.get(item, [])

def find_pattern_base(fptree, node, item):
    pattern_base = {}
    while node is not None:
        prefix_path = []
        temp_node = node
        while temp_node.parent is not None:
            if temp_node.name != "null" and temp_node.name != item:
                prefix_path.append(temp_node.name)
            temp_node = temp_node.parent
        if len(prefix_path) > 0:
            pattern_base[frozenset(prefix_path)] = node.count
        if node.link is None:
            break
        node = node.link
    return pattern_base

def create_subtree(fptree, min_support):
    items = list(fptree.header_table.keys())
    for item in items:
        support = 0
        node = fptree.header_table[item]
        while node is not None:
            support += node.count
            node = node.link
        if support < min_support:
            del fptree.header_table[item]
        else:
            fptree.header_table[item] = support
    for item in fptree.header_table:
        nodes = []
        node = fptree.header_table[item]
        while node is not None:
            nodes.append(node)
            node = node.link
        fptree.header_table[item] = nodes
    conditional_tree = FPTree()
    for item in items:
        pattern_base = find_pattern_base(fptree.root, item)
        for transaction, count in pattern_base.items():
            transaction_list = list(transaction)
            for i in range(count):
                conditional_tree.add_transaction(transaction_list)
    frequent_items = conditional_tree.get_frequent_items(min_support)
    for item in frequent_items:
        conditional_tree.header_table[item] = frequent_items[item]
    return conditional_tree

def generate_frequent_patterns(fptree, min_support, prefix=[]):
    items = [v[0] for v in sorted(fptree.items(), key=lambda kv: kv[1]['support'])]
    for item in items:
        new_prefix = prefix.copy()
        new_prefix.append(item)
        support = fptree[item]["support"]
        yield (new_prefix, support)
        conditional_pattern_base = find_pattern_base(fptree[item]["node_link"], item)
        conditional_tree = create_subtree(conditional_pattern_base, min_support)
        if len(conditional_tree) > 0:
            for pattern in generate_frequent_patterns(conditional_tree, min_support, new_prefix):
                yield pattern
     

class FP_Growth:
    def __init__(self, transactions, min_support):
        self.transactions = transactions
        self.min_support = min_support

    def build_fptree(self):
        self.fptree = FPTree()
        for transaction in self.transactions:
            self.fptree.add_transaction(transaction)

    def generate_frequent_itemsets(self, node, suffix):
        frequent_itemsets = []
        support = node.count
        for item, child_node in node.children.items():
            itemset = suffix.copy()
            itemset.add(item)
            frequent_itemsets.append((itemset, support))
            frequent_itemsets.extend(self.generate_frequent_itemsets(child_node, itemset))
        return frequent_itemsets

    def mine_frequent_itemsets(self):
        self.build_fptree()
        frequent_itemsets = []
        for item, count in self.fptree.get_frequent_items(self.min_support).items():
            frequent_itemsets.append(([item], count))
        for itemset in itertools.chain.from_iterable(
                self.generate_frequent_itemsets(self.fptree.header_table[item], set()) for item in self.fptree.header_table):
            frequent_itemsets.append(itemset)
        return frequent_itemsets
     

fp = FP_Growth(transactions, min_support)
frequent_itemsets = fp.mine_frequent_itemsets()

print(frequent_itemsets)