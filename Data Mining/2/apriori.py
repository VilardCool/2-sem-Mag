from itertools import combinations

DATASET = (("a","b","c","d"), ("b","c","d"), ("a","e","f","g","h"), ("b","c","d","e","g","j"), ("b","c","d","e","f"),
            ("a","f","g"), ("a","i","j"), ("a","b","e","h"), ("f","g","h","i","j"), ("e","f","h"))

#DATASET = (("a","b","c","d", "e"), ("a","c","d", "f"), ("a","b","c","d","e","g"), ("c","d","e","f"), ("c","e","f","h"),
#            ("d","e","f"), ("a","f","g"), ("d","e","g","h"), ("a","b","c","f"), ("c","d","e","h"))

MIN_SUPPORT = 4
MIN_CONFIDENCE = 75.0

c1 = dict()

for itemset in DATASET:
    for i in itemset:
        c1[i] = c1.get(i,0) + 1
for item in list(c1):
    if c1[item]<MIN_SUPPORT:
        del c1[item]

items = list(c1.keys())
support = [c1]

for i in range(2,len(items)):
    s = dict()
    for combo in combinations(items,i):
        for itemset in DATASET:
            if set(combo).issubset(itemset):
                s[combo] = s.get(combo,0) + 1
        if s.get(combo) and s[combo]<MIN_SUPPORT:
            del s[combo]
    if not s:
        break
    support.append(s)
print(support)

rules = dict()
for combo in support[-1]:
    for item in combo:
        c = list(combo)
        c.remove(item)
        len_c = len(c)
        c = c[0] if len_c == 1 else tuple(c)
        rule_1 = support[-1][combo]/support[0][item]*100
        rule_2 = support[-1][combo]/support[len_c-1][c]*100
        if rule_1>=MIN_CONFIDENCE: rules[f"{item}->{c}"] = rule_1
        if rule_2>=MIN_CONFIDENCE: rules[f"{c}->{item}"] = rule_2

print(rules)