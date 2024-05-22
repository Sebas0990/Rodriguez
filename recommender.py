import numpy as np
from collections import defaultdict

class Recommender:
    def __init__(self):
    self.data = {
        'RULES': [],
        'database': [],
        'prices': []
    }

    def eclat(self, transactions, minsup_count):
        item_tidsets = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                item_tidsets[item].add(tid)

        def recursive_eclat(prefix, items_tidsets):
            for item, tidset in items_tidsets.items():
                if len(tidset) >= minsup_count:
                    new_itemset = prefix + [item]
                    self.frequent_itemsets.append((new_itemset, len(tidset)))
                    suffix_tidsets = {next_item: next_tidset & tidset for next_item, next_tidset in items_tidsets.items() if next_item > item}
                    recursive_eclat(new_itemset, suffix_tidsets)

        recursive_eclat([], item_tidsets)
        return self.frequent_itemsets

    def calculate_confidence(self, antecedent_support, support):
    if antecedent_support > 0:
        return support / antecedent_support
    else:
        return 0

    def create_association_rules(self, frequent_itemsets, minconf):
    rules = []
    itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for i, antecedent in enumerate(itemset):
                consequent = frozenset(itemset[:i] + itemset[i+1:])
                antecedent_support = itemset_support.get(frozenset([antecedent]), 0)
                confidence = self.calculate_confidence(antecedent_support, support)
                if confidence >= minconf:
                    rules.append((frozenset([antecedent]), consequent, {'confidence': confidence}))
    return rules

    def train(self, prices, database, minsup_count=10, minconf=0.1):
    self.database = database
    self.prices = prices
    frequent_itemsets = self.eclat(database, minsup_count)
    self.RULES = self.create_association_rules(frequent_itemsets, minconf)
    return self
    
    def get_recommendations(self, cart, max_recommendations=5):
    normalized_prices = self.prices
    recommendations = defaultdict(float)  # Usamos un defaultdict de tipo float para simplificar los cálculos de suma y promedio
    for antecedent, consequent, metrics in self.RULES:
        if antecedent.issubset(cart):
            for item in consequent:
                if item not in cart:
                    price_factor = normalized_prices.get(item, 0)  # Utilizamos get para manejar casos en los que el ítem no tenga precio
                    metric_factor = metrics['confidence']
                    score = metric_factor * (1 + price_factor)
                    recommendations[item] += score
    
    avg_recommendations = {item: score / len(self.RULES) for item, score in recommendations.items()}  # Dividimos por la cantidad total de reglas
    sorted_recommendations = sorted(avg_recommendations.items(), key=lambda x: x[1], reverse=True)[:max_recommendations]
    return sorted_recommendations

        return [item for item, _ in sorted_recommendations[:max_recommendations]]
