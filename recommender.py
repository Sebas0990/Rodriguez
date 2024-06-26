import numpy as np
from collections import defaultdict

class Recommender:
    def __init__(self):
        self.RULES = defaultdict(list)
        self.frequent_itemsets = None
        self.database = []
        self.prices = []

    def eclat(self, transactions, minsup_count):
        print("eclat")
        item_tidsets = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                item_tidsets[item].add(tid)

        item_tidsets = {item: tids for item, tids in item_tidsets.items() if len(tids) >= minsup_count}

        def eclat_recursive(prefix, items_tidsets, frequent_itemsets):
            sorted_items = sorted(items_tidsets.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (item, tidset_i) in enumerate(sorted_items):
                new_itemset = prefix + (item,)
                frequent_itemsets.append((new_itemset, len(tidset_i)))
                suffix_tidsets = {}
                for item_j, tidset_j in sorted_items[i + 1:]:
                    new_tidset = tidset_i & tidset_j
                    if len(new_tidset) >= minsup_count:
                        suffix_tidsets[item_j] = new_tidset
                eclat_recursive(new_itemset, suffix_tidsets, frequent_itemsets)

        frequent_itemsets = []
        eclat_recursive(tuple(), item_tidsets, frequent_itemsets)
        self.frequent_itemsets = frequent_itemsets

    def calculate_supports(self, D, X, Y=None):
        print("calculate_sup")
        count_X, count_XY, count_Y = 0, 0, 0 if Y else None
        for transaction in D:
            has_X = set(X).issubset(transaction)
            has_Y = set(Y).issubset(transaction) if Y else False
            if has_X:
                count_X += 1
                if Y and has_Y:
                    count_XY += 1
            if Y and has_Y:
                count_Y += 1
        sup_X = count_X / len(D)
        sup_XY = count_XY / len(D)
        sup_Y = count_Y / len(D) if Y is not None else None
        return sup_X, sup_XY, sup_Y
    
    def createAssociationRules(self, F, minconf, transactions):
        print("CreateASSO")
        B = defaultdict(list)
        itemset_support = {frozenset(itemset): support for itemset, support in F}
        for itemset, support in F:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset([itemset[i]])
                    consequent = frozenset(itemset[:i] + itemset[i+1:])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    if antecedent_support > 0:
                        conf = support / antecedent_support
                        if conf >= minconf:
                            metrics = {
                                'confidence': conf  
                            }
                            B[antecedent].append((consequent, metrics))
        return B

    def train(self, prices, database, minsup_count=10, minconf=0.1):
        print("training")
        self.database = database
        self.prices = prices
        self.eclat(database, minsup_count)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=minconf, transactions=self.database)
        return self
    
    def get_recommendations(self, cart, max_recommendations=5):
        print("recommendations")
        print(cart)
        normalized_prices = self.prices

        recommendations = {}
        for antecedent, rules in self.RULES.items():
            if antecedent.issubset(cart):
                for consequent, metrics in rules:
                    for item in consequent:
                        if item not in cart:
                            price_factor = normalized_prices[item] if item < len(normalized_prices) else 0
                            metric_factor = metrics['confidence']
                            score = metric_factor * (1 + price_factor)
                            recommendations[item] = recommendations.get(item, []) + [score]

        avg_recommendations = {item: sum(scores) / len(scores) for item, scores in recommendations.items()}
        sorted_recommendations = sorted(avg_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]

