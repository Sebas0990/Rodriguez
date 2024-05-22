import numpy as np
from collections import defaultdict
from itertools import combinations

class Recommender:
    def __init__(self):
        self.RULES = []
        self.database = []
        self.prices = []

    def eclat(self, transactions, minsup_count, max_length):
        item_tidsets = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                item_tidsets[item].add(tid)

        item_tidsets = {item: tids for item, tids in item_tidsets.items() if len(tids) >= minsup_count}

        def eclat_recursive(prefix, items_tidsets, frequent_itemsets, max_length):
            sorted_items = sorted(items_tidsets.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (item, tidset_i) in enumerate(sorted_items):
                new_itemset = prefix + (item,)
                if len(new_itemset) <= max_length:
                    frequent_itemsets.append((new_itemset, len(tidset_i)))
                    suffix_tidsets = {}
                    for item_j, tidset_j in sorted_items[i + 1:]:
                        new_tidset = tidset_i & tidset_j
                        if len(new_tidset) >= minsup_count:
                            suffix_tidsets[item_j] = new_tidset
                    eclat_recursive(new_itemset, suffix_tidsets, frequent_itemsets, max_length)

        frequent_itemsets = []
        eclat_recursive(tuple(), item_tidsets, frequent_itemsets, max_length)
        return frequent_itemsets

    def apriori(self, transactions, minsup_count, max_length):
        itemsets = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                itemsets[frozenset([item])] += 1

        itemsets = {itemset: support for itemset, support in itemsets.items() if support >= minsup_count}

        k = 2
        while True:
            next_itemsets = defaultdict(int)
            for itemset in combinations(itemsets.keys(), k):
                combined_set = frozenset.union(*itemset)
                if len(combined_set) <= max_length:
                    support_count = sum(1 for transaction in transactions if combined_set.issubset(transaction))
                    if support_count >= minsup_count:
                        next_itemsets[combined_set] = support_count
            if not next_itemsets:
                break
            itemsets.update(next_itemsets)
            k += 1

        frequent_itemsets = [(itemset, support) for itemset, support in itemsets.items()]
        return frequent_itemsets

    def train(self, prices, database, minsup_count=10, minconf=0.1, max_length=5):
        self.database = database
        self.prices = prices

        # Eclat
        frequent_itemsets_eclat = self.eclat(database, minsup_count, max_length)

        # Apriori
        frequent_itemsets_apriori = self.apriori(database, minsup_count, max_length)

        # Combine results from both algorithms
        frequent_itemsets_combined = frequent_itemsets_eclat + frequent_itemsets_apriori

        self.RULES = self.create_association_rules(frequent_itemsets_combined, minconf)
        return self
    
    def calculate_confidence(self, antecedent_support, support):
        return support / antecedent_support if antecedent_support > 0 else 0

    def create_association_rules(self, frequent_itemsets, minconf):
        rules = []
        itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
        for itemset, support in frequent_itemsets:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset([itemset[i]])
                    consequent = frozenset(itemset[:i] + itemset[i+1:])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    confidence = self.calculate_confidence(antecedent_support, support)
                    if confidence >= minconf:
                        rules.append((antecedent, consequent, {'confidence': confidence}))
        return rules

    def get_recommendations(self, cart, max_recommendations=5):
        normalized_prices = self.prices
        recommendations = defaultdict(list)
        for rule in self.RULES:
            if rule[0].issubset(cart):
                for item in rule[1]:
                    if item not in cart:
                        price_factor = normalized_prices[item] if item < len(normalized_prices) else 0
                        metric_factor = rule[2]['confidence']
                        score = metric_factor * (1 + price_factor)
                        recommendations[item].append(score)
        
        avg_recommendations = {item: sum(scores) / len(scores) if len(scores) > 0 else 0 for item, scores in recommendations.items()}
        sorted_recommendations = sorted(avg_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]

# Ejemplo de uso
if __name__ == "__main__":
    recommender = Recommender()
    prices = [10, 20, 30, 15, 25]
    database = [[1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 2, 3, 
