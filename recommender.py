import numpy as np
from collections import defaultdict

class Recommender:
    def __init__(self):
        self.RULES = []
        self.database = []
        self.prices = []

    def eclat(self, transactions, minsup_count):
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
                if suffix_tidsets:
                    eclat_recursive(new_itemset, suffix_tidsets, frequent_itemsets)

        frequent_itemsets = []
        eclat_recursive(tuple(), item_tidsets, frequent_itemsets)
        return frequent_itemsets

    def calculate_confidence(self, antecedent_support, support):
        return support / antecedent_support if antecedent_support > 0 else 0

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
                        rules.append((antecedent, consequent, {'confidence': confidence}))
        return rules

    def train(self, prices, database, minsup_count=10, minconf=0.1):
        self.database = database
        self.prices = prices
        frequent_itemsets = self.eclat(database, minsup_count)
        self.RULES = self.create_association_rules(frequent_itemsets, minconf)
        return self
    
    def get_recommendations(self, cart, max_recommendations=5):
        normalized_prices = self.prices
        recommendations = defaultdict(list)
        for antecedent, consequent, metrics in self.RULES:
            if antecedent.issubset(cart):
                for item in consequent - set(cart):
                    price_factor = normalized_prices[item] if item < len(normalized_prices) else 0
                    score = metrics['confidence'] * (1 + price_factor)
                    recommendations[item].append(score)
        
        avg_recommendations = {item: sum(scores) / len(scores) if len(scores) > 0 else 0 for item, scores in recommendations.items()}
        sorted_recommendations = sorted(avg_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]

# Ejemplo de uso
if __name__ == "__main__":
    recommender = Recommender()
    prices = [10, 20, 30, 15, 25]
    database = [[1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 2, 3, 4], [1, 3, 4, 5]]
    recommender.train(prices, database)
    cart = [1, 2]
    recommendations = recommender.get_recommendations(cart)
    print("Recomendaciones:", recommendations)

    cart = [1, 2]
    recommendations = recommender.get_recommendations(cart)
    print("Recomendaciones:", recommendations)
