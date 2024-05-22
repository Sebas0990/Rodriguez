from collections import defaultdict
from itertools import combinations

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
            frequent_itemsets.append((prefix, len(items_tidsets)))
            for item, tidset in list(items_tidsets.items()):
                new_prefix = prefix + [item]
                new_tidsets = {other_item: other_tidset for other_item, other_tidset in items_tidsets.items() if other_item > item}
                if new_tidsets:
                    eclat_recursive(new_prefix, {other_item: other_tidset & tidset for other_item, other_tidset in new_tidsets.items()}, frequent_itemsets)

        frequent_itemsets = []
        eclat_recursive([], item_tidsets, frequent_itemsets)
        return frequent_itemsets

    def calculate_confidence(self, antecedent_support, support):
        return support / antecedent_support if antecedent_support > 0 else 0

    def create_association_rules(self, frequent_itemsets, transactions, minconf):
        rules = []
        itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
        for itemset, support in frequent_itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = frozenset(itemset - antecedent)
                        antecedent_support = itemset_support.get(antecedent, 0)
                        confidence = self.calculate_confidence(antecedent_support, support)
                        if confidence >= minconf:
                            rules.append((antecedent, consequent, {'confidence': confidence}))
        return rules

    def train(self, prices, database, minsup_count=2, minconf=0.1):
        self.database = database
        self.prices = prices
        frequent_itemsets = self.eclat(database, minsup_count)
        self.RULES = self.create_association_rules(frequent_itemsets, database, minconf)
        return self
    
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
