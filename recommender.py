import numpy as np
from collections import defaultdict

class Recommender:
    def __init__(self):
        self.rules = []
        self.support_data = {}
        self.prices = []

    def eclat(self, transactions, minsup_count):
        item_tids = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                item_tids[item].add(tid)

        item_tids = {item: tids for item, tids in item_tids.items() if len(tids) >= minsup_count}

        def eclat_recursive(prefix, items_tids, frequent_itemsets):
            sorted_items = sorted(items_tids.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (item, tids) in enumerate(sorted_items):
                new_itemset = prefix + [item]
                frequent_itemsets.append((new_itemset, len(tids)))
                suffix_tids = {}
                for item_j, tids_j in sorted_items[i + 1:]:
                    new_tids = tids & tids_j
                    if len(new_tids) >= minsup_count:
                        suffix_tids[item_j] = new_tids
                if suffix_tids:
                    eclat_recursive(new_itemset, suffix_tids, frequent_itemsets)

        frequent_itemsets = []
        eclat_recursive([], item_tids, frequent_itemsets)
        return frequent_itemsets

    def generate_association_rules(self, frequent_itemsets, minconf, transactions):
        rules = []
        itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
        for itemset, support in frequent_itemsets:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset([itemset[i]])
                    consequent = frozenset(itemset[:i] + itemset[i+1:])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = support / antecedent_support
                        if confidence >= minconf:
                            rules.append((antecedent, consequent, confidence))
        return rules

    def train(self, prices, transactions, minsup_count=10, minconf=0.1):
        self.prices = prices
        frequent_itemsets = self.eclat(transactions, minsup_count)
        self.rules = self.generate_association_rules(frequent_itemsets, minconf, transactions)
        return self

    def get_recommendations(self, cart, max_recommendations=5):
        recommendations = defaultdict(float)
        for antecedent, consequent, confidence in self.rules:
            if antecedent.issubset(cart):
                for item in consequent:
                    if item not in cart:
                        price_factor = self.prices[item] if item < len(self.prices) else 0
                        score = confidence * (1 + price_factor)
                        recommendations[item] += score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]


