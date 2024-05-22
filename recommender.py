import numpy as np
from collections import defaultdict

class Eclat:
    def __init__(self, minsup_count=10):
        self.minsup_count = minsup_count
        self.item_tids = defaultdict(set)
        self.frequent_itemsets = []

    def fit(self, transactions):
        self._find_frequent_itemsets(transactions, ())
        return self.frequent_itemsets

    def _find_frequent_itemsets(self, transactions, prefix):
        for item, tids in self._find_item_tids(transactions, prefix).items():
            if len(tids) >= self.minsup_count:
                self.frequent_itemsets.append((prefix + (item,), len(tids)))
                self._find_frequent_itemsets(transactions, prefix + (item,))

    def _find_item_tids(self, transactions, prefix):
        item_tids = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            if set(prefix).issubset(transaction):
                for item in transaction:
                    if item not in prefix:
                        item_tids[item].add(tid)
        return item_tids

class AssociationRuleGenerator:
    def __init__(self, minconf=0.1):
        self.minconf = minconf
        self.rules = []

    def fit(self, frequent_itemsets):
        self._generate_rules(frequent_itemsets)
        return self.rules

    def _generate_rules(self, frequent_itemsets):
        itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
        for itemset, support in frequent_itemsets:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset([itemset[i]])
                    consequent = frozenset(itemset[:i] + itemset[i+1:])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = support / antecedent_support
                        if confidence >= self.minconf:
                            self.rules.append((antecedent, consequent, confidence))

class Recommender:
    def __init__(self):
        self.prices = []
        self.rules = []

    def train(self, prices, transactions):
        self.prices = prices
        eclat = Eclat()
        frequent_itemsets = eclat.fit(transactions)
        rule_generator = AssociationRuleGenerator()
        self.rules = rule_generator.fit(frequent_itemsets)
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



