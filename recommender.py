import numpy as np
from collections import defaultdict

class Recommender:
    def __init__(self):
        self.RULES = []
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

    def train(self, prices, database):
        print("training")
        self.database = database
        self.prices = prices
        minsup_count = 10
        self.eclat(database, minsup_count)  # Llamada al método eclat
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=0.1, transactions=self.database)
        return self

    def get_recommendations(self, cart, max_recommendations=5):
        print("recommendations")
        print(cart)
        normalized_prices = self.prices

        recommendations = {}
        for rule in self.RULES:
            if rule[0].issubset(cart):  
                for item in rule[1]:  
                    if item not in cart:  
                        price_factor = normalized_prices[item] if item < len(normalized_prices) else 0
                        metric_factor = rule[2]['confidence']  
                        score = metric_factor * (1 + price_factor)  
                        recommendations[item] = recommendations.get(item, 0) + score  
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]
