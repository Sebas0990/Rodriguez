import numpy as np
from collections import defaultdict

class Recommender:
    def __init__(self):
        self.RULES = []
        self.database = []
        self.prices = []

   from collections import defaultdict
from itertools import combinations

def apriori(transactions, minsup_count):
    item_support = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_support[item] += 1

    items = {item for item, count in item_support.items() if count >= minsup_count}

    def generate_candidates(itemsets, k):
        candidates = set()
        itemsets_list = list(itemsets)
        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                candidate = itemsets_list[i] | itemsets_list[j]
                if len(candidate) == k:
                    candidates.add(candidate)
        return candidates

    def get_frequent_itemsets(candidates, transactions, minsup_count):
        candidate_support = defaultdict(int)
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_support[frozenset(candidate)] += 1
        return {itemset for itemset, count in candidate_support.items() if count >= minsup_count}, candidate_support

    k = 1
    frequent_itemsets = []
    current_itemsets = {frozenset([item]) for item in items}

    while current_itemsets:
        current_frequent_itemsets, support_data = get_frequent_itemsets(current_itemsets, transactions, minsup_count)
        frequent_itemsets.extend([(tuple(itemset), support_data[frozenset(itemset)]) for itemset in current_frequent_itemsets])
        k += 1
        current_itemsets = generate_candidates(current_frequent_itemsets, k)

    return frequent_itemsets

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

    def train(self, prices, database, minsup_count=10, minconf=0.1):
        self.database = database
        self.prices = prices
        frequent_itemsets = self.eclat(database, minsup_count)
        self.RULES = self.create_association_rules(frequent_itemsets, minconf)
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

# Ejemplo de uso
if __name__ == "__main__":
    recommender = Recommender()
    prices = [10, 20, 30, 15, 25]
    database = [[1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 2, 3, 4], [1, 3, 4, 5]]
    recommender.train(prices, database)
    cart = [1, 2]
    recommendations = recommender.get_recommendations(cart)
    print("Recomendaciones:", recommendations)
