import numpy as np
from collections import defaultdict

class Recommender:
    class Apriori:
    def __init__(self):
        self.RULES = []
        self.database = []
        self.prices = []

    def apriori(self, transactions, minsup_count):
        def generate_candidates(frequent_itemsets, k):
            candidates = set()
            itemsets = list(frequent_itemsets.keys())
            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    candidate = itemsets[i] | itemsets[j]
                    if len(candidate) == k:
                        subsets = combinations(candidate, k - 1)
                        if all(frozenset(subset) in frequent_itemsets for subset in subsets):
                            candidates.add(candidate)
            return candidates

        def count_support(candidates, transactions):
            support_counts = defaultdict(int)
            for transaction in transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        support_counts[candidate] += 1
            return support_counts

        # Step 1: Generate frequent 1-itemsets
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1
        
        frequent_itemsets = {itemset: count for itemset, count in item_counts.items() if count >= minsup_count}
        all_frequent_itemsets = dict(frequent_itemsets)
        
        # Step 2: Generate larger frequent itemsets
        k = 2
        while frequent_itemsets:
            candidates = generate_candidates(frequent_itemsets, k)
            candidate_supports = count_support(candidates, transactions)
            frequent_itemsets = {itemset: count for itemset, count in candidate_supports.items() if count >= minsup_count}
            all_frequent_itemsets.update(frequent_itemsets)
            k += 1
        
        return all_frequent_itemsets
        
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
