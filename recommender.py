import numpy as np
from collections import defaultdict
from itertools import combinations

class Recommender:
    def __init__(self):
        self.RULES = []
        self.frequent_itemsets = None
        self.database = []
        self.prices = []

    def apriori(self, transactions, minsup_count):
        print("apriori")
        itemset_support = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                itemset_support[frozenset([item])] += 1

        itemset_support = {item: support for item, support in itemset_support.items() if support >= minsup_count}
        
        def join_step(itemsets, length):
            return set([i.union(j) for i in itemsets for j in itemsets if len(i.union(j)) == length])

        def prune_step(itemsets, prev_freq_itemsets):
            return set([itemset for itemset in itemsets if all(frozenset(subset) in prev_freq_itemsets for subset in combinations(itemset, len(itemset) - 1))])

        k = 2
        current_itemsets = set(itemset_support.keys())
        frequent_itemsets = [(itemset, support) for itemset, support in itemset_support.items()]

        while current_itemsets:
            candidate_itemsets = join_step(current_itemsets, k)
            candidate_itemsets = prune_step(candidate_itemsets, current_itemsets)
            itemset_counts = defaultdict(int)
            for transaction in transactions:
                transaction_set = set(transaction)
                for candidate in candidate_itemsets:
                    if candidate.issubset(transaction_set):
                        itemset_counts[candidate] += 1

            current_itemsets = set([itemset for itemset, count in itemset_counts.items() if count >= minsup_count])
            frequent_itemsets.extend([(itemset, count) for itemset, count in itemset_counts.items() if count >= minsup_count])
            k += 1

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
        B = []
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
                            B.append((antecedent, consequent, metrics))
        return B

    """def normalize_prices(self):
        print("normalized")
        if not self.prices:
            return []
        max_price = max(self.prices)
        min_price = min(self.prices)
        range_price = max_price - min_price or 1  

        normalized_prices = [(price - min_price) / range_price for price in self.prices]
        return normalized_prices"""
        
    def train(self, prices, database):
        print("training")
        self.database = database
        self.prices = prices
        minsup_count = 10
        self.apriori(database, minsup_count)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=0.1, transactions=self.database)
        return self
    
    def get_recommendations(self, cart, max_recommendations=5):
        print("recommendations")
        print(cart)
        # normalized_prices = self.normalize_prices()
        normalized_prices = self.prices

        recommendations = {}
        for rule in self.RULES:
            if rule[0].issubset(cart):  # Si el antecedente de la regla está presente en el carrito
                for item in rule[1]:  # Para cada elemento en el consecuente de la regla
                    if item not in cart:  # Si el elemento no está en el carrito
                        price_factor = normalized_prices[item] if item < len(normalized_prices) else 0
                        metric_factor = rule[2]['confidence']
                        score = metric_factor * (1 + price_factor)
                        recommendations[item] = recommendations.get(item, []) + [score]  # Guardar todos los scores
        # Calcular el promedio de los scores para cada ítem
        avg_recommendations = {item: sum(scores) / len(scores) for item, scores in recommendations.items()}
        # Ordenar las recomendaciones según el promedio de sus valores de mayor a menor
        sorted_recommendations = sorted(avg_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia de la clase Recommender
    recommender = Recommender()
    
    # Datos de ejemplo: precios y base de datos de transacciones
    prices = [10, 20, 30, 15, 25]
    database = [[1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 2, 3, 4], [1, 3, 4, 5]]
    
    # Entrenar el modelo
    recommender.train(prices, database)
    
    # Obtener recomendaciones para un carrito de compras
    cart = [1, 2]
    recommendations = recommender.get_recommendations(cart)
    print("Recomendaciones:", recommendations)

