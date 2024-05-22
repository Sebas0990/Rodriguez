import numpy as np

class Recommender:

    def train(self, prices, database) -> None:
        """
        Entrena el sistema de recomendación con los precios y la base de datos proporcionados.

        Args:
            prices (list): Lista de precios de los artículos.
            database (list): Base de datos de transacciones.

        Returns:
            self: Instancia actual del objeto Recommender.
        """
        # Inicializar variables de la instancia
        self.database = database
        self.prices = prices
        
        # Ejecutar el algoritmo de Eclat y generar reglas de asociación
        minsup_count = 10
        self.eclat(database, minsup_count)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=0.1, transactions=self.database)
        
        return self

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
        Genera recomendaciones basadas en el carrito de compras proporcionado.

        Args:
            cart (list): Lista de ítems en el carrito de compras.
            max_recommendations (int): Número máximo de recomendaciones a devolver.

        Returns:
            list: Lista de ítems recomendados.
        """
        # Obtener precios normalizados (actualmente no implementado)
        normalized_prices = self.prices
        
        # Calcular y ordenar las recomendaciones según el promedio de sus valores
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
        
        # Devolver las primeras max_recommendations recomendaciones
        return [item for item, _ in sorted_recommendations[:max_recommendations]]
