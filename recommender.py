import numpy as np

class Recommender:
    """
    This is the class to make recommendations.
    The class must not require any mandatory arguments for initialization.
    """

    def train(self, prices, database) -> None:
        """
        Allows the recommender to learn which items exist, which prices they have, 
        and which items have been purchased together in the past.
        
        :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
        :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
        :return: the object should return itself here (this is actually important!)
        """

        # Build co-occurrence matrix
        num_items = len(prices)
        co_occurrence_matrix = np.zeros((num_items, num_items), dtype=int)
        for transaction in database:
            for item1 in transaction:
                for item2 in transaction:
                    co_occurrence_matrix[item1, item2] += 1
        
        # Normalize co-occurrence matrix
        co_occurrence_matrix /= np.maximum(co_occurrence_matrix.sum(axis=1, keepdims=True), 1)

        # Save necessary data
        self.prices = prices
        self.co_occurrence_matrix = co_occurrence_matrix
        
        return self

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
        Makes a recommendation to a specific user
        
        :param cart: a list with the items in the cart
        :param max_recommendations: maximum number of items that may be recommended
        :return: list of at most `max_recommendations` items to be recommended
        """

        # Calculate recommendation scores based on co-occurrence with items in the cart
        recommendation_scores = np.sum(self.co_occurrence_matrix[cart], axis=0)
        
        # Remove items already in cart from recommendation scores
        for item in cart:
            recommendation_scores[item] = 0
        
        # Get indices of items sorted by recommendation scores
        recommended_items = np.argsort(recommendation_scores)[::-1][:max_recommendations]
        
        # Map indices to item IDs
        recommendations = [item for item in recommended_items if recommendation_scores[item] > 0]
        
        return recommendations

# Example usage:
recommender = Recommender()
recommender.train([10, 20, 30], [[0, 1], [1, 2]])
print(recommender.get_recommendations([0, 1], 3))  # Example call to get_recommendations

