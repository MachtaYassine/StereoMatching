import numpy as np

class CostAggregation:
    def __init__(self, aggregation_function):
        self.aggregation_function = aggregation_function

    def aggregate_costs(self, cost_volume):
        height, width, num_disparities = cost_volume.shape
        aggregated_costs = np.zeros((height, width, num_disparities), dtype=np.float32)

        for d in range(num_disparities):
            for i in range(height):
                for j in range(width):
                    # Define the neighborhood window based on your aggregation method
                    neighborhood = self.get_neighborhood(cost_volume, i, j, d)

                    # Apply the custom aggregation function
                    aggregated_costs[i, j, d] = self.aggregation_function(neighborhood)

        return aggregated_costs

    def get_neighborhood(self, cost_volume, i, j, d):
        #Custom function meant to define the neighborhood when aggragating: fixed window etc ...
        pass


