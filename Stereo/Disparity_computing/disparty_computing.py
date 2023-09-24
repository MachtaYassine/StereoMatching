import numpy as np
from tqdm import tqdm
class DisparityComputation:
    def __init__(self,strategy='winner-takes-all'):
        self.strategy=strategy
        self.P1,self.P2=0.025,0.5
        

    def compute_disparity_map(self,cost_volume):
        if self.strategy == 'winner-takes-all':
            return self.winner_takes_all(cost_volume)
        elif self.strategy == 'semi-global-matching':
            return self.semi_global_matching(cost_volume)
        elif self.strategy == 'dynamic-programming':
            return self.dynamic_programming(cost_volume)
        elif self.strategy == 'graph-cuts':
            return self.graph_cuts()
        elif self.strategy == 'belief-propagation':
            return self.belief_propagation(cost_volume)
        elif self.strategy == 'SemiGlobalMatching':
            return self.SemiGlobalMatching(cost_volume)
        else:
            raise ValueError("Invalid disparity computation strategy")

    def winner_takes_all(self,cost_volume):
        # Winner-takes-all strategy (argmin over disparity levels)
        disparity_map = np.argmin(cost_volume, axis=2)
        return disparity_map

    def semi_global_matching(self):
        # Implement the Semi-Global Matching (SGM) algorithm here
        # You may need to create a separate class for SGM
        pass

    def dynamic_programming(self):
        # Implement the dynamic programming-based disparity computation algorithm
        # This may involve creating a cost-to-go table and backtracking to find disparities
        pass

    def graph_cuts(self):
        # Implement the graph cuts-based disparity computation algorithm
        # You may need to use a library like PyMaxflow or implement the algorithm from scratch
        pass

    def belief_propagation(self):
        # Implement the belief propagation-based disparity computation algorithm
        # This may involve message passing between neighboring pixels
        pass
    
    def SemiGlobalMatching(self,cost_volume):
        height, width, max_disparity = cost_volume.shape
        num_directions = 8  # Number of aggregation directions (typically 8 in SGM)
        
        # Initialize the cost and aggregation tables
        cost_table = np.copy(cost_volume)
        aggregation_table = np.zeros((height, width, max_disparity, num_directions), dtype=np.float32)
        
        # Define aggregation directions (e.g., left, right, up, down, diagonal)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # Iterate over each direction
        for direction_idx, (dy, dx) in tqdm(enumerate(directions)):
            for y in range(height):
                for x in range(width):
                    for d in range(max_disparity):
                        # Calculate pixel coordinates in the current direction
                        y_dir = y + dy
                        x_dir = x + dx

                        if 0 <= y_dir < height and 0 <= x_dir < width:
                            # Calculate aggregation cost using P1 and P2 penalties
                            cost_aggregated = cost_table[y_dir, x_dir, d] + self.P1 * min(
                                aggregation_table[y_dir, x_dir, d, direction_idx],
                                min(aggregation_table[y_dir, x_dir, max(0, d - 1):d + 2, direction_idx])
                            )
                            
                            # Apply the P2 penalty based on the difference in disparities
                            disparity_difference = abs(d - aggregation_table[y_dir, x_dir, d, direction_idx])
                            cost_aggregated += self.P2 * disparity_difference

                            # Store the aggregated cost
                            aggregation_table[y, x, d, direction_idx] = cost_aggregated

        # Initialize the disparity map
        disparity_map = np.argmin(np.sum(aggregation_table, axis=-1), axis=-1)

        return disparity_map

# Example usage:

