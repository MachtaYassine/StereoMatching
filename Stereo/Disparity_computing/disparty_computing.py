import numpy as np
from tqdm import tqdm
from numba import jit


def log_array(name,array):
    with open(name+'.txt', "w") as file:
    # Iterate through the elements of the array and write them to the file
        for i,element in enumerate(array):
            if len(array.shape)==3:
                for j,element2 in enumerate(element):
                    file.write(f'all disparity costs for coordinate {i,j} : {element2}' + "\n")
            else:
                file.write(f'values in this row {i} : {element}' + "\n") 

def log(text):
    with open("log.txt", "a") as file:
        file.write(text + "\n")
class DisparityComputation:
    '''
    We know most of these are energy minimization problems so maybe we can group them ?

    '''
    def __init__(self,strategy='winner-takes-all'):
        self.strategy=strategy
        self.P1,self.P2=0.025,0.5
        

    def compute_disparity_map(self,cost_volume,occ=100,disparity=96):
        if self.strategy == 'winner-takes-all':
            return self.winner_takes_all(cost_volume)
        elif self.strategy == 'semi-global-matching':
            return self.semi_global_matching(cost_volume)
        elif self.strategy == 'dynamic-programming':
            return self.dynamic_programming(cost_volume,occ,disparity)
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

    # @jit(nopython=True,parallel=True)
    def dynamic_programming(self,cost_volume,occ,disparity):
        
        # Define parameters
        nRow, nCol,_ = cost_volume.shape
    

        # Initialize arrays for cost, disparity, and path tracking
        
        
        displeft = np.zeros((nRow, nCol))
        dispright = np.zeros((nRow, nCol))

        for y in tqdm(range(nRow)):
            C = np.zeros((nCol, nCol))
            M = np.zeros_like(C)
            for i in range(1, nCol):
                C[i, 0] = i * occ
                C[0, i] = i * occ
            C[0,0]=cost_volume[y,0,0]
                

            for i in range(1, nCol):
                for j in range(1, nCol):
                    if np.abs(i-j)<disparity:
                        temp = cost_volume[y,i,np.abs(i - j)]
                        min1 = C[i - 1, j - 1] + temp
                        min2 = C[i - 1, j] + occ
                        min3 = C[i, j - 1] + occ
                        cmin = min(min1, min2, min3)
                        C[i, j] = cmin  # Cost Matrix
                        
                        if cmin == min1:
                            M[i, j] = 1  # Path Tracker
                            # log(f"diagonal cost is the min {min1} and the others {min2} {min3}")
                        elif cmin == min2:
                            M[i, j] = 2
                            # log(f"left cost is the min {min2} and the others {min1} {min3}")
                        elif cmin == min3:
                            M[i, j] = 3
                            # log(f"right cost is the min {min3} and the others {min1} {min2}")
                    else:
                        if i>j:
                            C[i,j]=C[i,j-1]+occ
                        else:
                            C[i,j]=C[i-1,j]+occ
                        
            
            
            i = nCol - 1
            j = nCol - 1
            while i != 0 and j != 0:
                if M[i, j] == 1:
                    displeft[y, i] = abs(i - j)  # Disparity Image in Left Image coordinates
                    dispright[y, j] = abs(j - i)  # Disparity Image in Right Image coordinates
                    i -= 1
                    j -= 1
                elif M[i, j] == 2:
                    #find the closes non occluded object to the left and give the same disparity
                    displeft[y, i] =np.nan
                    i -= 1
                elif M[i, j] == 3:
                    if j!= nCol-1:
                        dispright[y, j] = dispright[y, j+1]
                    else:
                        dispright[y, j] =np.nan
                    j -= 1

            for i in range(1,nCol):
                if np.isnan(displeft[y,i]):
                    displeft[y,i] = displeft[y,i-1]

        # log_array('M',M)
        # log_array('C',C)
        return displeft,dispright
    
    def graph_cuts(self):
        # Implement the graph cuts-based disparity computation algorithm
        # You may need to use a library like PyMaxflow or implement the algorithm from scratch
        pass

    def belief_propagation(self):
        # Implement the belief propagation-based disparity computation algorithm
        # This may involve message passing between neighboring pixels
        pass
    
    
    def SemiGlobalMatching(self,cost_volume):
        '''
        This implementation is too slow and i'm not sure of the results
        '''
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

