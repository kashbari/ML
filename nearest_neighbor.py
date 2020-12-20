# coding: utf-8

## k Nearest Neighbors implementation using scipy

# # Warning
# 
# The solution notebook took about an hour, in total, to run with the dimension and number of points given below.  You should test this on smaller data sets and give yourself time to run the entire notebook before submitting.

# # Project: Nearest Neighbors in graph

# ## Overview
# Given a fixed collection of `N` points in $\mathbb{R}^k$, this project consists of implementing algorithms to find the closest point to the input data $y \in \mathbb{R}^k$.
# Here closest is measured in the Euclidian norm. 

# 
# ### Generating Data
# 
# ### Brute Force
# 
# ### kd-trees
# 
# ### Distortion
# 
# 1. For an input data `X`, complete the function `compute_distortion`.  The distortion is computed as the $\max_i \frac{\|f(X_i)\|_2}{\|X_i\|_2} \cdot \max_i \frac{\|X_i\|_2}{\|f(X_i)\|_2}$, where $f(X_i) = AX_i$ is the projection of $X_i$, the ith column of `X`.
# 
# 1. Run the given code block to compute the distortion using a small data set and a sample projection matrix `A`, created using your implementation of `create_projection_matrix`. 
# 
# 
# ### Iteration
# 


import math
import numpy as np
from scipy import linalg
from scipy import spatial
import time
import matplotlib.pyplot as plt  

# run_time_start = time.perf_counter()
## Generating Data
## Compute distance between two points 
def distance(x,y):
    return linalg.norm(x-y)


def create_column_data(num_elements, dimension, 
                       lower_bound = -1.0, upper_bound = 1.0):    
    """
    The returned np.array has number of cols=num_elements and number of rows=dimension.
    Each column of the returned object is a data point containing values that 
    fall in the half-open interval [lower_bound, upper_bound).  The memory layout of
    the returned np.array is Fortran order (column major).
    """
    return np.asfortranarray(np.random.uniform(lower_bound,upper_bound,(dimension,num_elements)))



## Large dimension and data test
#num_elements = 100000
#data_dimension = 1000

## Small dimension test
#num_elements = 20
#data_dimension = 5

X = create_column_data(num_elements, data_dimension, -1.0, 1.0)
y = create_column_data(1, data_dimension, -1.0, 1.0)


# ## Brute Force
# 
def nn_brute_force(test_point, data):
    """
    Computes the distance between test_point and every column in data.  The function
    assumes that the number of elements in test_point matches the length of each column
    in data.  The column with minimum distance is returned via a tuple containing the
    index of the column from data and the column itelf with shape modified to match 
    test_point.  Note that this function returns a view of the column from data and not
    a copy.
    """
    if test_point is None or data is None:
        return (None,None)
    dist = [distance(test_point,data[:,0].reshape(test_point.shape)),0]
    for k in range(1,data.shape[1]):
        d = distance(test_point,data[:,k].reshape(test_point.shape))
        if d < dist[0]:
            dist = [d,k]
    return (dist[1],data[:,dist[1]].reshape(test_point.shape))


# Sanity Check
(idx, nn) = nn_brute_force(y, X)

brute_force_idx = idx
brute_force_distance = distance(y, nn)
print(f'Nearest Neighbor found at index {brute_force_idx} with distance {brute_force_distance}')


# ## kd-trees
# 
#  This function should create a `scipy.spatial.KDTree` from the passed data and return the tree object.  If the passed data is `None` then `None` should be returned.


from scipy.spatial import KDTree
def nn_create_kd_tree(data):
    """
    data is a set of column-based points to be spatially sorted into a kd-tree.
    This function returns a scipy.spatial.KDTree object representing data or 
    None if data is None.
    """
    if data is None:
        return None
    return KDTree(data.transpose())




def nn_query_kd_tree(test_point, tree):
    """
    Given a test_point and a scipy.spatial.KDTree this function queries the
    kd-tree for the closest point to test_point.  The function returns a tuple
    containing the index of the closest point in the tree along with the 
    closest point itself reshaped to match the shape of test_point.  
    If either argument is None then (None, None) is returned.
    """
    if test_point is None or tree is None:
        return (None,None)
    d = tree.query(test_point.transpose())
    nn = tree.data[d[1]]
    return (d[1],nn.transpose())



# Sanity Check    
tree = nn_create_kd_tree(X)

(idx, nn) = nn_query_kd_tree(y, tree)

kdtree_idx = idx
kdtree_distance = distance(y, nn)
print(f'Nearest Neighbor found at index {kdtree_idx} with distance {kdtree_distance}')
if kdtree_distance != brute_force_distance:
    print('Brute Force and KDTree give different results.  Something is amiss.')
else:
    print('Brute Force and KDTree give the same distance.')




# ## Approximate Nearest Neighbors - Setup
# 


def create_projection_matrix(n, m): 
    """
    Returns an np.array with n rows and m columns whose values are ranomly sampled
    from a normal distribution (0.0, 1.0).  The memory layout for the returned
    np.array will be row-major.
    """
    return np.random.randn(n, m)


# Fix C and epsilon and use Johnson-Lindenstrauss Lemma to 
# find the smallest reduced dimension (k in the notes)
C = 2.5 # C \in (0,\inf)
epsilon = 0.5 # epsilon \in (0,\inf)
# k >= C log(n) / eps**2, we use ceiling to ensure k is an integer for our map
reduced_dimension = math.ceil(C * math.log(data_dimension) / epsilon**2)

print('Creating map from dimension', data_dimension, 'to', reduced_dimension)
A = create_projection_matrix(reduced_dimension, data_dimension)


# Run the following to test  brute force and kd-tree implementations on the projected data set.



(idx, nn) = nn_brute_force(A@y, A@X)

reduced_dim_brute_force_distance = distance(y, X[:,idx])
print(f'Reduced Dimension Nearest Neighbor found at index {idx} with distance {reduced_dim_brute_force_distance} in original space')
print(f'Original Nearest Neighbor found at index {brute_force_idx} with distance {brute_force_distance}')



tree = nn_create_kd_tree(A@X)

(idx, nn) = nn_query_kd_tree(A@y, tree)

reduced_dim_kdtree_distance = distance(y, X[:,idx])
print(f'Nearest Neighbor found at index {idx} with distance {reduced_dim_kdtree_distance} in original space')
print(f'Original Nearest Neighbor found at index {kdtree_idx} with distance {kdtree_distance}')



# ## Distortion
# 


def compute_distortion(X, T_X):
    """
    Computes and returns the distortion between the set of data points (columns in x)
    and their image (columns in T_x).  The distortion is computed as 
    max_i(||T_X_i|| / ||X_i||) * max_i(||X_i|| / ||T_X_i||), where T_X_i and X_i are the
    ith columns of T_X and X, respectively and max_i is the max over all i.  Also, ||a|| 
    here refers to the L2 norm.
    """
    n = []
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            n.append(distance(X[:,i],X[:,j])/distance(T_X[:,i],T_X[:,j]))
    return max(n)/min(n)
        


# ## Iteration
# 
# Even if the distortion imposed by the linear map is small the resulting nearest neighbor in the reduced space may not be the nearest neighbor in the original space.  This can be seen by running many instances of random projections and comparing their results.
# 


def iterate_reduced_nn(X, y, reduced_dimension, num_trials=10):
    """
    This function performs a number of nearest neighbor trials, each of which 
    consists of the projection of the input data set x and test point y to a 
    lower dimension and the nearest neighbor of the test point is sought in 
    the data set (in the lower dimension).  The nearest neighbor computation 
    will be performed by first creating a scipy.spatial.KDTree with the 
    projected x and then querying it using the projected y.  The index of each
    identified nearest neighbor will be added to a list and only the unique
    indices returned (no duplicates).
    
    X - A column based data set from which the nearest neighbor to y is sought
    y - A test point with length equal to column length of X, which is to be used
        as a query point
    reduced_dimension - An integer representing the dimension of the reduced space
        in which the nearest neighbor computation will be performed.
    num_trials - The number of projection matrices to be tested.
    """
    red_nn = []
    for k in range(num_trials):
        P = create_projection_matrix(reduced_dimension, X.shape[0])
        X_p = P@X
        y_p = P@y
        tree = nn_create_kd_tree(X_p)
        (idx, nn) = nn_query_kd_tree(y_p, tree)
        if idx not in red_nn:
            red_nn.append(idx)
    return red_nn
        


def nn_iterative(X, y, reduced_dimension, num_trials=10):
    """
    Perform a number of approximate nearest neighbor trials in a reduced space. 
    The results of those trials are used to select a subset of the data points in 
    X and repeat the nearest neighbor computation in the original dimension using
    only the selected data points.  The function returns a tuple containing the 
    index of the nearest neighbor in the original data set along with the nearest 
    neighbor itself reshaped to match the shape of test_point. 
    
    X - A column based data set from which the nearest neighbor to y is sought
    y - A test point with length equal to column length of X, which is to be used
        as a query point
    reduced_dimension - An integer representing the dimension of the reduced space
        in which the nearest neighbor computation will be performed.
    num_trials - The number of projection matrices to be tested.
    """
    red_nn = iterate_reduced_nn(X, y, reduced_dimension, num_trials)
    cand = [int(k) for k in red_nn]
    X_res = X[:,cand]
    tree = nn_create_kd_tree(X_res)
    return nn_query_kd_tree(y, tree)
    


# ### Test Code
# 
# Run the following code block and check how close this approximate nearest neighbor is to the one computed using brute force. 


(idx, nn) = nn_iterative(X, y, reduced_dimension)

iterative_distance = distance(y, nn)
print(f'Nearest Neighbor found at index {idx} with distance {iterative_distance}')
if iterative_distance != brute_force_distance:
    print(f'Brute Force and Iterative give different results.  Brute Force distance = {brute_force_distance}')
else:
    print('Brute Force and Iterative give the same distance.')



