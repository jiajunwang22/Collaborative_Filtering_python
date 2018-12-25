# Collaborative_Filtering_python
part of Data Mining HW

1. UV Decomposition

  Implement the incremental UV decomposition algorithm as discussed in class (and described in the text), where the element is learned one at a time.

  Assume initially all elements in latent factor matrix U and V are 1’s.

  The learning starts with learning elements in U row by row, i.e., U[1,1], U[1,2], …, U[2, 1], …

  It then moves on to learn elements in V column by column, i.e., V[1,1], V[2,1], …, V[1, 2], …

  When learning an element, it uses the latest value learned for all other elements. It should compute the optimal value for the element to minimize the current RMSE as described in class.

  The learning process stops after a specified number of iterations, where a round of learning all elements in U and V is one iteration.

  The algorithm should output RMSE after each iteration (remember that the mean is computed based on non-blank elements in the input matrix M).
  
2. ALS

modified the parallel implementation of ALS (alternating least squares) algorithm in Spark, so that it takes a utility matrix as the input.

The code for the algorithm is als.py under the <your spark installation directory>/examples/src/main/python.
