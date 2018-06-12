import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    ### YOUR CODE HERE
    out = np.dot(vector1, vector2)
    ### END YOUR CODE

    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    ### YOUR CODE HERE
    out1 = dot_product(vector1, vector2)
    print(out1)
    out2 = dot_product(M, vector1.T)
    print(out2)
    out = out1 * out2
    print('out(dot):', out)
    #out = (vector1.T * vector2) * (M * vector1)
    
    out1 = np.matmul(vector1.T, vector2)
    print(out1)
    out2 = np.matmul(M, vector1)
    print(out2)
    out = out1 * out2
    print('out(matmul):', out)
    ### END YOUR CODE
    
    out1 = vector1.T @ vector2
    print(out1)
    out2 = M @ vector1
    print(out2)
    out = out1 * out2
    print('out(@):', out)

    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    ### END YOUR CODE

    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    ### YOUR CODE HERE
    singular_values = s[:n]
    ### END YOUR CODE
    return singular_values

def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, )

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    w = None
    v = None
    ### YOUR CODE HERE
    w, v = np.linalg.eig(matrix)
    ### END YOUR CODE
    return w, v

def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
        
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w, v = eigen_decomp(matrix)
    eigen_values = []
    eigen_vectors = []
    ### YOUR CODE HERE
    eigen_values = w[:num_values]
    eigen_vectors = v[:num_values]
    ### END YOUR CODE
    return eigen_values, eigen_vectors
