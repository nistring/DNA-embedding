import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def floyd_warshall_inplace(double[:, :] d):
    """
    In-place Floyd-Warshall algorithm optimized with Cython.
    """
    cdef int K = d.shape[0]
    cdef int i, j, k
    cdef double new_dist
    
    for k in range(K):
        for i in prange(K, nogil=True):
            for j in range(K):
                new_dist = d[i, k] + d[k, j]
                if new_dist < d[i, j]:
                    d[i, j] = new_dist
