# cython: language_level=3, boundscheck=False, wraparound=False

cpdef int hdist(str a, str b):
    cdef int la = len(a), lb = len(b), m = la if la < lb else lb
    cdef int d = 0, i
    cdef bytes ba = a.encode('latin1'), bb = b.encode('latin1')
    cdef const unsigned char[:] va = ba
    cdef const unsigned char[:] vb = bb
    for i in range(m):
        if va[i] != vb[i]:
            d += 1
    return d + (la - lb if la > lb else lb - la)
