# cython: language_level=3
# cython: initializedcheck=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

######################################################################################################
# High-performance Binary Partition Tree builder.
######################################################################################################

import random as random
import numpy as np
cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport sqrt, M_PI, log

from libc.stdint cimport uint8_t, uint32_t, uint64_t
# ctypedef unsigned char uint8_t

# cdef inline Py_ssize_t SSMIN(Py_ssize_t a, Py_ssize_t b):
#     return a if a < b else b

# cdef inline Py_ssize_t SSMAX(Py_ssize_t a, Py_ssize_t b):
#     return a if a > b else b

cdef inline uint8_t UMIN(uint8_t a, uint8_t b):
    return a if a < b else b

cdef inline uint8_t UMAX(uint8_t a, uint8_t b):
    return a if a > b else b

#TODO: count 1 with SWAR
cdef inline int popcount64(uint64_t x):
    x = x - ((x >> 1) & <uint64_t>0x5555555555555555)
    x = (x & <uint64_t>0x3333333333333333) + ((x >> 2) & <uint64_t>0x3333333333333333)
    x = (x + (x >> 4)) & <uint64_t>0x0F0F0F0F0F0F0F0F
    return <int>((x * <uint64_t>0x0101010101010101) >> 56)

######################################################################################################
# Heap with reverse indexing, which can alter the weight of the nodes dynamically
######################################################################################################

cdef struct reverse_heap:
    size_t N
    double* W
    size_t* heap
    ssize_t* rev_heap
    size_t heap_size

ctypedef reverse_heap reverse_heap_t


cdef void reverse_heap_initialize(reverse_heap_t* p_heap, size_t initN):
    p_heap.N = initN
    p_heap.W = <double*> PyMem_Malloc(p_heap.N * sizeof(double))
    p_heap.heap = <size_t*> PyMem_Malloc(p_heap.N * sizeof(size_t))
    p_heap.rev_heap = <ssize_t*> PyMem_Malloc(p_heap.N * sizeof(ssize_t))
    p_heap.heap_size = 0
    cdef size_t i
    for i in range(p_heap.N):
        p_heap.W[i] = -1
        p_heap.rev_heap[i] = -1
    
cdef void reverse_heap_deallocate(reverse_heap_t* p_heap):
    PyMem_Free(p_heap.W) ; p_heap.W = NULL
    PyMem_Free(p_heap.heap) ; p_heap.heap = NULL
    PyMem_Free(p_heap.rev_heap) ; p_heap.rev_heap = NULL

cdef void reverse_heap_percolate_up(reverse_heap_t* p_heap, size_t i):
    cdef size_t parent
    while i > 0:
        parent = ((i + 1) // 2) - 1
        if p_heap.W[p_heap.heap[parent]] < p_heap.W[p_heap.heap[i]]:
            break
        p_heap.rev_heap[p_heap.heap[i]] = parent
        p_heap.rev_heap[p_heap.heap[parent]] = i
        p_heap.heap[i], p_heap.heap[parent] = p_heap.heap[parent], p_heap.heap[i]
        i = parent
        # reverse_heap_verify(p_heap)
    # reverse_heap_verify(p_heap, True)

cdef void reverse_heap_percolate_down(reverse_heap_t* p_heap, size_t i):
    cdef size_t left, right
    while True:
        left = ((i + 1) * 2) - 1
        right = left + 1
        if right < p_heap.heap_size \
                and p_heap.W[p_heap.heap[right]] < p_heap.W[p_heap.heap[left]] \
                and p_heap.W[p_heap.heap[right]] < p_heap.W[p_heap.heap[i]]:
            p_heap.rev_heap[p_heap.heap[i]] = right
            p_heap.rev_heap[p_heap.heap[right]] = i
            p_heap.heap[i], p_heap.heap[right] = p_heap.heap[right], p_heap.heap[i]
            assert p_heap.W[p_heap.heap[i]] <= p_heap.W[p_heap.heap[right]] \
                    and p_heap.W[p_heap.heap[i]] <= p_heap.W[p_heap.heap[left]]
            i = right
            # reverse_heap_verify(p_heap)
        elif left < p_heap.heap_size and p_heap.W[p_heap.heap[left]] < p_heap.W[p_heap.heap[i]]:
            p_heap.rev_heap[p_heap.heap[i]] = left
            p_heap.rev_heap[p_heap.heap[left]] = i
            p_heap.heap[i], p_heap.heap[left] = p_heap.heap[left], p_heap.heap[i]
            assert p_heap.W[p_heap.heap[i]] <= p_heap.W[p_heap.heap[left]]
            i = left
            # reverse_heap_verify(p_heap)
        else:
            break
    # reverse_heap_verify(p_heap, True)

cdef void reverse_heap_percolate_up_or_down(reverse_heap_t* p_heap, size_t i):
    cdef size_t parent = ((i + 1) // 2) - 1
    if i != 0 and p_heap.W[p_heap.heap[parent]] > p_heap.W[p_heap.heap[i]]:
        reverse_heap_percolate_up(p_heap, i)
    else:
        reverse_heap_percolate_down(p_heap, i)

# cdef void reverse_heap_verify(reverse_heap_t* p_heap, bint check_weights=False):
#     cdef size_t i, parent
#     for i in range(p_heap.heap_size):
#         assert <size_t>p_heap.rev_heap[p_heap.heap[i]] == i
#         if check_weights:
#             parent = ((i + 1) // 2) - 1
#             assert i == 0 or p_heap.W[p_heap.heap[i]] >= p_heap.W[p_heap.heap[parent]]
#     for i in range(p_heap.heap_size):
#         assert p_heap.rev_heap[i] == -1 or p_heap.heap[p_heap.rev_heap[i]] == i

cdef reverse_heap_push(reverse_heap_t* p_heap, size_t elem, double w):
    cdef size_t i = p_heap.heap_size
    assert p_heap.rev_heap[elem] == -1
    p_heap.rev_heap[elem] = i
    p_heap.heap[p_heap.heap_size] = elem
    p_heap.heap_size += 1
    p_heap.W[elem] = w
    # reverse_heap_verify(p_heap)
    reverse_heap_percolate_up(p_heap, i)

cdef reverse_heap_remove(reverse_heap_t* p_heap, size_t elem):
    cdef ssize_t i = p_heap.rev_heap[elem]
    assert i >= 0 and i < (<ssize_t>p_heap.heap_size)
    p_heap.rev_heap[elem] = -1
    p_heap.W[elem] = -1.0
    if i == (<ssize_t>p_heap.heap_size) - 1:
        p_heap.heap_size -= 1
        # reverse_heap_verify(p_heap)
    else:
        p_heap.heap[i] = p_heap.heap[p_heap.heap_size-1]
        p_heap.rev_heap[p_heap.heap[p_heap.heap_size-1]] = i
        p_heap.heap_size -= 1
        # reverse_heap_verify(p_heap)
        reverse_heap_percolate_up_or_down(p_heap, i)

# cdef size_t reverse_heap_pop(reverse_heap_t* p_heap):
#     cdef size_t elem0 = p_heap.heap[0]
#     reverse_heap_remove(p_heap, elem0)
#     return elem0

cdef size_t reverse_heap_top(reverse_heap_t* p_heap):
    return p_heap.heap[0]

cdef size_t reverse_heap_size(reverse_heap_t* p_heap):
    return p_heap.heap_size

# cdef double reverse_heap_get_weight(reverse_heap_t* p_heap, size_t elem):
#     return p_heap.W[elem]

# cdef double reverse_heap_top_weight(reverse_heap_t* p_heap):
#     return p_heap.W[p_heap.heap[0]]

cdef reverse_heap_update_weight(reverse_heap_t* p_heap, size_t elem, double w):
    cdef ssize_t i = p_heap.rev_heap[elem]
    assert i >= 0 and i < (<ssize_t>p_heap.heap_size)
    p_heap.W[elem] = w
    reverse_heap_percolate_up_or_down(p_heap, i)

######################################################################################################

# cdef inline double bhattacharyya_distance(double mu1, double var1, double mu2, double var2):
#     """
#     Compute the Bhattacharyya distance between two distributions.

#     Parameters:
#     mu1: float
#         Mean of distribution 1.
#     var1: float
#         Variance of distribution 1.
#     mu2: float
#         Mean of distribution 2.
#     var2: float
#         Variance of distribution 2.

#     Returns:
#     float
#         Bhattacharyya distance between the two distributions.
#     """
#      # Handle the case where both variances are zero
#     if var1 == 0 and var2 == 0:
#         if mu1 == mu2:
#             return 0.0  # Distributions are identical
#         else:
#             return abs(mu1-mu2) #np.inf  # No overlap, distance is infinite

#     # Handle the case where one variance is zero
#     if var1 == 0:
#         return 0.25 * (mu1 - mu2)**2 / var2 + 0.5 * log(var2) if mu1 != mu2 else 0.0
#     if var2 == 0:
#         return 0.25 * (mu1 - mu2)**2 / var1 + 0.5 * log(var1) if mu1 != mu2 else 0.0

#     # Regular case where neither variance is zero
#     cdef double mean_diff = mu1 - mu2
#     cdef double var_sum = var1 + var2
#     cdef double var_product = sqrt(var1 * var2)
#     # Compute the mean difference term
#     cdef double mean_diff_term = (mean_diff**2) / (4 * var_sum)
#     # Compute the variance term
#     cdef double variance_term = 0.5 * log(var_sum / (2 * var_product))
#     # Calculate the Bhattacharyya distance
#     cdef double distance = mean_diff_term + variance_term
#     return distance

# Compute the harmonic mean between two non-negative real numbers.
cdef inline double harmonic_mean(double r1, double r2):
    assert r1>0 and r2>0
    return 2 * r1 * r2 / (r1 + r2)

######################################################################################################

ctypedef unsigned int cluster_t

cdef struct cluster_descr:
    uint8_t minR, maxR
    uint8_t minG, maxG
    uint8_t minB, maxB
    uint32_t sumR, sumG, sumB
    # uint64_t sumR2, sumG2, sumB2
    uint32_t area
    uint32_t perimeter
    cluster_t root
    uint64_t mask_bits
    int adjnext
ctypedef cluster_descr cluster_descr_t


cdef struct adjacency_descr:
    cluster_t cl[2]
    int next[2]
    int prev[2]
    size_t edge_length
ctypedef adjacency_descr adjacency_descr_t

######################################################################################################
# Binary Partition Tree
# Inspired by the algorithm of: 
#  "AGAT: Building and evaluating binary partition trees for image segmentation"
######################################################################################################

cdef class BinaryPartitionTreeBuilder:
    cdef size_t     H  # image height
    cdef size_t     W  # image width
    cdef size_t     C  # image channels
    cdef cluster_t  U  # number of unitary clusters
    cdef cluster_t  N  # number of total clusters
    cdef cluster_t  TC # clusters built so far

    cdef bint       use_8ways
    cdef bint       use_v0_rule
    cdef double     use_randomization

    # Cluster descriptors
    cdef cluster_descr_t* clst
    # Left and right clusters
    cdef cluster_t[2]* branches
    
    # Adjacency descriptors
    cdef int num_adjs
    cdef adjacency_descr_t* adjs
    cdef int next_free_adj

    # Adjacency heap for fast merge
    cdef reverse_heap_t heap
    
    #TODO: ADDED PREBUILD PARTITION
    cdef object prebuilt_partitions

    #TODO: ADDED INIT PREBUILD PARTITION
    def __init__(self, image, 
                 use_8ways=True, 
                 use_v0_rule=False,
                #  xxuse_sqrt_area=False, 
                 use_randomization=0.0,
                 prebuilt_partitions=None):

        self.prebuilt_partitions = prebuilt_partitions 
        
        self.use_8ways = use_8ways
        self.use_v0_rule = use_v0_rule
        # self.use_sqrt_area = xxuse_sqrt_area
        self.use_randomization = <double>use_randomization

        if image.dtype!=np.uint8:
            raise Exception('Image pixel type is expected to be uint8.')
        if len(image.shape)!=3:
            raise Exception('Image shape is expected to be 3-dimensional.')
        self.H  = image.shape[0]
        self.W  = image.shape[1]
        self.C  = image.shape[2]
        if self.C!=3 and self.C!=1:
            raise Exception('Image is expected to be RGB (H*W*3) or grayscale (H*W*1).')

        self.U = self.W * self.H
        self.N = 2*self.U - 1
        # allocate cluster descriptors
        self.clst = <cluster_descr_t*> PyMem_Malloc(self.N * sizeof(cluster_descr_t))
        # initialize unitary clusters
        cdef size_t x, y, i
        #TODO: initialize mask_bits
        for i in range(self.U):
            y = i // self.W
            x = i % self.W
            self.clst[i].minR = self.clst[i].maxR = self.clst[i].sumR = image[y,x,0]
            self.clst[i].minG = self.clst[i].maxG = self.clst[i].sumG = image[y,x,1] if self.C==3 else 0
            self.clst[i].minB = self.clst[i].maxB = self.clst[i].sumB = image[y,x,2] if self.C==3 else 0
            # self.clst[i].sumR = image[y,x,0]
            # self.clst[i].sumG = image[y,x,1]
            # self.clst[i].sumB = image[y,x,2]
            # self.clst[i].sumR2 = (<int>image[y,x,0])**2
            # self.clst[i].sumG2 = (<int>image[y,x,1])**2
            # self.clst[i].sumB2 = (<int>image[y,x,2])**2
            self.clst[i].root = i
            self.clst[i].area = 1
            self.clst[i].perimeter = 4 
            self.clst[i].adjnext = -1
            self.clst[i].mask_bits = 0
        self.TC = self.U
        
        cdef cluster_t c0
        #TODO: check prebuilt_partitions
        if self.prebuilt_partitions is not None:
            for x0 in range(<size_t>self.W):
                for y0 in range(<size_t>self.H):
                    c0 = <cluster_t>(y0*self.W + x0)
                    self.clst[c0].mask_bits = 1 << self.prebuilt_partitions[y0, x0]

        # allocate non-unitary cluster branching descriptors
        self.branches = <cluster_t[2]*> PyMem_Malloc((self.N - self.U) * sizeof(cluster_t[2]))
        # initialize adjacency list
        self.num_adjs = (self.W-1)*(self.H-1)*(4 if self.use_8ways else 2) + self.W+self.H-2
        self.adjs = <adjacency_descr_t*> PyMem_Malloc(self.num_adjs * sizeof(adjacency_descr_t))
        reverse_heap_initialize(&self.heap, self.num_adjs)
        self.next_free_adj = 0

        # build initial 4/8-way adjacencies
        
        for x0 in range(<size_t>self.W):
            for y0 in range(<size_t>self.H):
                c0 = <cluster_t>(y0*self.W + x0)
                self.init_adj(c0, x0+1, y0,   1) # right
                self.init_adj(c0, x0,   y0+1, 1) # down
                if self.use_8ways:
                    self.init_adj(c0, x0+1, y0+1, 0) # bottom-right diagonal
                    self.init_adj(c0, x0-1, y0+1, 0) # bottom-left diagonal

        # print(f'W={self.W} H={self.H} C={self.C} U={self.U} N={self.N} TC={self.TC} num_adjs={self.num_adjs}')

    #     for cl in range(self.U):
    #         self.check_list_order(cl)


    # # check that cluster indices are in descending order in the linked lists
    # cdef check_list_order(self, cluster_t cl):
    #     print(f'check_list_order {cl}: ', end='', flush=True)
    #     cdef int a, idx, other_idx, prev_a=-1, perim=0
    #     cdef cluster_t prev_cl = <cluster_t>(-1)
    #     a = self.clst[cl].adjnext
    #     while a >= 0:
    #         idx = 0 if self.adjs[a].cl[0]==cl else 1
    #         assert self.adjs[a].cl[idx]==cl
    #         other_idx = 0 if idx==1 else 1
    #         assert self.adjs[a].prev[idx] == prev_a
    #         # assert prev_cl==<cluster_t>(-1) or prev_cl > self.adjs[a].cl[other_idx]
    #         msg = '' if prev_cl==<cluster_t>(-1) or prev_cl > self.adjs[a].cl[other_idx] else '*'
    #         print(f'{self.adjs[a].cl[other_idx]}{msg} ', end='', flush=True)
    #         perim += self.adjs[a].edge_length
    #         prev_cl = self.adjs[a].cl[other_idx]
    #         prev_a = a
    #         a = self.adjs[a].next[idx]

    #     print(f'    ({perim}/{self.clst[cl].perimeter})', end='')
    #     print(flush=True)


    def __dealloc__(self):
        PyMem_Free(self.clst) ;      self.clst = NULL
        PyMem_Free(self.branches) ;  self.branches = NULL
        PyMem_Free(self.adjs) ;      self.adjs = NULL
        reverse_heap_deallocate(&self.heap)


    # add an initial adjacency link between cluster c0 and the cluster 
    # in (x1,y1), if such position is valid.
    cdef void init_adj(self, cluster_t c0, int x1, int y1, size_t edge_length):
        if x1<0 or x1>=(<int>self.W) or y1<0 or y1>=(<int>self.H):
            return
        cdef cluster_t c1 = <cluster_t>(y1*self.W + x1)
        assert c0 < c1
        assert 0 <= c0 <= self.TC and 0 <= c1 <= self.TC #, f'c0={c0}, c1={c1}, TC={self.TC}'
        # get next free adjacency node
        cdef int a = self.next_free_adj 
        self.next_free_adj += 1
        assert self.next_free_adj <= self.num_adjs
        # print('adjacency ',c0,c1,a, flush=True)
        self.adjs[a].edge_length = edge_length
        # link to both clusters
        self.adjs[a].cl[0] = c0
        self.adjs[a].cl[1] = c1

        self.add_adjacency_to(c0, a)
        self.add_adjacency_to(c1, a)
        
        reverse_heap_push(&self.heap, a, self.get_adj_priority(a))


    # insert the adjacency @a to the linked list of cluster @cl 
    # finding the right position (must keep descending order
    # of the opposite connected cluster ids).
    cdef void add_adjacency_to(self, cluster_t c0, int a):
        cdef int idx, other_idx, next_a, next_idx, other_next_idx, prev_a=-1, prev_idx=-1
        idx = 0 if self.adjs[a].cl[0]==c0 else 1
        assert self.adjs[a].cl[idx]==c0
        other_idx = 0 if idx==1 else 1

        next_a, next_idx = self.clst[c0].adjnext, -1
        while next_a >= 0:
            next_idx = 0 if self.adjs[next_a].cl[0]==c0 else 1
            assert self.adjs[next_a].cl[next_idx]==c0
            other_next_idx = 0 if next_idx==1 else 1
            # keep descending order w.r.t. the connected clusters
            if self.adjs[next_a].cl[other_next_idx] < self.adjs[a].cl[other_idx]:
                break 
            prev_a, prev_idx = next_a, next_idx
            next_a = self.adjs[next_a].next[next_idx]

        # insert a between prev_a and next_a
        self.adjs[a].prev[idx] = prev_a
        if prev_a==-1: # insert as head
            self.clst[c0].adjnext = a
        else:
            self.adjs[prev_a].next[prev_idx] = a

        self.adjs[a].next[idx] = next_a
        if next_a!=-1:
            self.adjs[next_a].prev[next_idx] = a


    # remove node @a from the linked list of cluster @c0
    cdef inline void unlink_adj(self, cluster_t c0, int a):
        cdef int idx, next_a, next_idx, prev_a, prev_idx
        idx = 0 if self.adjs[a].cl[0]==c0 else 1
        assert self.adjs[a].cl[idx]==c0

        prev_a, next_a = self.adjs[a].prev[idx], self.adjs[a].next[idx]

        if prev_a == -1:
            self.clst[c0].adjnext = self.adjs[a].next[idx]
        else:
            prev_idx = 0 if self.adjs[prev_a].cl[0]==c0 else 1
            self.adjs[prev_a].next[prev_idx] = next_a

        if next_a != -1:
            next_idx = 0 if self.adjs[next_a].cl[0]==c0 else 1
            self.adjs[next_a].prev[next_idx] = prev_a

        self.adjs[a].prev[idx], self.adjs[a].next[idx] = -1, -1


    # insert node @a to the head of the linked list of cluster @c0
    cdef inline void relink_adj_head(self, cluster_t c0, int a):
        cdef int idx, next_a, next_idx, prev_a, prev_idx
        idx = 0 if self.adjs[a].cl[0]==c0 else 1
        assert self.adjs[a].cl[idx]==c0

        assert self.adjs[a].next[idx] == -1 and self.adjs[a].prev[idx] == -1

        if self.clst[c0].adjnext == -1:
            self.clst[c0].adjnext = a
        else:
            next_a = self.clst[c0].adjnext
            next_idx = 0 if self.adjs[next_a].cl[0]==c0 else 1
            self.clst[c0].adjnext = a
            self.adjs[a].next[idx] = next_a
            self.adjs[next_a].prev[next_idx] = a


    # build a new cluster by merging two adjacent ones
    # implemented as a DSO - disjount set union - using union-find structure
    cdef inline void merge(self, int merged_adj):
        cdef cluster_t cA = self.adjs[merged_adj].cl[0]
        cdef cluster_t cB = self.adjs[merged_adj].cl[1]
        # print()
        # self.check_list_order(cA)
        # self.check_list_order(cB)
        # print(f'merge {merged_adj}: {cA} {cB}: ', end='', flush=True)
        cdef cluster_t cAB = self.TC # create the new root
        self.TC += 1
        assert self.TC <= self.N
        self.clst[cAB].minR = UMIN(self.clst[cA].minR, self.clst[cB].minR)
        self.clst[cAB].maxR = UMAX(self.clst[cA].maxR, self.clst[cB].maxR)
        self.clst[cAB].minG = UMIN(self.clst[cA].minG, self.clst[cB].minG)
        self.clst[cAB].maxG = UMAX(self.clst[cA].maxG, self.clst[cB].maxG)
        self.clst[cAB].minB = UMIN(self.clst[cA].minB, self.clst[cB].minB)
        self.clst[cAB].maxB = UMAX(self.clst[cA].maxB, self.clst[cB].maxB)
        self.clst[cAB].sumR = self.clst[cA].sumR + self.clst[cB].sumR
        self.clst[cAB].sumG = self.clst[cA].sumG + self.clst[cB].sumG
        self.clst[cAB].sumB = self.clst[cA].sumB + self.clst[cB].sumB
        # self.clst[cAB].sumR2 = self.clst[cA].sumR2 + self.clst[cB].sumR2
        # self.clst[cAB].sumG2 = self.clst[cA].sumG2 + self.clst[cB].sumG2
        # self.clst[cAB].sumB2 = self.clst[cA].sumB2 + self.clst[cB].sumB2
        self.clst[cAB].root = cAB
        self.clst[cAB].area = self.clst[cA].area + self.clst[cB].area
        self.clst[cAB].perimeter = (self.clst[cA].perimeter + 
                                    self.clst[cB].perimeter -
                                    2 * self.adjs[merged_adj].edge_length)
        self.clst[cAB].adjnext = -1

        #TODO: MERGE
        self.clst[cAB].mask_bits = (self.clst[cA].mask_bits | self.clst[cB].mask_bits)

        # make cAB the root of both cA and cB
        assert self.clst[cA].root==cA and self.clst[cB].root==cB # cA and cB are root nodes
        self.clst[cA].root = self.clst[cB].root = cAB 
        self.branches[cAB - self.U][0] = cA
        self.branches[cAB - self.U][1] = cB

        # remove merged_adj from the linked lists of both clusters cA and cB
        reverse_heap_remove(&self.heap, merged_adj)
        self.unlink_adj(cA, merged_adj)
        self.unlink_adj(cB, merged_adj)
        self.adjs[merged_adj].cl[0] = self.adjs[merged_adj].cl[1] = <cluster_t>(-1)

        # merge the remaining linked lists of cA and cB into the new list of cAB
        # keep the descending order, remove all links between internal nodes
        cdef int a, idx, other_idx, lstA, lstB
        cdef int idxA=-1, other_idxA=-1, idxB=-1, other_idxB=-1
        cdef int tailAB=-1, idxAB=-1, prev_a=-1
        cdef bint pickA
        cdef cluster_t thisC, otherC, prev_cl=<cluster_t>(-1)
        lstA = self.clst[cA].adjnext
        lstB = self.clst[cB].adjnext
        self.clst[cA].adjnext = self.clst[cB].adjnext = -1 # unlink the adjacencies
        while lstA!=-1 or lstB!=-1:
            # pick the list with the smallest "other cluster id"
            if lstA!=-1:
                idxA = 0 if self.adjs[lstA].cl[0]==cA else 1
                assert self.adjs[lstA].cl[idxA]==cA
                other_idxA = 0 if idxA==1 else 1
            if lstB!=-1:
                idxB = 0 if self.adjs[lstB].cl[0]==cB else 1
                assert self.adjs[lstB].cl[idxB]==cB
                other_idxB = 0 if idxB==1 else 1

            if lstA!=-1 and lstB==-1:
                pickA = True
            elif lstA==-1 and lstB!=-1:
                pickA = False
            else:
                # print(f' [{<int>self.adjs[lstA].cl[other_idxA]}/{<int>self.adjs[lstB].cl[other_idxB]}]', end='', flush=True)
                pickA = (self.adjs[lstA].cl[other_idxA] >= self.adjs[lstB].cl[other_idxB])

            if pickA: # pick from A
                a, idx, other_idx = lstA, idxA, other_idxA
                thisC, otherC = cA, cB
                lstA = self.adjs[lstA].next[idxA]
            else: # pick from B
                a, idx, other_idx = lstB, idxB, other_idxB
                thisC, otherC = cB, cA
                lstB = self.adjs[lstB].next[idxB]

            # print(f' {<int>self.adjs[a].cl[other_idx]}', end='', flush=True)
            assert self.adjs[a].cl[other_idx]!=otherC

            if prev_cl!=<cluster_t>(-1) and self.adjs[a].cl[other_idx]>=prev_cl:
                # redundant node, free 
                assert self.adjs[a].cl[other_idx]==prev_cl
                self.adjs[prev_a].edge_length += self.adjs[a].edge_length
                self.unlink_adj(self.adjs[a].cl[other_idx], a)
                # remove from the list of adjacencies that can be merged
                reverse_heap_remove(&self.heap, a)
                # print('R', end='', flush=True)
                # invalidate the adjacency
                self.adjs[a].cl[0] = self.adjs[a].cl[1] = <cluster_t>(-1)
                self.adjs[a].next[0] = self.adjs[a].next[1] = -1
                self.adjs[a].prev[0] = self.adjs[a].prev[1] = -1
                self.adjs[a].edge_length = <size_t>(-1)
            else: 
                # link to the cAB list
                self.adjs[a].cl[idx] = cAB
                if tailAB==-1: # first node, make the head
                    self.adjs[a].next[idx] = self.clst[cAB].adjnext
                    self.adjs[a].prev[idx] = -1
                    self.clst[cAB].adjnext = a
                    tailAB, idxAB = a, idx
                else: # append to the tail
                    self.adjs[tailAB].next[idxAB] = a
                    self.adjs[a].next[idx] = -1
                    self.adjs[a].prev[idx] = tailAB
                    tailAB, idxAB = a, idx
                prev_cl = self.adjs[a].cl[other_idx]
                prev_a = a
                # since the linked node passes from thisC to cAB, it needs to be moved
                # to the tail of the linked list of the other adjacent cluster.
                self.unlink_adj(self.adjs[a].cl[other_idx], a)
                self.relink_adj_head(self.adjs[a].cl[other_idx], a)
                # print('=', end='', flush=True)

        # finally, update all heap weights for all edges in the perimeter of cluster cAB
        a = self.clst[cAB].adjnext
        while a >= 0:
            idx = 0 if self.adjs[a].cl[0]==cAB else 1
            assert self.adjs[a].cl[idx]==cAB
            reverse_heap_update_weight(&self.heap, a, self.get_adj_priority(a))            
            a = self.adjs[a].next[idx]

        # print(flush=True)
        # print(f'merge {cA}:{self.clst[cA].perimeter} {cB}:{self.clst[cB].perimeter}  ->  {cAB}:{self.clst[cAB].perimeter}', flush=True)
        # self.check_list_order(cAB)


    # compute the weight of an adjacency between two clusters
    cdef inline double get_adj_priority(self, int a):
        cdef int cl0 = self.adjs[a].cl[0]
        cdef int cl1 = self.adjs[a].cl[1]

        #TODO: IoU
        cdef double partition_score
        cdef uint64_t mask0 
        cdef uint64_t mask1 
        cdef int inter_count
        cdef int union_count
        if self.prebuilt_partitions is not None:
            
            mask0 = self.clst[cl0].mask_bits
            mask1 = self.clst[cl1].mask_bits

            inter_count = popcount64(mask0 & mask1) + 1
            union_count = popcount64(mask0 | mask1) + 1
            # partition_score = 1.0 - <double>inter_count / <double>union_count
            # USE union only as partition_score instead of IoU
            partition_score = <double>union_count
            partition_score = partition_score**2
        else:
            partition_score = 1.0

        # Base quantities
        cdef double area_01 = self.clst[cl0].area + self.clst[cl1].area

        cdef double perim_01 = (self.clst[cl0].perimeter + self.clst[cl1].perimeter - 
                                2 * self.adjs[a].edge_length)
        # assert (self.clst[cl0].perimeter + self.clst[cl1].perimeter > 
        #         2 * self.adjs[a].edge_length)

        cdef int rangeR = (UMAX(self.clst[cl0].maxR, self.clst[cl1].maxR) - 
                           UMIN(self.clst[cl0].minR, self.clst[cl1].minR) + <int>1)
        cdef int rangeG = (UMAX(self.clst[cl0].maxG, self.clst[cl1].maxG) - 
                           UMIN(self.clst[cl0].minG, self.clst[cl1].minG) + <int>1)
        cdef int rangeB = (UMAX(self.clst[cl0].maxB, self.clst[cl1].maxB) - 
                           UMIN(self.clst[cl0].minB, self.clst[cl1].minB) + <int>1)
        assert 1<=rangeR<=256 and 1<=rangeG<=256 and 1<=rangeB<=256

        # V1.0 version (NeurIPS paper)
        if self.use_v0_rule:
            return (rangeR**2 + rangeG**2 + rangeB**2) * area_01 * sqrt(perim_01)


        # new scores
        cdef double area_score, color_range_score, perim_score
        cdef int add_diff = True

        area_score = area_01
        # area_score = 1.0/(abs(self.clst[cl0].area - self.clst[cl1].area) + 1)
        # area_score = area_01 - abs(self.clst[cl0].area - self.clst[cl1].area) + 1
        # area_score = area_01 - sqrt(abs(self.clst[cl0].area - self.clst[cl1].area)) + 1 #NO
        # area_score = area_score**1.2
        # area_score += (area_01 - self.clst[cl0].area)**2 + (area_01 - self.clst[cl1].area)**2 NO
        # area_score = sqrt(area_01)
        assert area_score>0

        # DIFF
        if add_diff:
            area_score += abs((area_01 - self.clst[cl0].area) * (area_01 - self.clst[cl1].area))


        perim_score = perim_01
        # perim_score = perim_01 - sqrt(4 * M_PI * area_01)
        # perim_score = (0.25 * perim_score)**2
        perim_score = perim_score**2
        # perim_score = max(perim_01**2 - self.adjs[a].edge_length**2, 1)
        assert perim_score>0, f'{perim_01} {self.adjs[a].edge_length}'
        
        # DIFF
        if add_diff:
            perim_score += (perim_01 - self.clst[cl0].perimeter)**2
            perim_score += (perim_01 - self.clst[cl1].perimeter)**2


        # cdef double circle_perim = 2 * M_PI * sqrt(area_01 / M_PI)
        # perim_score = perim_01 / circle_perim

        color_range_score = (rangeR**2 + rangeG**2 + rangeB**2)
        # color_range_score = sqrt((rangeR**2 + rangeG**2 + rangeB**2) / 3.0) / 256.0
        # assert 0 <= color_range_score <= 1.0
        # color_range_score = (rangeR + rangeG + rangeB)


        # DIFF
        cdef double rangediffR_0, rangediffR_1, rangediffG_0, rangediffG_1, rangediffB_0, rangediffB_1
        if add_diff:
            rangediffR_0 = rangeR - (self.clst[cl0].maxR - self.clst[cl0].minR + 1)
            rangediffR_1 = rangeR - (self.clst[cl1].maxR - self.clst[cl1].minR + 1)
            rangediffG_0 = rangeG - (self.clst[cl0].maxG - self.clst[cl0].minG + 1)
            rangediffG_1 = rangeG - (self.clst[cl1].maxG - self.clst[cl1].minG + 1)
            rangediffB_0 = rangeB - (self.clst[cl0].maxB - self.clst[cl0].minB + 1)
            rangediffB_1 = rangeB - (self.clst[cl1].maxB - self.clst[cl1].minB + 1)
            color_range_score += (rangediffR_0**2 + rangediffR_1**2 +
                                rangediffG_0**2 + rangediffG_1**2 +
                                rangediffB_0**2 + rangediffB_1**2)

        cdef double area_0 = self.clst[cl0].area
        cdef double ER_0 = self.clst[cl0].sumR / area_0
        cdef double EG_0 = self.clst[cl0].sumG / area_0
        cdef double EB_0 = self.clst[cl0].sumB / area_0
        # cdef double VarR_0 = (self.clst[cl0].sumR2 / area_0) - ER_0*ER_0
        # cdef double VarG_0 = (self.clst[cl0].sumG2 / area_0) - EG_0*EG_0
        # cdef double VarB_0 = (self.clst[cl0].sumB2 / area_0) - EB_0*EB_0

        cdef double area_1 = self.clst[cl1].area
        cdef double ER_1 = self.clst[cl1].sumR / area_1
        cdef double EG_1 = self.clst[cl1].sumG / area_1
        cdef double EB_1 = self.clst[cl1].sumB / area_1
        # cdef double VarR_1 = (self.clst[cl1].sumR2 / area_1) - ER_1*ER_1
        # cdef double VarG_1 = (self.clst[cl1].sumG2 / area_1) - EG_1*EG_1
        # cdef double VarB_1 = (self.clst[cl1].sumB2 / area_1) - EB_1*EB_1
        # assert VarR_1>=0 and VarG_1>=0 and VarB_1>=0

        cdef double color_mean_score
        cdef double distanceR = abs(ER_0 - ER_1)
        cdef double distanceG = abs(EG_0 - EG_1)
        cdef double distanceB = abs(EB_0 - EB_1)
        color_mean_score = (distanceR**2 + distanceG**2 + distanceB**2)
        # color_mean_score = sqrt((distanceR**2 + distanceG**2 + distanceB**2) / 3.0) / 256.0
        # assert 0 <= color_mean_score <= 1.0
        # color_mean_score = (distanceR + distanceG + distanceB)

        # cdef double color_var_score
        # color_var_score = (VarR_01 + VarG_01 + VarB_01)**0.25 + 1

        cdef double ER_01, EG_01, EB_01
        if add_diff:
            ER_01 = (self.clst[cl0].sumR + self.clst[cl1].sumR) / area_01
            EG_01 = (self.clst[cl0].sumG + self.clst[cl1].sumG) / area_01
            EB_01 = (self.clst[cl0].sumB + self.clst[cl1].sumB) / area_01
            # cdef double VarR_01 = (self.clst[cl0].sumR2 + self.clst[cl1].sumR2) / area_01 - ER_01*ER_01
            # cdef double VarG_01 = (self.clst[cl0].sumG2 + self.clst[cl1].sumG2) / area_01 - EG_01*EG_01
            # cdef double VarB_01 = (self.clst[cl0].sumB2 + self.clst[cl1].sumB2) / area_01 - EB_01*EB_01
            # DIFF
            color_mean_score += ((ER_01-ER_0)**2 + (ER_01-ER_1)**2 +
                                 (EG_01-EG_0)**2 + (EG_01-EG_1)**2 +
                                 (EB_01-EB_0)**2 + (EB_01-EB_1)**2)
        # color_var_score = sqrt(VarR_01 + VarG_01 + VarB_01 + 1)
        # color_score += color_mean_score #+ color_var_score
        # color_score = sqrt(color_score)

        # cdef double BD_R = bhattacharyya_distance(ER_0, VarR_0, ER_1, VarR_1)
        # cdef double BD_G = bhattacharyya_distance(EG_0, VarG_0, EG_1, VarG_1)
        # cdef double BD_B = bhattacharyya_distance(EB_0, VarB_0, EB_1, VarB_1)
        # color_score = (BD_R + BD_G + BD_B) + 0.001
        # print(ER_0, VarR_0, area_0, '', ER_1, VarR_1, area_1, '', BD_R)


        # color_score = (rangeR + rangeG + rangeB)
        # color_score = (rangeR + rangeG + rangeB)**2

        # perim_score -= sqrt(4 * M_PI * area_01)
        # assert perim_score >= 0.0
        # if perim_score < 1.0:
        #     perim_score = 1.0

        # area_score = sqrt(area_score)
        # perim_score = sqrt(perim_score)

        # Morphological score combination
        cdef double color_score, shape_score, score
        color_score = color_range_score + color_mean_score #* color_var_score
        # color_score = color_score**2
        shape_score = area_score + perim_score

        # Combined score
        # score = 0.5*(color_score/255) + (shape_score / (self.W+self.H))
        score = color_score * shape_score # (half) geometric mean
        # score = color_score + shape_score # (half) arithmetic mean
        # score = harmonic_mean(color_score, shape_score)
        # score = abs(color_score - shape_score) / abs(log(color_score) - log(shape_score))
        assert score > 0

        # if random.random() > 0.999:
        #     print(f'{cl0}+{cl1} {score}=f({color_score}, {shape_score})')

        # Randomization
        if self.use_randomization != 0.0:
            score *= 1.0 + self.use_randomization * (random.random() - 0.5)

        #TODO:IoU
        if self.prebuilt_partitions is not None:
            score *= partition_score

        return score


    def get_cluster_of_xy(self, x, y):
        cdef cluster_t cl = y*self.W + x
        return self.get_cluster_of(cl)


    def get_cluster_of(self, cluster_t cl):
        assert cl <= self.TC
        while self.clst[cl].root != cl:
            cl = self.clst[cl].root
        return cl


    def merge_adjacency(self, merged_adj):
        return self.merge(merged_adj)


    def get_adjacency_to_merge(self):
        if reverse_heap_size(&self.heap) == 0:
            return None
        a = reverse_heap_top(&self.heap)
        return a
    

    def get_adjacency(self, cl):
        assert cl <= self.TC and self.clst[cl].root == cl
        lst = []
        cdef int a, idx, other_idx
        a = self.clst[cl].adjnext
        while a>= 0:
            idx = 0 if self.adjs[a].cl[0]==cl else 1
            assert self.adjs[a].cl[idx]==cl
            other_idx = 0 if idx==1 else 1
            lst.append(self.adjs[a].cl[other_idx])
            a = self.adjs[a].next[idx]
        return lst
    

    # build the BPT
    def compute(self):
        cdef int a
        # built the BPT by pairwise-merging clusters, in increasing adjacency weight order
        while reverse_heap_size(&self.heap) > 0:
            a = reverse_heap_top(&self.heap)
            # print(f'merge {self.adjs[a].cl[0]}, {self.adjs[a].cl[1]}')
            self.merge(a)
            # self.merge(self.adjs[a].cl[0], self.adjs[a].cl[1])
        assert self.N == self.TC


    # BPT recursive encoding data structures
    cdef cnp.ndarray pixels
    cdef cnp.ndarray leaf_idx
    cdef cnp.ndarray cl_start
    cdef cnp.ndarray cl_end
    cdef cnp.ndarray cl_left
    cdef cnp.ndarray cl_right
    cdef size_t pixel_counter


    # build an encoded representation of the hierarchical clusters
    def encode(self):
        self.pixels   = cnp.ndarray(shape=(self.U), dtype=np.uint32)
        self.leaf_idx = cnp.ndarray(shape=(self.U), dtype=np.uint32)
        self.cl_start = cnp.ndarray(shape=(self.N - self.U), dtype=np.uint32)
        self.cl_end   = cnp.ndarray(shape=(self.N - self.U), dtype=np.uint32)
        self.cl_left  = cnp.ndarray(shape=(self.N - self.U), dtype=np.uint32)
        self.cl_right = cnp.ndarray(shape=(self.N - self.U), dtype=np.uint32)
        self.pixel_counter = 0

        self.visit_tree(self.TC - 1)

        return (self.W, self.H, self.U, self.N, 
                self.pixels, self.leaf_idx,
                self.cl_start, self.cl_end,
                self.cl_left, self.cl_right)


    # recursively visit the binary partition tree, storing the encoding
    cdef int visit_tree(self, unsigned int i):
        assert i < self.TC
        # print('visit_tree', i)
        cdef size_t j
        if i < self.U: # unitary cluster (single pixel)
            self.pixels[self.pixel_counter] = i
            self.leaf_idx[i] = self.pixel_counter
            self.pixel_counter += 1
        else: # multi-pixel cluster
            j = i - self.U
            self.cl_start[j] = self.pixel_counter
            self.cl_left[j]  = self.visit_tree(self.branches[j][0])
            self.cl_right[j] = self.visit_tree(self.branches[j][1])
            self.cl_end[j]   = self.pixel_counter
        return i


######################################################################################################
# Support for index arrays
######################################################################################################

# separate all flat indices (indices of a 2d-image in ravel() form)
# into two sets, at the left or at the right of xval
def split_ravel_indices_by_x(cell_indices, int width, int height, int xval):
    left, right = [], []
    cdef int idx, x
    for idx in cell_indices:
        x = idx % width
        if x < xval:
            left.append(idx)
        else:
            right.append(idx)
    return left, right

# same but at the top/bottom of yval
def split_ravel_indices_by_y(cell_indices, int width, int height, int yval):
    top, bottom = [], []
    cdef int idx, y
    for idx in cell_indices:
        y = idx // width
        if y < yval:
            top.append(idx)
        else:
            bottom.append(idx)
    return top, bottom

def index_array_bbox(cell_indices, int width, int height):
    cdef int xmin = width, ymin = height, xmax = -1, ymax = -1
    cdef int idx, x, y
    for idx in cell_indices:
        y = idx // width
        x = idx % width
        if xmin > x:    xmin = x
        if xmax < x:    xmax = x
        if ymin > y:    ymin = y
        if ymax < y:    ymax = y
    return (xmin, xmax+1, ymin, ymax+1)

def compute_partition_properties(partition_matrix, num_partitions):
    # Initialize arrays to store areas and centroid coordinates sums
    area = np.zeros(num_partitions, dtype=np.int32)
    sum_x = np.zeros(num_partitions, dtype=np.double)
    sum_y = np.zeros(num_partitions, dtype=np.double)
    
    # Get the dimensions of the matrix
    cdef int rows, cols
    cdef int i, j, partition_no

    rows, cols = partition_matrix.shape

    # Perform a single scan of each pixel
    for i in range(rows):
        for j in range(cols):
            partition_no = partition_matrix[i, j]
            area[partition_no] += 1
            sum_x[partition_no] += i
            sum_y[partition_no] += j
    
    # Calculate the centroid coordinates
    div_area = np.copy(area)
    div_area[div_area == 0] = 1 # avoid divisions by 0
    center_x = sum_x / div_area
    center_y = sum_y / div_area

    return area, center_x, center_y

######################################################################################################









