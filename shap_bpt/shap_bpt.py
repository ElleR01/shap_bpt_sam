# cython: language_level=3
# cython: initializedcheck=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

######################################################################################################
# High-performance Binary Partition Tree builder.
######################################################################################################

import numpy as np
cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport sqrt, M_PI, log

from libc.stdint cimport uint64_t

ctypedef unsigned char uint8_t


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

cdef size_t reverse_heap_top(reverse_heap_t* p_heap):
    return p_heap.heap[0]

cdef size_t reverse_heap_size(reverse_heap_t* p_heap):
    return p_heap.heap_size

cdef reverse_heap_update_weight(reverse_heap_t* p_heap, size_t elem, double w):
    cdef ssize_t i = p_heap.rev_heap[elem]
    assert i >= 0 and i < (<ssize_t>p_heap.heap_size)
    p_heap.W[elem] = w
    reverse_heap_percolate_up_or_down(p_heap, i)

######################################################################################################

ctypedef unsigned int cluster_t

#TODO: ADDED MASK_BITS
cdef struct cluster_descr:
    uint8_t minR, maxR
    uint8_t minG, maxG
    uint8_t minB, maxB
    size_t area
    size_t perimeter
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
    cdef bint       use_color_term
    cdef bint       use_area_term
    cdef bint       use_perim_term

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
    def __init__(self, image, use_8ways=True,
                 use_color_term=True, use_area_term=True, use_perim_term=True, prebuilt_partitions=None ):
        self.prebuilt_partitions = prebuilt_partitions 
        
        self.use_8ways = use_8ways

        self.use_color_term = use_color_term
        self.use_area_term = use_area_term
        self.use_perim_term = use_perim_term

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
            self.clst[i].minR = self.clst[i].maxR = image[y,x,0]
            self.clst[i].minG = self.clst[i].maxG = image[y,x,1] if self.C==3 else 0
            self.clst[i].minB = self.clst[i].maxB = image[y,x,2] if self.C==3 else 0
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
        self.adjs[merged_adj].cl[0] = self.adjs[merged_adj].cl[1] = <cluster_t>-1

        # merge the remaining linked lists of cA and cB into the new list of cAB
        # keep the descending order, remove all links between internal nodes
        cdef int a, idx, other_idx, lstA, lstB
        cdef int idxA=-1, other_idxA=-1, idxB=-1, other_idxB=-1
        cdef int tailAB=-1, idxAB=-1, prev_a=-1
        cdef bint pickA
        cdef cluster_t thisC, otherC, prev_cl=<cluster_t>-1
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
                pickA = (self.adjs[lstA].cl[other_idxA] >= self.adjs[lstB].cl[other_idxB])

            if pickA: # pick from A
                a, idx, other_idx = lstA, idxA, other_idxA
                thisC, otherC = cA, cB
                lstA = self.adjs[lstA].next[idxA]
            else: # pick from B
                a, idx, other_idx = lstB, idxB, other_idxB
                thisC, otherC = cB, cA
                lstB = self.adjs[lstB].next[idxB]

            assert self.adjs[a].cl[other_idx]!=otherC

            if prev_cl!=<cluster_t>-1 and self.adjs[a].cl[other_idx]>=prev_cl: 
                # redundant node, free 
                assert self.adjs[a].cl[other_idx]==prev_cl
                self.adjs[prev_a].edge_length += self.adjs[a].edge_length
                self.unlink_adj(self.adjs[a].cl[other_idx], a)
                # remove from the list of adjacencies that can be merged
                reverse_heap_remove(&self.heap, a)
                # invalidate the adjacency
                self.adjs[a].cl[0] = self.adjs[a].cl[1] = <cluster_t>-1
                self.adjs[a].next[0] = self.adjs[a].next[1] = -1
                self.adjs[a].prev[0] = self.adjs[a].prev[1] = -1
                self.adjs[a].edge_length = <size_t>-1
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

        # finally, update all heap weights for all edges in the perimeter of cluster cAB
        a = self.clst[cAB].adjnext
        while a >= 0:
            idx = 0 if self.adjs[a].cl[0]==cAB else 1
            assert self.adjs[a].cl[idx]==cAB
            reverse_heap_update_weight(&self.heap, a, self.get_adj_priority(a))            
            a = self.adjs[a].next[idx]

    
    # compute the weight of an adjacency between two clusters
    cdef inline double get_adj_priority(self, int a):
        cdef int cl0 = self.adjs[a].cl[0]
        cdef int cl1 = self.adjs[a].cl[1]
        cdef double area_score, color_score, perim_score

        #TODO: IoU
        cdef double partition_iou
        cdef uint64_t mask0 
        cdef uint64_t mask1 
        cdef int inter_count
        cdef int union_count
        if self.prebuilt_partitions is not None:
            
            mask0 = self.clst[cl0].mask_bits
            mask1 = self.clst[cl1].mask_bits

            inter_count = popcount64(mask0 & mask1) + 1
            union_count = popcount64(mask0 | mask1) + 1
            partition_iou = <double>inter_count / <double>union_count
        else:
            partition_iou = 1.0

        area_score = self.clst[cl0].area + self.clst[cl1].area

        assert (self.clst[cl0].perimeter + self.clst[cl1].perimeter > 
                2 * self.adjs[a].edge_length)

        perim_score = (self.clst[cl0].perimeter + self.clst[cl1].perimeter - 
                       2 * self.adjs[a].edge_length)

        cdef int rangeR = (UMAX(self.clst[cl0].maxR, self.clst[cl1].maxR) - 
                           UMIN(self.clst[cl0].minR, self.clst[cl1].minR) + <int>1);
        cdef int rangeG = (UMAX(self.clst[cl0].maxG, self.clst[cl1].maxG) - 
                           UMIN(self.clst[cl0].minG, self.clst[cl1].minG) + <int>1);
        cdef int rangeB = (UMAX(self.clst[cl0].maxB, self.clst[cl1].maxB) - 
                           UMIN(self.clst[cl0].minB, self.clst[cl1].minB) + <int>1);

        assert 1<=rangeR<=256 and 1<=rangeG<=256 and 1<=rangeB<=256

        color_score = (rangeR**2 + rangeG**2 + rangeB**2)

        # return color_score * area_score * sqrt(perim_score)

        # for ablation study
        cdef double score = 1.0
        if self.use_color_term:
            score *= color_score
        if self.use_area_term:
            score *= area_score
        if self.use_perim_term:
            score *= sqrt(perim_score)
        
        #TODO:IoU
        if self.prebuilt_partitions is not None:
            score *= (1.0 - partition_iou)

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




