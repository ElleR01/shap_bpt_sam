import matplotlib.pyplot as plt
import heapq as heapq
from tqdm.auto import tqdm
import numpy as np
import math
from scipy.spatial import distance_matrix

import torch

######################################################################################################
# Shapley image explanations with data-dependent Binary Partition Trees
######################################################################################################

# Cython implementation of the BPT algorithm
from . import bpt as bpt

######################################################################################################

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for Shapley values - similar to 'seismic' but with lighter tones.
colormap_alt_b2r = \
    LinearSegmentedColormap.from_list("shapley_values_colormap_b2r", 
                                      [(0.0, '#0053d1'),
                                       (0.2, '#248df4'),
                                       (0.5, 'white'),  
                                       (0.8, '#f23754'),
                                       (1.0, '#cb0021')])

colormap_alt_b2r_reversed = \
    LinearSegmentedColormap.from_list("shapley_values_colormap_b2r_reversed", 
                                      [(0.0, '#33ccff'),
                                       (0.5, 'black'),  
                                       (1.0, '#ff3300')])

colormap_alt_r2b = \
    LinearSegmentedColormap.from_list("shapley_values_colormap_r2b", 
                                      [(0.0, '#ff3300'),
                                       (0.5, 'white'),  
                                       (1.0, '#33ccff')])

colormap_alt_r2b_reversed = \
    LinearSegmentedColormap.from_list("shapley_values_colormap_r2b_reversed", 
                                      [(0.0, '#ff3300'),
                                       (0.5, 'black'),  
                                       (1.0, '#33ccff')])

colormap_default = colormap_alt_b2r

######################################################################################################

# surrogate of isinstance() that does not require to import 
# any module, just compares class names
def is_class_or_superclass_byname(cls, class_name):
    # Check if the current class name matches
    if cls.__name__ == class_name:
        return True
    # Recursively check all superclasses
    for base in cls.__bases__:
        if base.__name__ == class_name:
            return True
        # Recursively check the superclasses of the base class
        if base.__name__!=cls and is_class_or_superclass_byname(base, class_name):
            return True
    
    # If no match is found
    return False

######################################################################################################

def deterministic_random(a, b, c, d, e):
    prime1 = 87838613
    prime2 = 21717043
    prime3 = 60926329
    prime4 = 66817727
    prime5 = 66852631

    combined_value = (a * prime1 + b * prime2 + c * prime3 + d * prime4 + e * prime5) % (2**32)
    assert combined_value >= 0

    # normalize between 0.0 and 1.0
    random_value = combined_value / (2**32 - 1)
    assert 0.0 <= random_value <= 1.0
    
    return random_value

######################################################################################################

class BaseSegment:
    def __init__(self, parent=None):
        self.parent = parent
        
    def split(self):
        raise Exception()
    
    def fill_mask(self, mat, ascend_hier=True):
        return
        
    def add_inside_coalition(self, shap_values, contrib):
        raise Exception()

    def set_image_value(self, image, value):
        raise Exception()
    
    def get_average_image_value(self, image):
        raise Exception()
        
    def area(self):
        raise Exception()

    def contains(self, aa):
        raise Exception()

    def equals(self, other):
        raise Exception()

    def get_bounding_box(self):
        raise Exception()
    
######################################################################################################
# A symmetric, disjoint, axis-aligned, hierarchical partition 
######################################################################################################

class AxisAlignedSegment(BaseSegment):
    def __init__(self, xmin, xmax, ymin, ymax, parent):
        super().__init__(parent)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def midpoint(self, base, length):
        return base + length // 2
    
    def make_new(self, xmin, xmax, ymin, ymax, parent):
        return AxisAlignedSegment(xmin, xmax, ymin, ymax, parent)
        
    #@override
    def split(self, lparent, rparent):
        size_x = self.xmax - self.xmin
        size_y = self.ymax - self.ymin
        assert size_x>=1 or size_y>=1
        lxmin = rxmin = self.xmin
        lxmax = rxmax = self.xmax
        lymin = rymin = self.ymin
        lymax = rymax = self.ymax
        if size_x > size_y and size_x > 1: # split over x
            lxmax = rxmin = self.midpoint(self.xmin, size_x)
        else: # split over y
            lymax = rymin = self.midpoint(self.ymin, size_y)
        lsg = self.make_new(lxmin, lxmax, lymin, lymax, lparent)
        rsg = self.make_new(rxmin, rxmax, rymin, rymax, rparent)
        # print(f'split {self.area()} -> {lsg.area()} + {rsg.area()}    {self}')
        return (lsg, rsg)
    
    #@override
    def fill_mask(self, mat, ascend_hier=True):
        mat[self.ymin:self.ymax, self.xmin:self.xmax] = True
        if ascend_hier:
            self.parent.fill_mask(mat, ascend_hier) 
        
    #@override
    def add_inside_coalition(self, shap_values, contrib):
        for c in range(len(contrib)):
            shap_values[c, self.ymin:self.ymax, self.xmin:self.xmax] += contrib[c]

    #@override
    def set_image_value(self, image, value):
        image[ self.ymin:self.ymax, self.xmin:self.xmax ] = value

    #@override
    def get_average_image_value(self, image):
        return np.mean(np.mean(image[self.ymin:self.ymax, 
                                     self.xmin:self.xmax, ::], axis=1), axis=0)
    
    #@override
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    #@override
    def contains(self, aa):
        return (self.xmin <= aa.xmin and aa.xmax <= self.xmax and
                self.ymin <= aa.ymin and aa.ymax <= self.ymax)

    #@override
    def equals(self, other):
        if not isinstance(other, AxisAlignedSegment):
            return False
        return (self.xmin == other.xmin and other.xmax == self.xmax and
                self.ymin == other.ymin and other.ymax == self.ymax)

    #@override
    def get_bounding_box(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)
    
    # @classmethod
    # def make_root(dim0, dim1):
    #     AxisAlignedSegment(0, dim0, 0, dim1, BaseSegment())

######################################################################################################
# Mondrian segments
######################################################################################################

class MondrianSegment(AxisAlignedSegment):
    def __init__(self, xmin, xmax, ymin, ymax, rand_seed, parent):
        super().__init__(xmin, xmax, ymin, ymax, parent)
        self.rand_seed = rand_seed

    #@override
    def midpoint(self, base, length):
        if length <= 4: # side is too small, cut in half
            return base + length // 2
        rnd = deterministic_random(self.xmin, self.xmax, self.ymin, self.ymax, self.rand_seed)
        return base + int(length * (0.25 + 0.5*rnd))
    
    #@override
    def make_new(self, xmin, xmax, ymin, ymax, parent):
        return MondrianSegment(xmin, xmax, ymin, ymax, self.rand_seed, parent)

    #@override
    def equals(self, other):
        if not isinstance(other, MondrianSegment):
            return False
        return (self.xmin == other.xmin and other.xmax == self.xmax and
                self.ymin == other.ymin and other.ymax == self.ymax and
                self.rand_seed == other.rand_seed)


######################################################################################################
# Binary Partition Tree reader
######################################################################################################

class BPT:
    def __init__(self):
        self.width = self.height = -1
        self.N = self.U = 0
        self.pixels = None
        self.leaf_idx = None
        self.cl_start = self.cl_end = None
        self.cl_left = self.cl_right = None

    # def load_from_file(self, bpt_fname):
    #     with open(bpt_fname, 'r') as f:
    #         self.width = int(f.readline())
    #         self.height = int(f.readline())
    #         self.U = int(f.readline())
    #         self.N = int(f.readline())
    #         self.pixels = np.array([int(n) for n in f.readline().split()])
    #         self.leaf_idx = np.array([int(n) for n in f.readline().split()])
    #         self.cl_start = np.array([int(n) for n in f.readline().split()])
    #         self.cl_end = np.array([int(n) for n in f.readline().split()])
    #         self.cl_left = np.array([int(n) for n in f.readline().split()])
    #         self.cl_right = np.array([int(n) for n in f.readline().split()])

    def from_bpt_builder(self, bpt_builder, use_torch):
        enc = bpt_builder.encode()
        (self.width, self.height, self.U, self.N, 
         self.pixels, self.leaf_idx,
         self.cl_start, self.cl_end,
         self.cl_left, self.cl_right) = enc
        
        if use_torch is not None:
            self.pixels = torch.from_numpy(self.pixels.astype(int))

        # print(f'width={self.width}')
        # print(f'height={self.height}')
        # print(f'U={self.U}')
        # print(f'N={self.N}')
        # print(f'pixels={self.pixels}')
        # print(f'leaf_idx={self.leaf_idx}')
        # print(f'cl_start={self.cl_start}')
        # print(f'cl_end={self.cl_end}')
        # print(f'cl_left={self.cl_left}')
        # print(f'cl_right={self.cl_right}')

    def print_tree(self, index=None, lvl=0):
        if index is None: index = self.N-1
        print(' ' * lvl, end='')
        print(f'index={index} ', end='')
        if index < self.U: # leaf node
            pass
            # print(f' pixel {self.pixels[index]}')
        else:
            s = self.cl_start[ index - self.U ]
            e = self.cl_end[ index - self.U ]
            l, r = self.cl_left[ index - self.U ], self.cl_right[ index - self.U ]
            al = 1 if l < self.U else self.cl_end[ l - self.U ] - self.cl_start[ l - self.U ]
            ar = 1 if r < self.U else self.cl_end[ r - self.U ] - self.cl_start[ r - self.U ]
            print(f'  {e-s} -> {al} + {ar}    left={l} right={r}')
            self.print_tree(self.cl_left[ index - self.U ], lvl+1)
            self.print_tree(self.cl_right[ index - self.U ], lvl+1)

######################################################################################################

def add_noise(img, sigma=1.0, alpha=0.5):
    from scipy.ndimage import gaussian_filter
    assert 0.0 <= alpha <= 1.0
    rndgen = np.random.Generator(np.random.PCG64(1234))
    img_noise = rndgen.standard_normal(size=img.shape)*64.0 + 128.0
    img_noise = gaussian_filter(img_noise, sigma=1.0)
    img = np.clip(img*alpha + img_noise*(1.0-alpha), 0.0, 255.0)
    return img

######################################################################################################

def image_rgb2lab(rgb_image):
    from skimage.color import rgb2lab
    lab_image = rgb2lab(rgb_image)# / 255.0)
    # The ranges of Lab values are: L (0:100), a (-128:127), b (-128:127)
    lab_image_scaled = (lab_image + [0, 128, 128]) * (255.0/100.0, 255.0/256.0, 255.0/256.0)
    return lab_image_scaled.astype(np.uint8)

######################################################################################################

# input image is expected to be of type uint8, with shape H*W*3 or H*W*1
def build_bpt_from_image(image, use_lab=True, use_torch=False, **kwargs):
    if image.dtype!=np.uint8:
        raise Exception('Image pixel type is expected to be uint8.')
    if len(image.shape)==2:
        image = image.reshape((image.shape[0], image.shape[1], 1))
    if len(image.shape)!=3:
        raise Exception('Image shape is expected to be 3-dimensional.')
    if image.shape[2]!=3 and image.shape[2]!=1:
        raise Exception('Image is expected to be RGB (H*W*3) or grayscale (H*W*1).')

    if use_lab:
        image = image_rgb2lab(image)
        # import cv2 as cv2
        # image = cv2.cvtColor(image[:, :, ::-1], cv2.COLOR_BGR2LAB)[:, :, ::-1]

    bpt_builder = bpt.BinaryPartitionTreeBuilder(image=image, **kwargs)
    bpt_builder.compute()
    bptree = BPT()
    bptree.from_bpt_builder(bpt_builder, use_torch)
    del bpt_builder
    return bptree

######################################################################################################
# A non-symmetric, disjoint, hierarchical partition of a Binary Partition Tree node
######################################################################################################

class BPT_Segment(BaseSegment):
    def __init__(self, bpt, index, parent):
        super().__init__(parent)
        self.bpt = bpt
        self.index = index

    #@override
    def split(self, lparent, rparent):
        if self.area() == 1:
            return None
        ls = BPT_Segment(self.bpt, self.bpt.cl_left[ self.index - self.bpt.U ], lparent)
        rs = BPT_Segment(self.bpt, self.bpt.cl_right[ self.index - self.bpt.U ], rparent)
        # print(f'split {self.area()} -> {ls.area()} + {rs.area()}    {self.index} -> {ls.index}|{rs.index}  {self}')
        return (ls, rs)
    
    #@override
    def fill_mask(self, mat, ascend_hier=True):
        s,e = self.pixels_interval()
        if isinstance(mat, torch.Tensor):
            mat.view(-1)[ self.bpt.pixels[s:e] ] = True        
        else:
            mat.ravel()[ self.bpt.pixels[s:e] ] = True
        if ascend_hier:
            self.parent.fill_mask(mat, ascend_hier)
            
    #@override
    def add_inside_coalition(self, shap_values, contrib):
        s,e = self.pixels_interval()
        for c in range(len(contrib)):
            shap_values[c].ravel()[ self.bpt.pixels[s:e] ] += contrib[c]
         
    #@override
    def set_image_value(self, image, value):
        shape = (image.shape[0]*image.shape[1], ) + image.shape[2::]
        s,e = self.pixels_interval()
        image.reshape(shape)[ self.bpt.pixels[s:e] ] = value

    #@override
    def get_average_image_value(self, image):
        shape = (image.shape[0]*image.shape[1], ) + image.shape[2::]
        s,e = self.pixels_interval()
        return np.mean(image.reshape(shape)[ self.bpt.pixels[s:e] ], axis=0)

    #@override
    def area(self):
        s,e = self.pixels_interval()
        return float(e - s)

    #@override
    def pixels_interval(self):
        if self.index < self.bpt.U: # leaf node
            return (self.bpt.leaf_idx[self.index], 
                    self.bpt.leaf_idx[self.index] + 1)
        else:
            return (self.bpt.cl_start[ self.index - self.bpt.U ],
                    self.bpt.cl_end[ self.index - self.bpt.U ])

    #@override
    def contains(self, other):
        s1, e1 = self.pixels_interval()
        s2, e2 = other.pixels_interval()
        return s1 <= s2 and e2 <= e1

    #@override
    def equals(self, other):
        if not isinstance(other, BPT_Segment):
            return False
        s1, e1 = self.pixels_interval()
        s2, e2 = other.pixels_interval()
        return s1 == s2 and e2 == e1

    #@override
    def get_bounding_box(self):
        s,e = self.pixels_interval()
        pixels = self.bpt.pixels[s:e]
        xc, yc = pixels % self.bpt.width, pixels // self.bpt.width
        return (np.min(xc), np.max(xc), np.min(yc), np.max(yc))
    
######################################################################################################
# General segmentation-based coalition structures
######################################################################################################

# Arbitrary segmentation defined by an integer partition matrix
class PartitionMatrix:
    # determine the Voronoi cell index of each pixel
    def __init__(self, partition_matrix):
        assert len(partition_matrix.shape)==2
        assert np.issubdtype(partition_matrix.dtype, np.integer)
        # assert partition_matrix.dtype in (np.dtype('int32'), np.dtype('int64'))
        self.partition_matrix = partition_matrix
        self.width = self.partition_matrix.shape[1]
        self.height = self.partition_matrix.shape[0]

        self.num_partitions = np.max(self.partition_matrix) + 1
        
        # ravelled indices of each partition
        self.pixel_indices = [ np.where(self.partition_matrix.ravel() == n)[0] for n in range(self.num_partitions)]

        area, center_x, center_y = bpt.compute_partition_properties(self.partition_matrix, self.num_partitions)
        self.area = area
        self.center_x = center_x
        self.center_y = center_y

        # set of cells with non-zero area
        self.nonempty_segments = [i for i in range(len(self.area)) if self.area[i]>0]

    def plot(self):
        plt.imshow(self.partition_matrix, cmap='prism')
        plt.scatter(self.center_y, self.center_x, c='w', edgecolors='k')
        plt.show()

######################################################################################################

def voronoi_segmentation(width, height, num_points):
    centers_x, centers_y = [], []
    sq = int(math.sqrt(num_points))
    for i in range(num_points):
        centers_x.append(round(np.random.rand(1)[0]* (width / sq * i)) % width)
        centers_y.append(round(np.random.rand(1)[0]* (height / sq * i)) % height)
    A = [(x+.5,y+.5) for x in range(width) for y in range(height)]
    B = [(centers_x[i], centers_y[i]) for i in range(num_points)]
    D = distance_matrix(A,B, p=2)
    partition_matrix = np.argmin(D,axis=1).reshape((width,height)).astype(np.int32)
    return PartitionMatrix(partition_matrix)

######################################################################################################

# Segment made by a group of partitions
class MultiPartition(BaseSegment):
    def __init__(self, parts_idx, area, partition_matrix, parent):
        super().__init__(parent)
        self.parts_idx = parts_idx
        self.parts_area = area
        self.partition_matrix = partition_matrix

    #@override
    def split(self, lparent, rparent):
        assert len(self.parts_idx) >= 1
        cx = [self.partition_matrix.center_x[i] for i in self.parts_idx]
        cy = [self.partition_matrix.center_y[i] for i in self.parts_idx]
        size_x = np.max(cx) - np.min(cx)
        size_y = np.max(cy) - np.min(cy)

        if size_x >= size_y and size_x > 1: # split over x
            sorted_parts = sorted([(self.partition_matrix.center_x[i], i) for i in self.parts_idx])
        else: # split over y
            sorted_parts = sorted([(self.partition_matrix.center_y[i], i) for i in self.parts_idx])
        mid = len(sorted_parts)//2
        lparts = [i for _,i in sorted_parts[:mid]]
        rparts = [i for _,i in sorted_parts[mid:]]
        larea, rarea = 0, 0
        for c in lparts:
            larea += self.partition_matrix.area[c]
        for c in rparts:
            rarea += self.partition_matrix.area[c]
        assert larea>0 and rarea>0
        lsg = self.split_multipartition(lparts, larea, lparent)
        rsg = self.split_multipartition(rparts, rarea, rparent)
        return [ lsg, rsg ]
    
    # is the parts_idx[] set still made by multiple parts_idx, or just one?
    def split_multipartition(self, parts_idx, area, parent):
        assert len(parts_idx) >= 1
        if len(parts_idx) > 1:
            return MultiPartition(parts_idx, area, self.partition_matrix, parent)
        else:
            return AxisAlignedSubPartition(self.partition_matrix.pixel_indices[parts_idx[0]], 
                                           self.partition_matrix, parent)

    #@override
    def fill_mask(self, mat, ascend_hier=True):
        for lc in self.parts_idx:
            mat.ravel()[ self.partition_matrix.pixel_indices[lc] ] = True
        if ascend_hier:
            self.parent.fill_mask(mat, ascend_hier)
        
    #@override
    def add_inside_coalition(self, shap_values, contrib):
        for c in range(len(contrib)):
            for lc in self.parts_idx:
                shap_values[c].ravel()[ self.partition_matrix.pixel_indices[lc] ] += contrib[c]

    #@override
    def set_image_value(self, image, value):
        shape = (image.shape[0]*image.shape[1], ) + image.shape[2::]
        flat_image = image.reshape(shape)
        for c in self.parts_idx:
            flat_image[ self.partition_matrix.pixel_indices[c]  ] = value

    #@override
    def get_average_image_value(self, image):
        shape = (image.shape[0]*image.shape[1], ) + image.shape[2::]
        flat_image = image.reshape(shape)
        clr, cnt = np.zeros(image.shape[2::]), 0
        for c in self.parts_idx:
            clr += np.sum(flat_image[ self.partition_matrix.pixel_indices[c] ], axis=0)
            cnt += self.partition_matrix.area[c]
        return clr / cnt

    #@override
    def area(self):
        return self.parts_area  
    
    #@override
    def contains(self, other):
        raise Exception('Unimplemented')

    #@override
    def equals(self, other):
        if not isinstance(other, MultiPartition):
            return False
        return (self.parts_idx==other.parts_idx and 
                self.partition_matrix==other.partition_matrix)

    #@override
    def get_bounding_box(self):
        for i, lc in enumerate(self.parts_idx):
            _xmin, _xmax, _ymin, _ymax = bpt.index_array_bbox(self.partition_matrix.pixel_indices[lc], 
                                                              self.partition_matrix.width, 
                                                              self.partition_matrix.height)
            if i==0:
                xmin, xmax, ymin, ymax = _xmin, _xmax, _ymin, _ymax
            else:
                xmin, xmax = min(xmin, _xmin), max(xmax, _xmax)
                ymin, ymax = min(ymin, _ymin), max(ymax, _ymax)
        return (xmin, xmax, ymin, ymax)

######################################################################################################

# Segment made by an axis-aligned partition of a single Partition in the partition matrix
class AxisAlignedSubPartition(BaseSegment):
    def __init__(self, indices, partition_matrix, parent):
        super().__init__(parent)
        self.indices = indices
        self.partition_matrix = partition_matrix
        xmin, xmax, ymin, ymax = bpt.index_array_bbox(self.indices, self.partition_matrix.width, 
                                                      self.partition_matrix.height)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
    #@override
    def split(self, lparent, rparent):
        size_x = self.xmax - self.xmin
        size_y = self.ymax - self.ymin
        if size_x <= 1 and size_y <= 1:
            return None # cannot split
        assert self.area() > 1
        if size_x >= size_y and size_x > 1: # split over x
            xval = (self.xmin + size_x // 2)
            splitted = bpt.split_ravel_indices_by_x(self.indices, self.partition_matrix.width, 
                                                    self.partition_matrix.height, xval)
        else: # split over y
            yval = (self.ymin + size_y // 2)
            splitted = bpt.split_ravel_indices_by_y(self.indices, self.partition_matrix.width, 
                                                    self.partition_matrix.height, yval)
        
        # split and recompute exact cell boundaries and area
        lsg = AxisAlignedSubPartition(splitted[0], self.partition_matrix, lparent)
        rsg = AxisAlignedSubPartition(splitted[1], self.partition_matrix, rparent)
        return [ lsg, rsg ]
    
    #@override
    def fill_mask(self, mat, ascend_hier=True):
        mat.ravel()[ self.indices ] = True
        if ascend_hier:
            self.parent.fill_mask(mat, ascend_hier)
        
    #@override
    def add_inside_coalition(self, shap_values, contrib):
        for c in range(len(contrib)):
            shap_values[c].ravel()[ self.indices ] += contrib[c]

    #@override
    def set_image_value(self, image, value):
        shape = (image.shape[0]*image.shape[1], ) + image.shape[2::]
        image.reshape(shape)[ self.indices ] = value

    #@override
    def get_average_image_value(self, image):
        shape = (image.shape[0]*image.shape[1], ) + image.shape[2::]
        return np.mean(image.reshape(shape)[ self.indices ], axis=0)

    #@override
    def area(self):
        return len(self.indices)

    #@override
    def contains(self, aa):
        raise Exception()

    #@override
    def equals(self, other):
        if not isinstance(other, AxisAlignedSubPartition):
            return False
        return (self.indices==other.indices and 
                self.partition_matrix==other.partition_matrix and
                self.xmin <= other.xmin and other.xmax <= self.xmax and
                self.ymin <= other.ymin and other.ymax <= self.ymax)

    #@override
    def get_bounding_box(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

######################################################################################################





















































######################################################################################################
# A partition of the features inside the coalition structure, made by two sets:
#  - Q = a persistent coalition
#  - T = a recursively refinable coalition
# The segment represent the partition, and it is a chain of segments:
#  - The full chain is Q u T
#  - The last element of the chain is T
# A coalition object represents the marginal contribution of nu(Q u T) - nu(Q)
# Assumes: intersection(Q, T) = emptyset
######################################################################################################

class Coalition:
    def __init__(self, segment, weight):
        self.segment = segment  # segment/partition of this coalition
        # self.segment alone is T, self.segment.parent and above is Q (the persistent set)
        self.weight  = weight   # recursive weight of the Owen formula

        self.nu_QuT   = None    # nu(Q u T)
        self.nu_Q     = None    # nu(Q)
        self.priority = None


    def set_marginal_contributions(self, nu_QuT, nu_Q):
        self.nu_QuT = nu_QuT  # contribution of Q u T
        self.nu_Q   = nu_Q    # contribution of Q without T
        # priority to be split for further partition refinements
        self.priority = -np.max(np.abs(np.subtract(self.nu_QuT, self.nu_Q))) * self.weight 


    def prepare_split(self, explainer):
        # split the current Owen coalition (Q,T).
        # Assume the downward expansion in the partition hierarchy:  down(T) = {A, B}

        # Define each sub-partitions in terms of their persistent coalition Q' and their new partition T'
        # sgm_Q_A:   Q'=Q, T'=A      sgm_Q_B:   Q'=Q, T'=B
        sgm_Q_A,   sgm_Q_B   = self.segment.split(self.segment.parent, self.segment.parent)
        # sgm_QuB_A: Q'=QuB, T'=A    sgm_QuA_B: Q'=QuA, T'=B
        sgm_QuB_A, sgm_QuA_B = self.segment.split(sgm_Q_B, sgm_Q_A) # flip parents
        assert self.segment.area() == sgm_Q_A.area() + sgm_Q_B.area()
        
        # build the new masks for (Q u A) and (Q u B)
        m_QuA, m_QuB = explainer.empty_mask(), explainer.empty_mask()
        sgm_Q_A.fill_mask(m_QuA)
        sgm_Q_B.fill_mask(m_QuB)
                
        # generate the four recursive branches
        #                                                   w  |  Q' | T'| nu(Q'uT') -> nu(Q')
        co_Q_A   = Coalition(sgm_Q_A,   self.weight/2.0) # w/2 |  Q  | A |  nu(QuA)  -> nu(Q)
        co_Q_B   = Coalition(sgm_Q_B,   self.weight/2.0) # w/2 |  Q  | B |  nu(QuB)  -> nu(Q)
        co_QuA_B = Coalition(sgm_QuA_B, self.weight/2.0) # w/2 | QuA | B |  nu(QuT)  -> nu(QuA)
        co_QuB_A = Coalition(sgm_QuB_A, self.weight/2.0) # w/2 | QuB | A |  nu(QuT)  -> nu(QuB)

        splits = [co_Q_A, co_Q_B, co_QuA_B, co_QuB_A]
        # self.check_efficiency(explainer, splits)
        return (m_QuA, m_QuB, splits)


    def complete_split(self, splits, nu_QuA, nu_QuB):
        co_Q_A, co_Q_B, co_QuA_B, co_QuB_A = splits
        # set marginals:                    nu(Q'uT') -> nu(Q')
        co_Q_A  .set_marginal_contributions(nu_QuA,      self.nu_Q)
        co_Q_B  .set_marginal_contributions(nu_QuB,      self.nu_Q)
        co_QuA_B.set_marginal_contributions(self.nu_QuT, nu_QuA)     # Note that T = A u B
        co_QuB_A.set_marginal_contributions(self.nu_QuT, nu_QuB)

        
    def plot(self, ax, explainer):
        mQ = explainer.empty_mask('float')
        mT = explainer.empty_mask('float')
        self.segment.parent.fill_mask(mQ, ascend_hier=True) # Q = persistent set
        self.segment.fill_mask(mT, ascend_hier=False) # T = refinable partition
        ax.imshow(mQ*0.5 + mT, cmap='gray', vmin=0.0, vmax=1.0) 
        ax.set_xticks([]) ; ax.set_yticks([])
    

    def __lt__(self, other):
        return self.priority < other.priority
    

    def get_shapley(self, shap_values): 
        # compute the weighted marginals and add them to the partition
        contrib = (np.subtract(self.nu_QuT, self.nu_Q) * self.weight)
        contrib = contrib / self.segment.area()
        self.segment.add_inside_coalition(shap_values, contrib)


    # def check_efficiency(self, explainer, splits):
    #     # The sum of Shapley values for each coalition member is the value of the grand coalition
    #     shap_values1 = np.zeros((explainer.num_explained_classes, 
    #                              explainer.image_to_explain.shape[0], 
    #                              explainer.image_to_explain.shape[1]))
    #     shap_values2 = np.zeros_like(shap_values1)
    #     self.get_shapley(shap_values1)
    #     for s in splits:
    #         s.get_shapley(shap_values2)
    #     print(np.sum(shap_values1[0,:,:]), np.sum(shap_values2[0,:,:]))


######################################################################################################
# Explainer object. Implementation of the recursive refinement following Owen formula
######################################################################################################

class Explainer:
    def __init__(self, nu, image_to_explain=None, num_explained_classes=None, 
                 explained_class_ids=None, verbose=False, torch_device=None, soften_masks=False):
        self.nu = nu # characteristic function (black box model taking boolean masks as input)
        self.verbose = verbose
        self.soften_masks = soften_masks

        predicted_nuN = predicted_nu0 = None

        # get foreground and background predictions
        if is_class_or_superclass_byname(nu.__class__, 'MultiClassCharacteristicFunction'):
            predicted_nuN = nu.predicted_nuN
            predicted_nu0 = nu.predicted_nu0
            assert explained_class_ids is None
            explained_class_ids = range(nu.num_explained_classes)
            assert image_to_explain is None
            self.image_to_explain = nu.masked_model.image_to_explain
            assert torch_device is None
            self.torch_device = nu.masked_model.device
        else:
            self.torch_device = torch_device
            self.image_to_explain = image_to_explain
            predicted_nuN = self.nu(self.to_device(self.expand_dims(self.full_mask())))[0]
            predicted_nu0 = self.nu(self.to_device(self.expand_dims(self.empty_mask())))[0]

        assert isinstance(self.image_to_explain, np.ndarray)
        assert self.image_to_explain.dtype == np.dtype('uint8') # expects 8-bit image
        assert len(self.image_to_explain.shape)==3
        assert self.image_to_explain.shape[2] in [1,3,4]

        # determine the explained class
        if explained_class_ids is None: # explain the N most probable classes
            assert num_explained_classes is not None
            self.output_indexes = np.flip(np.argsort(predicted_nuN))[:num_explained_classes]
            self.num_explained_classes = num_explained_classes
        else: # explain the classes in the explained_class_ids[] array
            assert num_explained_classes is None
            self.output_indexes = np.array(explained_class_ids)
            self.num_explained_classes = len(self.output_indexes)

        self.base_nuN = predicted_nuN[self.output_indexes]
        self.base_nu0 = predicted_nu0[self.output_indexes]
        
        if self.verbose >= 2:
            print('Explained indexes: ', self.output_indexes)
            print(' nu(N):', self.base_nuN)
            print(' nu(0):', self.base_nu0)


    def empty_mask(self, dtype='bool'):
        if self.torch_device is None:
            return np.zeros((self.image_to_explain.shape[0], 
                             self.image_to_explain.shape[1]), 
                             dtype=dtype)
        else:
            return torch.zeros((self.image_to_explain.shape[0], 
                                self.image_to_explain.shape[1]), 
                            #    device=self.torch_device,
                               dtype=self.torch_type(dtype))


    def full_mask(self, dtype='bool'):
        if self.torch_device is None:
            return np.ones((self.image_to_explain.shape[0], 
                            self.image_to_explain.shape[1]), 
                            dtype=dtype)
        else:
            return torch.ones([self.image_to_explain.shape[0], 
                               self.image_to_explain.shape[1]], 
                            #   device=self.torch_device,
                              dtype=self.torch_type(dtype))
        

    def expand_dims(self, x):
        return np.expand_dims(x, axis=0) if self.torch_device is None else x.unsqueeze(0)
    

    def to_device(self, x):
        return x if self.torch_device is None else x.to(device=self.torch_device)
    

    def torch_type(self, dtype):
        if dtype=='bool':  return torch.bool
        if dtype=='float': return torch.float32
        raise Exception('undefined')


    def soften_mask(self, m):
        from skimage.filters import gaussian
        from torchvision.transforms.functional import gaussian_blur
        # m2 = gaussian(m, 8, channel_axis=-1)
        # print(m.shape, m.dtype)
        m2 = gaussian_blur(m.reshape([1]+list(m.shape)).type(torch.float32), kernel_size=7, sigma=7).reshape(m.shape)
        # plt.imshow(m, cmap='gray', vmin=0, vmax=1) ; plt.show()
        # plt.imshow(m2, cmap='gray', vmin=0, vmax=1) ; plt.show()
        return m2


    # get an explanation of the image_to_explain masked by @boolMask
    def predict_masked(self, masks):
        if self.soften_masks:
            masks = [self.soften_mask(m) for m in masks]
        if self.torch_device is None:
            preds = self.nu(np.array(masks))
        else:
            preds = self.nu(torch.stack(masks, dim=0).to(device=self.torch_device))

        preds = preds[:, self.output_indexes]
        # we expect the characteristic function nu to return an ndarray
        assert isinstance(preds, np.ndarray) 
        assert np.issubdtype(preds.dtype, np.floating)
        assert len(preds.shape)==2
        assert preds.shape[0]==len(masks)
        assert preds.shape[1]==self.num_explained_classes

        return preds
    

    # get the Owen approximation of the Shapley coefficients
    def explain_instance(self, max_evals, method='BPT', bpt=None,
                         batch_size=64, verbose_plot=False, pbar=None,
                         min_area=1, max_weight=None, callback=None):
        assert min_area >= 1

        if self.verbose:
            pbar = pbar if pbar is not None else tqdm(total=max_evals, disable=False, leave=False)
 
        if isinstance(method, BaseSegment):
            sgm_root = method
        elif isinstance(method, list):
            assert callback is None
            # ensemble mode - average multiple coalition structures
            avg_shap_values = np.mean(
                [self.explain_instance(max_evals=max_evals//len(method), 
                                       method=sgm_root,
                                       batch_size=batch_size, 
                                       verbose_plot=verbose_plot, pbar=pbar,
                                       min_area=min_area, max_weight=max_weight)
                 for sgm_root in method], axis=0)
            return avg_shap_values
        elif method=='BPT':
            if bpt is None:
                bpt = build_bpt_from_image(self.image_to_explain, 
                                           use_torch=(self.torch_device is not None))
            sgm_root = BPT_Segment(bpt, bpt.N-1, BaseSegment())
        elif method=='AA':
            sgm_root = AxisAlignedSegment(0, self.image_to_explain.shape[1],
                                          0, self.image_to_explain.shape[0], BaseSegment())
        else:
            print('Unknown method', method) ; return None

        # initialize recursive partitioning
        coalitions_queue = [] # heap queue sorted by coalition priority
        unitary_coalitions = []         # reached unitary (undivisible) coalitions
        # initial coalition: w=1, Q'=emptyset, T'=root, nu(Q'uT')=nu(N) -> nu(Q')=nu(0)
        init_coalition = Coalition(sgm_root, 1.0) 
        init_coalition.set_marginal_contributions(self.base_nuN, self.base_nu0)
        heapq.heappush(coalitions_queue, init_coalition)
        eval_count = 0

        # apply the recursive Owen formula until the evaluation budget @max_evals is consumed
        while len(coalitions_queue)>0 and eval_count < max_evals:
            if callback is not None:
                callback(self, eval_count, max_evals, 
                         coalitions_queue, unitary_coalitions)

            # build next evaluation batch
            batch_masks = []
            batch_splits = []
            while len(coalitions_queue)>0 and len(batch_masks) < batch_size and \
                  eval_count + len(batch_masks) < max_evals:
                coalition = heapq.heappop(coalitions_queue)
                if (coalition.segment.area() <= min_area or
                    (max_weight is not None and coalition.weight<=max_weight)): 
                    unitary_coalitions.append(coalition) # do not split further
                else:
                    (m_QuA, m_QuB, splits) = coalition.prepare_split(self)
                    batch_masks.append(m_QuA)
                    batch_masks.append(m_QuB)
                    batch_splits.append((coalition, splits))

            # evaluate the batch and complete the splits
            if len(batch_masks) > 0:
                payoffs = self.predict_masked(batch_masks)
                eval_count += len(batch_masks)
                if self.verbose: 
                    pbar.update(len(batch_masks))

                for i in range(len(batch_splits)):
                    # assign the marginal constributions to the four splits
                    nu_QuA, nu_QuB = payoffs[i*2], payoffs[i*2 + 1]
                    coalition, splits = batch_splits[i]
                    coalition.complete_split(splits, nu_QuA, nu_QuB)
                    for s in splits:
                        heapq.heappush(coalitions_queue, s)
                    if verbose_plot:
                        plotted = [coalition] + splits
                        fig,axes = plt.subplots(1, 5, figsize=(5,1))
                        for i, s in enumerate(plotted):
                            s.plot(axes[i], self)
                        plt.show()
        
        if callback is not None:
            callback(self, eval_count, max_evals, 
                     coalitions_queue, unitary_coalitions)

        # collect Shapley values
        shap_values = self.build_saliency_map(coalitions_queue, unitary_coalitions)

        if self.verbose:
            pbar.refresh()
            if len(unitary_coalitions) > 0 and self.verbose >= 2: 
                print(f'Reached {len(unitary_coalitions)} unitary coalitions.')
        return shap_values
        
    
    def build_saliency_map(self, coalitions_queue, unitary_coalitions):
        shap_values = np.zeros((self.num_explained_classes, 
                                self.image_to_explain.shape[0], 
                                self.image_to_explain.shape[1]))
        # for coalition in coalitions_queue:
        #     coalition.get_shapley(shap_values)
        # for coalition in unitary_coalitions:
        #     coalition.get_shapley(shap_values)
        if self.soften_masks:
            for coalition in coalitions_queue + unitary_coalitions:
                mask = self.empty_mask('float')
                coalition.segment.fill_mask(mask, ascend_hier=False)
                mask = np.array(self.soften_mask(mask))
                contrib = (np.subtract(coalition.nu_QuT, coalition.nu_Q) * coalition.weight)
                contrib = contrib / coalition.segment.area()
                for c in range(len(contrib)):
                    shap_values[c] += contrib[c] * mask
        else:
            for coalition in coalitions_queue:
                coalition.get_shapley(shap_values)
            for coalition in unitary_coalitions:
                coalition.get_shapley(shap_values)

        return shap_values


######################################################################################################

def plot_shapley_values(explainer, shap_values, names=None, class_names=None, show=True,  
                        alpha=0.80, cmap=colormap_default, show_nu_values=True):
    shap_values = np.array(shap_values)
    if len(shap_values.shape)==3: shap_values = np.array([shap_values])
    max_val = np.nanpercentile(np.abs(shap_values.flatten()), 99.9)
    num_explained_classes = len(explainer.base_nuN)
    num_rows = len(shap_values)
    fig,axes = plt.subplots(num_rows+1, num_explained_classes+1, 
                            figsize=(2*(num_explained_classes+1), 2*(num_rows+0.3)), 
                            squeeze=False,
                            height_ratios=[1]*num_rows + [0.3])
    base_image = explainer.image_to_explain
    if np.max(base_image)>1: base_image = base_image.astype(np.uint8)
    if len(base_image.shape)==2:
        base_image = np.stack([base_image, base_image, base_image], axis=-1)
    base_image_is_grayscale = base_image.shape[2]==1
    if base_image_is_grayscale:
        img_grey = base_image[:, :, 0]
    else:
        img_grey = (0.2989 * base_image[:, :, 0] +
                    0.5870 * base_image[:, :, 1] + 
                    0.1140 * base_image[:, :, 2])
    if 'reversed' in cmap.name:
        img_grey = 1.0 - img_grey
    for r in range(num_rows):
        axes[r,0].imshow(base_image, cmap='gray' if base_image_is_grayscale else None)
        for i in range(num_explained_classes):
            axes[r,i+1].imshow(img_grey.astype(base_image.dtype), cmap='gray')
            im=axes[r,i+1].imshow(shap_values[r,i], cmap=cmap, vmin = -max_val, vmax = max_val, alpha=alpha)
            if r==0: 
                if is_class_or_superclass_byname(explainer.nu.__class__, 'MultiClassCharacteristicFunction'):
                    column_title = f'{explainer.output_indexes[i]}: {explainer.nu.class_names[explainer.output_indexes[i]]}'
                elif class_names is not None:
                    column_title = f'{explainer.output_indexes[i]}: {class_names[explainer.output_indexes[i]]}'
                else:
                    column_title = f'{explainer.output_indexes[i]}'
                if show_nu_values:
                    # column_title += f'\n$\\nu(N)={explainer.base_nuN[i]:.4} \\rightarrow \\nu(\\varnothing)={explainer.base_nu0[i]:.4}$'
                    column_title += f'\nv(N)={explainer.base_nuN[i]:.4}\nv(0)={explainer.base_nu0[i]:.4}'
                axes[r,i+1].set_title(column_title, fontsize=10)
        for jjj in range(num_explained_classes+1): 
            axes[r,jjj].set_xticks([]) ; axes[r,jjj].set_yticks([])
    if names is not None:
        for r in range(num_rows):
            axes[r,0].set_ylabel(names[r])
    # Use the last row for the colorbar
    for ax in axes[-1,:]:
        ax.set_axis_off()
        # ax.set_box_aspect(0.1)
    cb = fig.colorbar(im, ax=axes[-1,:], label="Shapley value", 
                      orientation="horizontal", aspect=80, fraction=0.9)
    cb.outline.set_visible(False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    if show:
        plt.show()


######################################################################################################
######################################################################################################

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for Shapley values - similar to 'seismic' but with lighter tones.
shapley_values_colormap = LinearSegmentedColormap.from_list("shapley_values_colormap", 
                                                            [(0.0, '#0053d1'),
                                                             (0.2, '#248df4'),
                                                             (0.5, 'white'),  
                                                             (0.8, '#f23754'),
                                                             (1.0, '#cb0021')])

######################################################################################################
######################################################################################################

def plot_owen_values(explainer, shap_values, class_names, names=None):
    """
    Visualize ShapBPT explanations.

    Parameters
    ----------
    explainer : Explainer
        Fitted explanation object.

    shap_values : np.ndarray
        Explanation maps.

    class_names : list[str]
        Names of model output classes.

    names : list[str], optional
        Row labels for multiple explanation sets.

    Returns
    -------
    None
        Displays a matplotlib figure.
    """
    shap_values = np.array(shap_values)
    if len(shap_values.shape)==3: shap_values = np.array([shap_values])
    max_val = np.nanpercentile(np.abs(shap_values.flatten()), 99.9)
    num_explained_classes = len(explainer.base_nuN)
    num_rows = len(shap_values)
    fig,axes = plt.subplots(num_rows+1, num_explained_classes+1, 
                            figsize=(2*(num_explained_classes+1), 2*(num_rows+0.3)), 
                            squeeze=False,
                            height_ratios=[1]*num_rows + [0.3])
    base_image = explainer.image_to_explain
    if np.max(base_image)>1: base_image = base_image.astype(np.uint8)
    if len(base_image.shape)==2:
        base_image = np.stack([base_image, base_image, base_image], axis=-1)
    img_grey = (0.2989 * base_image[:, :, 0] +
                0.5870 * base_image[:, :, 1] + 
                0.1140 * base_image[:, :, 2])
    # axes[0].set_title(f'real: {class_names[expected_class]}')
    for r in range(num_rows):
        axes[r,0].imshow(base_image)
        for i in range(num_explained_classes):
            axes[r,i+1].imshow(img_grey.astype(base_image.dtype), alpha=0.50, cmap='gray')
            im=axes[r,i+1].imshow(shap_values[r,i], cmap=shapley_values_colormap, vmin = -max_val,
                                   vmax = max_val, alpha=0.80)
            if r==0: axes[r,i+1].set_title(f'{class_names[explainer.output_indexes[i]]}', fontsize=10)#+
                                #f'\n{explainer.base_f_S[i]:.5} to {explainer.base_f_0[i]:.5}')
        for jjj in range(num_explained_classes+1): axes[r,jjj].set_xticks([]) ; axes[r,jjj].set_yticks([])
    if names is not None:
        for r in range(num_rows):
            axes[r,0].set_ylabel(names[r])
    # Use the last row for the colorbar
    for ax in axes[-1,:]:
        ax.set_axis_off()
        # ax.set_box_aspect(0.1)
    cb = fig.colorbar(im, ax=axes[-1,:], label="Shapley/Owen value", 
                      orientation="horizontal", aspect=80, fraction=0.9)#, location='bottom') #,  fraction=0.5, 
    cb.outline.set_visible(False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # plt.tight_layout()
    plt.show()

######################################################################################################
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255.0 for i in range(0, lv, lv // 3))