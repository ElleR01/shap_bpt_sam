######################################################################################################
import numpy as np
import os
# import json
# import sys, os, importlib, math
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import cv2
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms
import imageio.v3 as iio
# from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from skimage.transform import resize

import shap_bpt as shap_bpt

######################################################################################################
# Generate replacement values for images
######################################################################################################

def create_replacements(image_to_explain, backgrounds='bgwsn'):
    grayscale = (image_to_explain.shape[2] == 1)
    replacement_image_set = []
    if 'b' in backgrounds:
        replacement_image_set.append(np.full_like(image_to_explain, 0))
    if 'g' in backgrounds:
        replacement_image_set.append(np.full_like(image_to_explain, 128 if grayscale else (123, 116, 103)))
    if 'w' in backgrounds:
        replacement_image_set.append(np.full_like(image_to_explain, 255))
    if 's' in backgrounds:
        replacement_image_set.append((gaussian(image_to_explain, 8, channel_axis=-1)*255).astype(np.uint8))
    if 'n' in backgrounds:
        x = np.clip(np.random.normal(128, 128, size=image_to_explain.shape), 0, 255).astype(np.uint8)
        replacement_image_set.append((gaussian(x, 2.0, channel_axis=-1) * 255).astype(np.uint8))

    return np.array(replacement_image_set)

######################################################################################################
# Masked classifier dealing with boolean masks
######################################################################################################

class BaseMaskedImageModel:
    def __init__(self, device, model):
        self.device = device
        self.model = model
        self.image_to_explain = None

    def prepare_for_image(self, image_to_explain):
        assert isinstance(image_to_explain, np.ndarray)
        assert image_to_explain.dtype == np.dtype('uint8') # expects 8-bit image
        assert len(image_to_explain.shape)==3
        assert image_to_explain.shape[2] in [1,3,4]

        self.image_to_explain = image_to_explain
        # Explained image
        # image_to_explain_preproc = self.preprocess(self.image_to_explain)
        image_to_explain_preproc = self.preprocess(self.image_to_explain.astype(np.float32) / 255.0)
        self.image_to_explain_tensor = image_to_explain_preproc.to(self.device)
        
    def preprocess(self, image):
        return torch.tensor(image).to(self.device)

    def predict(self, x):
        raise Exception('Must be implemented in the subclass')

    def class_name(self, i):
        raise Exception('Must be implemented in the subclass')

    def __call__(self, masks):
        raise Exception('Must be implemented in the subclass')

    def repr(self):
        raise Exception('Must be implemented in the subclass')
    
    def create_mask(self, size=None, num_masks=None, fill_value=0, dtype=torch.bool, on_device=True):
        if size is None:
            if num_masks is None: # sigle mask
                size=(self.image_to_explain.shape[0], self.image_to_explain.shape[1])
            else:
                size=(num_masks, self.image_to_explain.shape[0], self.image_to_explain.shape[1])
        return torch.full(size=size, fill_value=fill_value, dtype=dtype,
                          device=self.device if on_device else None)

######################################################################################################
# Masked classifier dealing with boolean masks using a neural network
######################################################################################################

class MaskedImageModel(BaseMaskedImageModel):
    def __init__(self, device, model):
        super().__init__(device, model)
        self.replacement_image_set = None

    def prepare_for_image(self, image_to_explain, replacement_image_set):
        super().prepare_for_image(image_to_explain)

        assert isinstance(replacement_image_set, np.ndarray)
        assert replacement_image_set.dtype == np.dtype('uint8') # expects 8-bit image
        assert len(replacement_image_set.shape)==4
        assert replacement_image_set.shape[1:] == image_to_explain.shape

        # Replacement values
        self.replacement_image_set = replacement_image_set
        replacement_values_preproc = [self.preprocess(bkgnd.astype(np.float32)/255.0)
                                        for bkgnd in self.replacement_image_set]
        # replacement_values_preproc = [self.preprocess(bkgnd)
        #                                 for bkgnd in self.replacement_image_set]

        self.replacement_value_tensors = torch.cat([torch.unsqueeze(rvp, dim=0) 
                                        for rvp in replacement_values_preproc]).to(self.device)
        
    # def preprocess(self, image):
    #     return torch.tensor(image).to(self.device)

    def __call__(self, masks):
        assert isinstance(masks, torch.Tensor)
        N,H,W = masks.shape # N boolean masks with size W*H
        B,_,_,_ = self.replacement_value_tensors.shape
        NCH,_,_ = self.image_to_explain_tensor.shape # number of channels
        # masks_tensor = torch.tensor(np.array(masks)).to(self.device) 
        masks_tensor = torch.reshape(masks, (N,1,H,W)) # N*H*W -> N*1*H*W
        masks_tensor = torch.tile(masks_tensor, dims=(1,NCH,B,1)).reshape(B*N,NCH,H,W) # N*1*H*W -> NB*NCH*H*W
        
        Xf = torch.tile(self.image_to_explain_tensor, dims=(N*B,1,1,1)) # NCH*H*W -> NB*NCH*H*W
        Xb = torch.tile(self.replacement_value_tensors, dims=(N,1,1,1)) # NCH*H*W -> NB*NCH*H*W
        if masks_tensor.dtype == torch.bool:
            X = torch.where(masks_tensor, Xf, Xb) # T=Xf, F=Xb
        else: # torch.is_floating_point(masks_tensor)
            X = masks_tensor*Xf + (1.0 - masks_tensor)*Xb
        result = self.predict(X)
        del X, Xb, Xf, masks_tensor
        result = result.reshape((-1, B, result.shape[1]))
        return np.mean(result, axis=1)

    def repr(self):
        return str(self.backgrounds)
        
######################################################################################################
# Masked classifier for the ImageNet problem (classify the topmost classes, ImageNet preprocessing)
######################################################################################################

class MaskedImageModelForImageNet(MaskedImageModel):
    def __init__(self, device, model, class_names, do_softmax=True):
        super().__init__(device, model)
        self.class_names = class_names
        self.do_softmax = do_softmax
        self.cam_target_layers = None
        # ImageNet preprocessing
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.preprocess_function = transforms.Compose(
            [#transforms.Lambda(lambda img: img.astype(np.float32) / 255.0),  # Rescale uint8 image by 255.0
             transforms.ToTensor(), 
             transforms.Normalize(mean=mean, std=std)]
        )

    #@Override
    def preprocess(self, image):
        return self.preprocess_function(image)
    
    #@Override
    def predict(self, x):
        if self.do_softmax:
            return F.softmax(self.model(x), dim=1).cpu().detach().numpy()
        else:
            return self.model(x).cpu().detach().numpy()
        
    #@Override
    def class_name(self, i):
        return self.class_names[i]
    
    #@Override
    def repr(self):
        return 'resnet50_'+super().repr()

######################################################################################################
# Perfect linear model returning the proportions of pixels inside the ground-truth
######################################################################################################

class MaskedImageGroundTruthModel(BaseMaskedImageModel):
    def __init__(self, device):
        super().__init__(device, model=None)
        self.ground_truth = None
        self.replacement_image_set = []

    def prepare_for_image(self, image_to_explain, ground_truth):
        super().prepare_for_image(image_to_explain)
        assert isinstance(ground_truth, np.ndarray)
        assert ground_truth.dtype == np.dtype('bool')
        assert image_to_explain.shape[:2] == ground_truth.shape

        self.ground_truth = ground_truth
        self.ground_truth_area = np.sum(self.ground_truth)

        # Ground truth values
        self.ground_truth_tensors = torch.tensor(self.ground_truth).to(self.device)
        
    def preprocess(self, image):
        return torch.tensor(image).to(self.device)

    def predict(self, x):
        return self.__call__(x)

    def class_name(self, i):
        assert i==0
        return 'ground_truth'

    def __call__(self, masks):
        assert isinstance(masks, torch.Tensor)
        N,H,W = masks.shape # N boolean masks with size W*H
        # masks_tensor = torch.tensor(masks).to(self.device) 
        # Compute intersection and union for each image in the batch
        if masks.dtype == torch.bool:
            intersection = (masks & self.ground_truth_tensors).sum(dim=(1, 2))  # Sum over h and w dimensions
            # union = (masks | self.ground_truth_tensors).sum(dim=(1, 2))         # Sum over h and w dimensions
        else: # torch.is_floating_point(masks)
            intersection = (masks * self.ground_truth_tensors).sum(dim=(1, 2))  # Sum over h and w dimensions
            # union = torch.max(masks, self.ground_truth_tensors).sum(dim=(1, 2)) # Sum over h and w dimensions
        # Compute IoU for each image in the batch masks[]
        # iou = intersection.float() / union.float()
        # Compute the linear function:  nu(S) = |S \cap G| / |S|
        iou = intersection.float() / self.ground_truth_area
        # Reshape to [n, 1] by adding an extra dimension
        iou = iou.unsqueeze(1)
        return iou.cpu().detach().numpy()

    def repr(self):
        return str(self.backgrounds)

######################################################################################################
# Characteristic function producing multiple payoffs
######################################################################################################

class MultiClassCharacteristicFunction:
    def __init__(self, masked_model,
                 num_explained_classes = None, 
                 explained_class_ids = None,
                 verbose = False):
        self.masked_model = masked_model
        # Foreground image to be explained
        predicted_nuN = self.masked_model(self.masked_model.create_mask(num_masks=1, fill_value=1))[0]

        if explained_class_ids is None: # explain the N most probable classes
            assert num_explained_classes is not None
            self.output_indexes = np.flip(np.argsort(predicted_nuN))[:num_explained_classes]
            self.num_explained_classes = num_explained_classes
        else: # explain the classes in the explained_class_ids[] array
            assert num_explained_classes is None
            self.output_indexes = np.array(explained_class_ids)
            self.num_explained_classes = len(self.output_indexes)

        # Background value 
        predicted_nu0 = self.masked_model(self.masked_model.create_mask(num_masks=1, fill_value=0))[0]

        self.class_names = [self.masked_model.class_name(i) for i in self.output_indexes]
        self.predicted_nuN = predicted_nuN[self.output_indexes]
        self.predicted_nu0 = predicted_nu0[self.output_indexes]

        if verbose:
            for i, idx in enumerate(self.output_indexes):
                print(f'{i:<2}({idx:3})  {self.class_names[i]:20}:   {self.predicted_nuN[i]:.4} -> {self.predicted_nu0[i]:.4}')

    # evaluate the masked model, returning only the selected (multi-)classes
    def __call__(self, masks):
        return self.masked_model(masks)[:, self.output_indexes]
    
    # the amount of outcome explained by this characteristic function. It is \nu(n) - \nu(\varnothing)
    def total_outcome(self):
        return (self.predicted_nuN - self.predicted_nu0)
        
######################################################################################################

def load_rgb_image_from_file(filename, image_size=None):
    image = iio.imread(filename) # Load the image
    if len(image.shape) == 2:
        # Convert grayscale image to 3D by stacking it into RGB format
        image = np.stack((image,)*3, axis=-1)
    if image.shape[2]==4:
        image = image[:, :, 0:3] # drop alpha
    if image_size is not None and image_size != image.shape[:2]:
        image = (resize(image, image_size, anti_aliasing=True) * 255.0).astype(np.uint8)
    return image.astype(np.uint8)

######################################################################################################

def make_characteristic_function_for_image(image_to_explain, masked_model, backgrounds, 
                                           num_explained_classes = None, 
                                           explained_class_ids = None,
                                           verbose=False):
    # image_to_explain = load_rgb_image_from_file(fname, image_size)
    assert isinstance(image_to_explain, np.ndarray)
    assert image_to_explain.dtype == np.dtype('uint8') # expects 8-bit image
    assert len(image_to_explain.shape)==3
    assert image_to_explain.shape[2] in [1,3,4]

    replacement_image_set = create_replacements(image_to_explain, backgrounds)
    masked_model.prepare_for_image(image_to_explain, replacement_image_set)
    if verbose:
        print(f'Image size: {image_to_explain.shape[1]}x{image_to_explain.shape[0]}')
        print(f'Number of replacement images: {len(replacement_image_set)}')
    nu = MultiClassCharacteristicFunction(masked_model, num_explained_classes, 
                                          explained_class_ids, verbose=verbose)
    if verbose:
        print(f'Explained classes: {nu.class_names}')
    return nu

######################################################################################################
# Insertion/Deletion curve AUC scores
######################################################################################################

def saliency_to_auc(nu, heatmap, class_id=0, batch_size=4, method='del', num_samples=101, 
                    rule='trapezoid'):
    assert isinstance(heatmap, np.ndarray)
    assert len(heatmap.shape)==2 and np.issubdtype(heatmap.dtype, np.floating)

    f0, fN = nu.predicted_nu0[class_id], nu.predicted_nuN[class_id]

    # nu_max = max(f_S, f_0)
    # nu_min = min(f_S, f_0)

    xs, ys, ms, masks, qs = [], [], [], [], []
    for i, value in enumerate(np.linspace(start=1.0, stop=0.0, num=num_samples)):
        if method=='del':
            epsilon = (1 if value==0.0 else 0)
            q = (np.quantile(heatmap, q=value) - epsilon)
            m = heatmap <= q
            nx = (1.0 - np.sum(m) / m.size)
        elif method=='ins':
            epsilon = (1 if value==1.0 else 0)
            q = (np.quantile(heatmap, q=value) + epsilon)
            m = heatmap >= q
            nx = (np.sum(m) / m.size)
        else:
            raise Exception()
            
        # add a new datapoint on the curve
        if len(xs)==0 or nx != xs[-1]: 
            assert m.dtype==bool and len(m.shape)==2
            xs.append(nx)
            masks.append(m)
            ms.append(np.sum(heatmap[m]))
            qs.append(q)

        # evaluate the characteristic function
        if len(masks) >= batch_size or (len(masks)>0 and i==(num_samples-1)):
            y = nu(torch.from_numpy(np.array(masks, dtype=bool)).to(device=nu.masked_model.device))[:, 0]
            # y = nu(np.array(masks))[:, class_id]
            ys.extend(y)
            masks = []

    assert len(masks)==0    
    xs, ys = np.array(xs), np.array(ys)
    assert(len(xs) == len(ys))

    # compute considering under/over shoots
    if fN > f0:
        overshoot_max = np.maximum(0, ys - fN) # overshoot for values exceeding the maximum f(S)
        overshoot_min = np.maximum(0, f0 - ys) # overshoot for values below the minimum f(0)
    else: # f(S) < f(0)
        overshoot_max = np.maximum(0, ys - f0) # overshoot for values exceeding the maximum f(0)
        overshoot_min = np.maximum(0, fN - ys) # overshoot for values below the minimum f(S)

    # clip ys, no oveshoots
    y_clipped = np.clip(ys, min(fN, f0), max(fN, f0))
    # adjust ys with the overshoot. Clip it inside the admitted range
    y_adjusted = np.clip(ys - 2*overshoot_max + 2*overshoot_min, min(fN, f0), max(fN, f0))

    # rebase to f(0)
    if fN > f0:
        flipped = False
        ys = ys - f0 
        y_clipped = y_clipped - f0 
        y_adjusted = y_adjusted - f0
    else: # f(S) < f(0)
        flipped = True
        ys = f0 - ys 
        y_clipped = f0 - y_clipped 
        y_adjusted = f0 - y_adjusted

    # rescaling
    ys_rescaled = ys / abs(fN - f0)
    y_clipped_rescaled = y_clipped / abs(fN - f0)
    y_adjusted_rescaled = y_adjusted / abs(fN - f0)

    auc, auc_r, auc_mae, auc_mse, auc_adj, auc_adjr, auc_clip, auc_clipr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    curve_range = range(1, len(xs)) if rule=='trapezoid' else range(len(xs))

    # compute the area under the curve with the midpoint Riemann sum (i.e. the trapezoidal rule)
    for i in curve_range:
        if rule=='trapezoid':
            delta_x = abs(xs[i] - xs[i-1])
            assert delta_x > 0
            y_mid   =         0.5*(ys[i-1] + ys[i])
            y_r_mid =         0.5*(ys_rescaled[i-1] + ys_rescaled[i])
            err_mid = y_mid - 0.5*(ms[i-1] - ms[i])
            y_clip_mid =       0.5*(y_clipped[i-1] + y_clipped[i])
            y_clipr_mid =      0.5*(y_clipped_rescaled[i-1] + y_clipped_rescaled[i])
            y_adj_mid =       0.5*(y_adjusted[i-1] + y_adjusted[i])
            y_adjr_mid =      0.5*(y_adjusted_rescaled[i-1] + y_adjusted_rescaled[i])
        else: # rectangles
            delta_x = 1.0/num_samples if i==len(xs)-1 else abs(xs[i+1] - xs[i])
            assert delta_x > 0
            y_mid   =         ys[i]
            y_r_mid =         ys_rescaled[i]
            err_mid = y_mid - ms[i]
            y_clip_mid =       y_clipped[i]
            y_clipr_mid =      y_clipped_rescaled[i]
            y_adj_mid =       y_adjusted[i]
            y_adjr_mid =      y_adjusted_rescaled[i]


        auc += abs(delta_x * y_mid) # base * height
        auc_r += abs(delta_x * y_r_mid) # base * height
        # auc_eff += abs(delta_x * err_mid) # base * height
        auc_mae += abs(delta_x * err_mid) # base * height
        auc_mse += abs(delta_x * (err_mid**2)) # base * height^2
        auc_clip += abs(delta_x * y_clip_mid)
        auc_clipr += abs(delta_x * y_clipr_mid)
        auc_adj += abs(delta_x * y_adj_mid)
        auc_adjr += abs(delta_x * y_adjr_mid)

    return {'xs':xs, 'ms':ms, 'qs':qs, 
            'f0':f0, 'fN':fN, 'flipped':flipped, 
            'ys':ys, 'ysr':ys_rescaled,
            'y_clip':y_clipped, 'y_clipr':y_clipped_rescaled, 
            'y_adj':y_adjusted, 'y_adjr':y_adjusted_rescaled, 
            'method':method, 'class_id':class_id,
            'auc':auc, 'auc_r':auc_r,
            'auc_mae':auc_mae, 'auc_mse':auc_mse, 'auc_rmse':np.sqrt(auc_mse), 
            'auc_clip':auc_clip, 'auc_clipr':auc_clipr,
            'auc_adj':auc_adj, 'auc_adjr':auc_adjr}
# def saliency_to_auc(nu, heatmap, batch_size=4, method='del', num_samples=101, class_id=0, rule='trapezoid'):
#     assert isinstance(heatmap, np.ndarray)
#     assert len(heatmap.shape)==2 and np.issubdtype(heatmap.dtype, np.floating)

#     nu_max = np.maximum(nu.predicted_nu0[class_id], nu.predicted_nuN[class_id])
#     nu_min = np.minimum(nu.predicted_nu0[class_id], nu.predicted_nuN[class_id])

#     xs, ys, ms, masks, qs = [], [], [], [], []
#     for i, value in enumerate(np.linspace(start=1.0, stop=0.0, num=num_samples)):
#         if method=='del':
#             epsilon = (1 if value==0.0 else 0)
#             q = (np.quantile(heatmap, q=value) - epsilon)
#             m = heatmap <= q
#             nx = (1.0 - np.sum(m) / m.size)
#         elif method=='ins':
#             epsilon = (1 if value==1.0 else 0)
#             q = (np.quantile(heatmap, q=value) + epsilon)
#             m = heatmap >= q
#             nx = (np.sum(m) / m.size)
#         else:
#             raise Exception()
            
#         # add a new datapoint on the curve
#         if len(xs)==0 or nx != xs[-1]: 
#             assert m.dtype==bool and len(m.shape)==2
#             xs.append(nx)
#             masks.append(m)
#             ms.append(np.sum(heatmap[m]))
#             qs.append(q)

#         # evaluate the characteristic function
#         if len(masks) >= batch_size or (len(masks)>0 and i==(num_samples-1)):
#             y = nu(torch.from_numpy(np.array(masks, dtype=bool)).to(device=nu.masked_model.device))[:, 0]
#             ys.extend(y)
#             masks = []

#     assert len(masks)==0    
#     xs, ys = np.array(xs), np.array(ys)
#     assert(len(xs) == len(ys))

#     # compute considering under/over shoots
#     overshoot_max = np.maximum(0, ys - nu_max) # overshoot for values exceeding the maximum
#     overshoot_min = np.maximum(0, nu_min - ys) # overshoot for values below the minimum
#     # adjust ys with the overshoot. Clip it inside the admitted range
#     y_adjusted = np.clip(ys - 2*overshoot_max + 2*overshoot_min, nu_min, nu_max)

#     # rescaling
#     ys_rescaled = (ys - nu_min) / (nu_max - nu_min)
#     y_adjusted_rescaled = (y_adjusted - nu_min) / (nu_max - nu_min)

#     auc, auc_r, auc_mae, auc_mse, auc_adj, auc_adjr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

#     curve_range = range(1, len(xs)) if rule=='trapezoid' else range(len(xs))

#     # compute the area under the curve with the midpoint Riemann sum (i.e. the trapezoidal rule)
#     for i in curve_range:
#         if rule=='trapezoid':
#             delta_x = abs(xs[i] - xs[i-1])
#             assert delta_x > 0
#             y_mid   =         0.5*(ys[i-1] + ys[i])
#             y_r_mid =         0.5*(ys_rescaled[i-1] + ys_rescaled[i])
#             err_mid = y_mid - 0.5*(ms[i-1] - ms[i])
#             y_adj_mid =       0.5*(y_adjusted[i-1] + y_adjusted[i])
#             y_adjr_mid =      0.5*(y_adjusted_rescaled[i-1] + y_adjusted_rescaled[i])
#         else: # rectangles
#             delta_x = 1.0/num_samples if i==len(xs)-1 else abs(xs[i+1] - xs[i])
#             assert delta_x > 0
#             y_mid   =         ys[i]
#             y_r_mid =         ys_rescaled[i]
#             err_mid = y_mid - ms[i]
#             y_adj_mid =       y_adjusted[i]
#             y_adjr_mid =      y_adjusted_rescaled[i]


#         auc += abs(delta_x * y_mid) # base * height
#         auc_r += abs(delta_x * y_r_mid) # base * height
#         # auc_eff += abs(delta_x * err_mid) # base * height
#         auc_mae += abs(delta_x * err_mid) # base * height
#         auc_mse += abs(delta_x * (err_mid**2)) # base * height^2
#         auc_adj += abs(delta_x * y_adj_mid)
#         auc_adjr += abs(delta_x * y_adjr_mid)

#     return {'xs':xs, 'ys':ys, 'ms':ms, 'qs':qs, 'ysr':ys_rescaled,
#             'y_adj':y_adjusted, 'y_adjr':y_adjusted_rescaled, 
#             'method':method, 'class_id':class_id,
#             'auc':auc, 'auc_r':auc_r, #'auc_eff':auc_eff, 
#             'auc_mae':auc_mae, 'auc_mse':auc_mse, 'auc_rmse':np.sqrt(auc_mse), 
#             'auc_adj':auc_adj, 'auc_adjr':auc_adjr}

######################################################################################################
# IoU curve computation and AUC score
######################################################################################################

def calc_IoU_curve(y_true, y_pred):
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_true.shape)==1 and len(y_pred.shape)==1 # assumes y_true and y_pred to be flattened arrays
    assert len(y_true)==len(y_pred)
    assert y_true.dtype==np.dtype('bool') and np.issubdtype(y_pred.dtype, np.floating)

    yd   = np.array(sorted(zip(y_pred, y_true), reverse=True))
    X2   = np.zeros(len(y_pred))
    IoU2 = np.zeros(len(y_pred))
    Th   = np.zeros(len(y_pred))
    
    nT = np.sum(y_true)
    nInt = 0
    for i in range(len(y_pred)):
        if yd[i,1]: 
            nInt += 1
        
        IoU2[i] = nInt / (i + nT - nInt)
        X2[i] = i
        Th[i] = yd[i,0]
        
    X2 = X2 / len(y_pred)
    auc_IoU = 0
    for i in range(1, len(y_pred)):
        auc_IoU += (X2[i] - X2[i-1]) * (IoU2[i] + IoU2[i-1]) / 2.0
    
    best_pt = np.argmax(IoU2)
    return {'X':X2, 'Y':IoU2, 'max_IoU_heatmap_threshold':Th[best_pt], 
            'max_IoU_score':IoU2[best_pt], 'x_best':X2[best_pt], 'auc_IoU':auc_IoU}

######################################################################################################

















# ######################################################################################################
# # LIME
# ######################################################################################################

import hashlib
from skimage.segmentation import quickshift

def binary_search_quickshift(image, target_seg_no, init_max_dist=100):
    def get_segments(image, md):
        segments = quickshift(image, ratio=0.2, kernel_size=4, max_dist=md, random_seed=1234) #, rng=1234)
        return len(np.unique(segments)), segments
    # do a binary search of the target number of segments
    lmd, rmd = 0, init_max_dist
    lsn,_ = get_segments(image, lmd)
    rsn,_ = get_segments(image, rmd)
    niter = 0
    while niter<40 and rsn!=target_seg_no:
        niter += 1
        mmd = (lmd + rmd) / 2.0
        msn,_ = get_segments(image, mmd)
        if msn <= target_seg_no <= lsn:
            rsn, rmd = msn, mmd
        else:
            lsn, lmd = msn, mmd
    return rmd, get_segments(image, rmd)[1]

def binary_search_quickshift_cached(image, target_seg_no, 
                                    cache_dir_prefix='~/CACHE'):
    # prepare unique name for cache
    sha1 = hashlib.sha1( np.ascontiguousarray(image).view(np.uint8) ).hexdigest()
    fname = f'qs_{sha1}_{target_seg_no}'
    fname = os.path.join(os.path.expanduser(cache_dir_prefix.replace('/', os.sep)),
                         fname+'.npy')
    # search cached segmentation or rebuild it
    if not os.path.exists(fname):
        _, sgm = binary_search_quickshift(image, target_seg_no)
        np.save(fname, sgm)

    sgm = np.load(fname)
    return 0.0, sgm

# dist_coeff, segments = binary_search_quickshift_cached(image_to_explain, num_lime_segments)

# ######################################################################################################

# from lime import lime_image
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# from skimage.segmentation import mark_boundaries

# def format_lime_heatmaps(nu, segments, expl):
#     class_heatmaps = []
#     for clsid in expl.top_labels:
#         heatmap = np.zeros_like(segments, dtype=np.float32)
#         for segm, importance in expl.local_exp[clsid]:
#             heatmap[ segments==segm ] += importance 
#         # normalize
#         heatmap = heatmap * (nu.predicted_nuN[clsid] - nu.predicted_nu0[clsid]) / np.sum(heatmap)
#         class_heatmaps.append(heatmap)
#     return np.array(class_heatmaps)

# def explain_with_LIME(nu, num_segments=100, num_samples=1000, 
#                       use_stratification=False, verbose=False):
#     if nu.masked_model.model is None:
#         return None
#     # generate segmentation
#     _, segments = binary_search_quickshift_cached(nu.masked_model.image_to_explain, num_segments)
#     def segments_getter(img):
#         return segments
        
#     def lime_predict(img):
#         return nu.masked_model.model(torch.Tensor(img).permute(0,3,1,2).to(nu.masked_model.device)).cpu().detach().numpy()[:, nu.output_indexes]

#     num_segments = len(np.unique(segments))
#     heatmap_list = []
#     for bg_c in nu.masked_model.replacement_value_tensors:
#         lime_explainer = lime_image.LimeImageExplainer(random_state=1234)
#         lime_expl = lime_explainer.explain_instance(nu.masked_model.image_to_explain_tensor.permute(1,2,0).cpu().detach().numpy(), 
#                                                     lime_predict,
#                                                     top_labels=nu.num_explained_classes,
#                                                     use_stratification=use_stratification,
#                                                     segmentation_fn=segments_getter,
#                                                     hide_color=bg_c.permute(1,2,0).cpu().detach().numpy(), 
#                                                     num_samples=num_samples,
#                                                     progress_bar=True)
#         if isinstance(lime_expl, tuple):
#             lime_expl = lime_expl[2]
#         print('R2:', lime_expl.score)
#         heatmap_list.append(format_lime_heatmaps(nu, segments, lime_expl))
#     return np.mean(heatmap_list, axis=0)

# ######################################################################################################
















######################################################################################################
# Revised LIME
######################################################################################################

from statsmodels.api import WLS, add_constant

# Fit a Weighted Least Squares linear regression model
def weighted_least_squares(X, Y, weights, fit_intercept=True):
    if fit_intercept:
        X = add_constant(X) # Add a constant (intercept term)
    variances = []
    regr_coeffs = []
    for i in range(Y.shape[1]):
        wls_model = WLS(Y[:, i], X, weights=weights).fit()
        cov_matrix = wls_model.cov_params()
        variances.append(np.diag(cov_matrix))
        regr_coeffs.append(wls_model.params)
    regr_coeffs = np.array(regr_coeffs)
    variances = np.array(variances)
    mean_variances = np.mean(variances, axis=0)
    if fit_intercept:
        regr_coeffs = regr_coeffs[:, 1:] # Remove intercept
        variances = variances[:, 1:] # Remove intercept
        mean_variances = mean_variances[1:] # Remove intercept
    return regr_coeffs, variances, mean_variances

######################################################################################################

from sklearn.linear_model import Ridge

def explain_with_revLIME(nu, num_target_segments, num_samples, batch_size, verbose=False):
    h,w,_ = nu.masked_model.image_to_explain.shape
    # generate segmentation
    _, segments = binary_search_quickshift_cached(nu.masked_model.image_to_explain, num_target_segments)
    # generate neighborhood matrix
    random_state = np.random.RandomState(1234)
    num_segments = len(np.unique(segments))
    print(num_target_segments, num_segments)
    masking_mat = random_state.randint(0, 2, num_samples*num_segments, dtype=bool).reshape((num_samples, num_segments))
    masking_mat[0, :] = True
    masking_mat[1, :] = False
    sample_weights = np.ones(num_samples)
    
    # evaluate the characteristic function nu()
    preds = []
    for i in tqdm(range(0, num_samples, batch_size), disable=not verbose):
        i2 = min(i+batch_size, num_samples)
        m = torch.zeros((i2-i, h, w), dtype=bool)
        for j in range(i2-i):
            for z in np.where(masking_mat[i+j] == 0)[0]:
                m[j][segments==z] = True
        for p in nu(m.to(device=nu.masked_model.device)):
            preds.append(p)
    preds = np.array(preds)

    # Fit a Weighted Least Squares linear regression model
    regr_coeffs, variance_coeffs, mean_variances = weighted_least_squares(masking_mat.astype(int), preds, sample_weights)

    # generate pixel-level variance map
    variance_maps = []
    for i in range(nu.num_explained_classes):
        vmap = np.zeros_like(segments, dtype=np.float32)
        for j, sigma2 in enumerate(variance_coeffs[i]):
            vmap[ segments==j ] += sigma2 
        variance_maps.append(vmap)
    
    # generate the pixel-level heatmaps
    lime_coeffs = []
    for i in range(nu.num_explained_classes):
        heatmap = np.zeros_like(segments, dtype=np.float32)
        for j, importance in enumerate(regr_coeffs[i]):
            heatmap[ segments==j ] += importance # TODO: should this be divided by the area?
        # normalize
        heatmap = heatmap * (preds[1, i] - preds[0, i]) / np.sum(heatmap)
        lime_coeffs.append(heatmap)
    return np.array(lime_coeffs), np.array(variance_maps)

######################################################################################################








######################################################################################################
# RISE explanation
######################################################################################################

def RISE_generate_masks(input_size, N, s, p1):
    from skimage.transform import resize
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *input_size))

    for i in range(N):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    return masks.astype(np.float32)

# RISE: Randomized input sampling for explanation of black-box models
def explain_with_RISE(nu, max_evals, s, p1, batch_size):
    H, W, _ = nu.masked_model.image_to_explain.shape
    rise_coeffs = None
    for i in tqdm(range(0, max_evals, batch_size), desc='Evaluating masked images'):
        nn = min(i + batch_size, max_evals) - i
        masks = RISE_generate_masks([H, W], nn, s, p1)
        preds = nu(torch.from_numpy(masks).to(device=nu.masked_model.device))
        # accumulate values
        masks_reshaped = masks.reshape(nn, H * W)
        r = np.dot(preds.T, masks_reshaped)
        rise_coeffs = r if rise_coeffs is None else rise_coeffs+r

    # reshape and normalize
    rise_coeffs = rise_coeffs.reshape((preds.shape[1], H, W)) / max_evals / p1
    for i in range(nu.num_explained_classes):
        rise_coeffs[i] -= np.min(rise_coeffs[i])
        rise_coeffs[i] = rise_coeffs[i] * (nu.predicted_nuN[i] - nu.predicted_nu0[i]) / np.sum(rise_coeffs[i])
    return rise_coeffs

######################################################################################################












######################################################################################################
# LIME over dynamic segmentation
######################################################################################################

from queue import PriorityQueue

def refine_partitions(bptsgm, coeffs, masking_mat, num_splits):
    assert len(bptsgm)==len(coeffs) and len(bptsgm)==masking_mat.shape[1]
    pq = PriorityQueue()
    for i in range(len(bptsgm)):
        pq.put((-coeffs[i], i))
    
    while num_splits>0 and not pq.empty():
        w, i = pq.get()
        sgm = bptsgm[i]
        if sgm.area() > 1:
            split = sgm.split(None, None)
            assert split is not None
            j = len(bptsgm)
            # print(f'split {i}->{j} with weight {-w:.5} area={sgm.area()}')
#             a0, a1 = split[0].area(), split[1].area()
#             assert sgm.area() == a0 + a1
            bptsgm[i] = split[0]
            bptsgm.append(split[1])
            c01 = coeffs[i]
            coeffs[i] = c01 / 2 #* a0 / (a0 + a1)
            coeffs.append(c01 / 2) #* a1 / (a0 + a1))
            pq.put((-coeffs[i], i))
            pq.put((-coeffs[j], j))
            # print(-w, " -> ", coeffs[i], a0,"  +  ", coeffs[j], a1)
            masking_mat = np.hstack((masking_mat, masking_mat[:, i:i+1]))
            num_splits -= 1

    return masking_mat

# generate the pixel-level heatmaps
def gen_heatmaps_from_partitions(image, bptsgm, regr_coeffs, preds):
    num_classes = len(preds[0])
    limebpt_coeffs = np.zeros((num_classes, image.shape[0], image.shape[1]), dtype=np.float32)
    for j, sgm in enumerate(bptsgm):
        sgm.add_inside_coalition(limebpt_coeffs, regr_coeffs[ :, j ])
    for j in range(num_classes):
        limebpt_coeffs[j] *= (preds[1][j] - preds[0][j]) / np.sum(limebpt_coeffs[j])
    return limebpt_coeffs

# generate the pixel-level variance maps
def gen_varmaps_from_partitions(image, bptsgm, variance_coeffs):
    num_classes = variance_coeffs.shape[0]
    variance_maps = np.zeros((num_classes, image.shape[0], image.shape[1]), dtype=np.float32)
    for j, sgm in enumerate(bptsgm):
        sgm.add_inside_coalition(variance_maps, variance_coeffs[ :, j ] * sgm.area())
    # for j in range(num_classes):
    #     limebpt_coeffs[j] *= (preds[1][j] - preds[0][j]) / np.sum(limebpt_coeffs[j])
    return variance_maps

# keep up the prediction matrix w.r.t. the masking matrix
def update_predictions(image, bptsgm, nu, preds, masking_mat, batch_size, pbar):
    for i in range(len(preds), len(masking_mat), batch_size):
        i2 = min(i+batch_size, len(masking_mat))
        m = np.zeros((i2-i, image.shape[0], image.shape[1]), dtype=bool)
        for j in range(i2-i):
            for z in np.where(masking_mat[i+j] == False)[0]:
                bptsgm[z].fill_mask(m[j], ascend_hier=False)
        for row in nu(torch.from_numpy(m).to(device=nu.masked_model.device)):
            preds.append(row)#[explained_classes] if explained_classes is not None else row)
        if pbar is not None: pbar.update(i2-i)

# determine the points where the evaluation budget stops to perform partition refinements
def define_refinement_plan(num_refinements, max_evals):
    refinement_steps = [15] + ([4] * num_refinements)
    # num_samples_per_refinements = np.linspace(0, max_evals, num_refinements+2, dtype=int)[1:]
    cs = np.cumsum(range(num_refinements+2))
    num_samples_per_refinements = ((cs / cs[-1]) * max_evals).astype(int)[1:]
    
    plan = list(zip(refinement_steps, num_samples_per_refinements)) 
    return plan

######################################################################################################

def explain_with_LimeBPT(nu, max_evals, batch_size, method='BPT', num_refinements=3, refine_using_vars=False):
    random_state = np.random.RandomState()#(seed=1234)
    bptree = shap_bpt.build_bpt_from_image(nu.masked_model.image_to_explain)
    if method=='BPT':
        bpt_root = shap_bpt.BPT_Segment(bptree, bptree.N-1, shap_bpt.BaseSegment())
    else:
        bpt_root = shap_bpt.AxisAlignedSegment(0, bptree.width, 0, bptree.height, shap_bpt.BaseSegment())
    bptsgm = [bpt_root]

    num_segments = len(bptsgm)
    split_coeffs = [1.0]
    masking_mat = np.array([[True], [False]])
    preds = []
    pbar = tqdm(range(max_evals))
    update_predictions(nu.masked_model.image_to_explain, bptsgm, nu, preds, masking_mat, batch_size, pbar)

    for num_new_refinements, next_target in define_refinement_plan(num_refinements, max_evals):
        print(f'At step {len(masking_mat)} adding {num_new_refinements} partitions from {len(bptsgm)} to {len(bptsgm) + num_new_refinements}.')
        # refine partitions & add columns to masking matrix
        masking_mat = refine_partitions(bptsgm, split_coeffs, masking_mat, num_new_refinements)
        # append new random samples as rows of the masking matrix
        num_new_samples = min(next_target - len(masking_mat), max_evals - len(masking_mat))
        new_samples = random_state.randint(0, 2, num_new_samples*len(bptsgm), dtype=bool)\
                        .reshape((num_new_samples, len(bptsgm)))
        masking_mat = np.concatenate((masking_mat, new_samples))
        # compute the new predictions and extend the preds[] matrix
        update_predictions(nu.masked_model.image_to_explain, bptsgm, nu, preds, masking_mat, batch_size, pbar)

        # Fit a Weighted Least Squares linear regression model
        sample_weights = np.ones(len(masking_mat))
        regr_coeffs, variance_coeffs, mean_variances = weighted_least_squares(masking_mat.astype(int), np.array(preds), sample_weights)
        if refine_using_vars:
            split_coeffs = list(mean_variances)
            # for i, sgm in enumerate(bptsgm):
            #     split_coeffs[i] /= sgm.area()
        else:
            split_coeffs = list(np.max(regr_coeffs, axis=0))

        # compute the feature importances for all the segments
        limebpt_coeffs = gen_heatmaps_from_partitions(nu.masked_model.image_to_explain, bptsgm, regr_coeffs, preds)
        limebpt_vmaps = gen_varmaps_from_partitions(nu.masked_model.image_to_explain, bptsgm, np.expand_dims(split_coeffs, axis=0))

        # v = np.quantile(np.abs(limebpt_coeffs[0]), 0.99)
        # v2 = np.quantile(np.abs(limebpt_vmaps[0]), 0.99)
        # fig, axes = plt.subplots(1,2, figsize=(6,3))
        # axes[0].imshow(limebpt_coeffs[0], vmin=-v, vmax=v, cmap=shap_bpt.shapley_values_colormap)
        # axes[1].imshow(limebpt_vmaps[0], cmap='Greens', vmax=v2)
        # for i, sgm in enumerate(bptsgm):
        #     x1,x2,y1,y2 = sgm.get_bounding_box()
        #     axes[1].text((x1+x2)/2, (y1+y2)/2, f'{i}', va='center', ha='center', color='red')
            
        # plt.suptitle(f'Step: {len(masking_mat)}, partitions: {len(bptsgm)}, refine_using_vars: {refine_using_vars}')
        # plt.tight_layout()
        # plt.show()

    pbar.close()
    return limebpt_coeffs, limebpt_vmaps

######################################################################################################











######################################################################################################
# GradCAM
# https://github.com/jacobgil/pytorch-grad-cam
######################################################################################################
from pytorch_grad_cam import GradCAM , HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

def explain_with_gradcam(nu):
    if nu.masked_model.model is None or nu.masked_model.cam_target_layers is None:
        return None
    cam = GradCAM(model=nu.masked_model.model, 
                  target_layers=nu.masked_model.cam_target_layers,
                  reshape_transform=nu.masked_model.cam_reshape_transform)
    class_heatmaps = []
    for i in range(nu.num_explained_classes):
        heatmap = cam(input_tensor=torch.unsqueeze(nu.masked_model.image_to_explain_tensor, dim=0), 
                      targets=[ClassifierOutputTarget(nu.output_indexes[i])])[0]
        # normalize
        heatmap = heatmap * nu.total_outcome()[i] / np.sum(heatmap)
        class_heatmaps.append(heatmap)
    return np.array(class_heatmaps)

######################################################################################################
# Integrated Decision Gradients
# https://arxiv.org/pdf/2305.20052.pdf
# https://github.com/chasewalker26/Integrated-Decision-Gradients/tree/main
######################################################################################################
# from saliencyMethods import IDG

# # get integrated decision gradients attribution
# def explain_with_idg(nu, use_abs=True):
#     if nu.masked_model.model is None or nu.masked_model.cam_target_layers is None:
#         return None
#     steps = 50
#     batch_size = 5 # default 25
#     baseline = 0
#     class_heatmaps = []
#     for i in range(nu.num_explained_classes):
#         heatmap = idg = IDG(torch.unsqueeze(nu.masked_model.image_to_explain_tensor, dim=0), 
#                             nu.masked_model.model, 
#                             steps, batch_size, baseline, 
#                             nu.masked_model.device, nu.output_indexes[i])
#         heatmap = idg.detach().cpu().numpy()
#         heatmap = np.mean(heatmap, axis=0) # reduce to one attribution per pixel
#         # normalize
#         if use_abs:
#             heatmap = np.abs(heatmap)
#         heatmap = heatmap * nu.total_outcome()[i] / np.sum(heatmap)
#         class_heatmaps.append(heatmap)
#     return np.array(class_heatmaps)

######################################################################################################










# ######################################################################################################
# # GradSHAP
# ######################################################################################################

# import shap as shap

# def explain_with_gradShap(nu, use_abs=True):
#     if nu.masked_model.model is None:
#             return None
#     e = shap.GradientExplainer(nu.masked_model.model, nu.masked_model.replacement_value_tensors)
#     expl = e.shap_values(torch.unsqueeze(nu.masked_model.image_to_explain_tensor, dim=0), 
#                          nsamples=20, #output_rank_order='custom',
#                          ranked_outputs=nu.num_explained_classes)
#     heatmaps = np.moveaxis(np.sum(expl[0][0], axis=0), 2, 0)
#     # print(heatmaps.shape)
#     for i in range(nu.num_explained_classes):
#         assert expl[1][0][i] == nu.output_indexes[i]
#         if use_abs:
#             heatmaps[i] = np.abs(heatmaps[i])
#         heatmaps[i] = heatmaps[i] * nu.total_outcome()[i] / np.sum(heatmaps[i])
#     return heatmaps

# ######################################################################################################
# # Partition Explainer of Shap
# ######################################################################################################

# def explain_with_shap_partition_explainer(nu, max_evals, batch_size):
#     def shap_predict(img):
#         y = nu.masked_model.predict(torch.Tensor(img).permute(0,3,1,2).to(nu.masked_model.device))
#         return y[:, nu.output_indexes]
    
#     masker = shap.maskers.Image(nu.masked_model.replacement_value_tensors[0].permute(1,2,0).detach().cpu().numpy(), 
#                                 nu.masked_model.image_to_explain.shape)
#     feature_names = [nu.masked_model.class_name(i) for i in nu.output_indexes]
#     partExpl = shap.Explainer(shap_predict, masker, algorithm="partition",
#                               feature_names=feature_names, output_names=feature_names)
#     shap_values_pe = partExpl(np.expand_dims(nu.masked_model.image_to_explain_tensor.permute(1,2,0).detach().cpu().numpy(), 0), 
#                               max_evals=max_evals, batch_size=batch_size)
#     shap_values = np.moveaxis(np.sum(shap_values_pe.values[0], axis=2), 2, 0)
#     return shap_values

# ######################################################################################################
