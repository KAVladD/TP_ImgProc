import numpy as np
import cv2
import matplotlib.pyplot as plt

def binar(mask):
    
    mask[mask != 0] = 1
    
    return mask

def binar1(mask):
    
    mask[mask == 1] = 0
    mask[mask != 0] = 1
    
    return mask

def binarization(true_mask, mask):
    
    true_mask = cv2.cvtColor(true_mask, cv2.COLOR_BGRA2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    
    true_mask = binar(true_mask)
    mask = binar1(mask)
    
    return true_mask, mask

def frame_true_pos(true_mask, mask):
    
    true_mask, mask = binarization(true_mask, mask)
    
    true_mask = np.array(true_mask)
    true = np.sum(true_mask) + 1 

    mask = np.array(mask)
    
    true_mask = true_mask[true_mask == mask]
    
    return len(true_mask[true_mask == 1]) / true

def frame_false_pos(true_mask, mask):
    
    true_mask, mask = binarization(true_mask, mask)
    
    true_mask = np.array(true_mask)    
    mask = np.array(mask)
    
    true = np.sum(true_mask) + 1
    
    mask = mask[mask != true_mask]
    
    return np.sum(mask == 1) / true

def frame_false_neg(true_mask, mask):
    
    true_mask, mask = binarization(true_mask, mask)
    
    true_mask = np.array(true_mask)
    mask = np.array(mask)
    
    true = np.sum(true_mask) + 1
    
    mask = mask[mask != true_mask]
    
    return (len(mask) - np.sum(mask == 1)) / true

def frame_IoU(true_mask, mask):
    
    true_mask, mask = binarization(true_mask, mask)
    
    true_mask = np.array(true_mask)    
    mask = np.array(mask)
    
    I = np.sum(true_mask[true_mask == mask])
    U = np.sum(true_mask) + np.sum(mask) - I + 1
    
    return I / U

def frame_Dice(true_mask, mask):
    
    true_mask, mask = binarization(true_mask, mask)
    
    true_mask = np.array(true_mask)    
    mask = np.array(mask)
    
    numerator = 2 * np.sum(true_mask[true_mask == mask])
    denominator = np.sum(true_mask) + np.sum(mask) + 1
    
    return numerator / denominator

def array_func(true_masks, masks, frame_func):
    
    n = max(len(true_masks), len(masks))
    
    val = []
    
    for i in range(n):
        
        val.append(frame_func(true_masks[i], masks[i]))
    
    return np.array(val)

def true_pos(true_masks, masks):
    
    return array_func(true_masks, masks, frame_true_pos)
                  
def false_pos(true_masks, masks):
    
    return array_func(true_masks, masks, frame_false_pos)

def false_neg(true_masks, masks):
    
    return array_func(true_masks, masks, frame_false_neg)

def IoU(true_masks, masks):
    
    return array_func(true_masks, masks, frame_IoU)

def Dice(true_masks, masks):
    
    return array_func(true_masks, masks, frame_Dice)