import numpy as np

def frame_true_pos(true_mask, mask):
    
    true_mask = np.array(true_mask)
    true = np.sum(true_mask)    

    mask = np.array(mask)
    
    true_mask = true_mask[true_mask == mask]
    
    return len(true_mask[true_mask == 1]) / true

def frame_false_pos(true_mask, mask):
    
    true_mask = np.array(true_mask)    
    mask = np.array(mask)
    
    true = np.sum(mask) 
    
    true_mask = true_mask[true_mask != mask]
    
    return len(true_mask[true_mask == 0]) / true

def true_pos(true_masks, masks):
    
    n = max(len(true_masks), len(masks))
    
    tp = []
    
    for i in range(n):
        
        tp.append(frame_true_pos(true_masks[i], masks[i]))
    
    return np.array(tp)
                  

def false_pos(true_masks, masks):
    
    n = max(len(true_masks), len(masks))
    
    fp = []
    
    for i in range(n):
        
        fp.append(frame_true_pos(true_masks[i], masks[i]))
    
    return np.array(fp)