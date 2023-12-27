import numpy as np
import matplotlib.pyplot as plt
import cv2
from metrics import false_pos, IoU, Dice, false_neg
from Canny_alg import video_process_canny

DATA_PATH = ''
video_names = ['test1/original/', 'test2/original/']
mask_names = ['test1/masked/', 'test2/masked/']

#read true masks
masks = [[0], [0]]

params = [(10, 150), 0.001] # для алгоритма canny

n = 2


def process_video(video, params):
    
    return video

def main():
    
    features = []
    
    for i in range(2):
        
        frames = []
        masks = []
        
        for j in range(1, 11):
            
            frame = cv2.imread(DATA_PATH+video_names[i]+str(j)+".png")
            mask = cv2.imread(DATA_PATH+mask_names[i]+str(j)+".png")
            
            frames.append(frame)
            masks.append(mask)
            
        gen_masks = video_process_canny(frames, params, masks[0])

        fp = false_pos(masks, gen_masks)
        print(fp)
        fn = false_neg(masks, gen_masks)
        print(fp)
        iou = IoU(masks, gen_masks)
        print(iou)
        dice = Dice(masks, gen_masks)
        print(dice)
        
        features.append([fp, fn])
                
    features = np.array(features)
    #print(features[0])
    #np.save("canny.npy", features)
        
if __name__ == "__main__":
    
    main()