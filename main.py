import numpy as np
import matplotlib.pyplot as plt
import cv2
from metrics import false_pos, IoU, Dice, false_neg
from Canny_alg import video_process_canny
from Sobel import video_process_sobel

DATA_PATH = ''
video_names = ['test1/original/', 'test2/original/']
mask_names = ['test1/masked/', 'test2/masked/']



params1 = [(10, 150), 0.001] # для алгоритма canny

n = 2

params2 = [3, 1] # Kernel and scale

n = 2


def process_video(video, params):
    
    return video

def main():
    
    features = []
    
    for i in range(2):
        
        frames = []
        masks = []
        
        for j in range(1, 49):
            
            frame = cv2.imread(DATA_PATH+video_names[i]+str(j)+".png")
            mask = cv2.imread(DATA_PATH+mask_names[i]+str(j)+".png")
            
            frames.append(frame)
            masks.append(mask)
            
        gen_masks = video_process_canny(frames, params1, masks[0])

        fp = false_pos(masks, gen_masks)
        fn = false_neg(masks, gen_masks)
        iou = IoU(masks, gen_masks)
        dice = Dice(masks, gen_masks)
        
        features.append([fp, fn, iou, dice])
                
    features = np.array(features)
    features = np.concatenate((features[0], features[1]), axis = 1)
    
    for i in features:     
        print(np.mean(i), np.std(i))
    #print(features[0])
    #np.save("canny.npy", features)
    
    print(features.shape)
        
if __name__ == "__main__":
    
    main()