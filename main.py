import numpy as np
import matplotlib.pyplot as plt
import cv2
from metrics import true_pos, false_pos

DATA_PATH = ''
video_names = ['Test Video 1.mkv', 'Test video 2.avi']
mask_names = ['', '']

#read true masks
masks = [[0], [0]]

params = [0]

n = 2

def process_video(video, params):
    
    return video

def main():
    
    features = []
    
    for i in range(1, 2):
        
        frames = []
        
        vid = cv2.VideoCapture(DATA_PATH + video_names[i])
        
        
        ret, frame = vid.read()
        while ret:
                    
        
            frames.append(frame)
            
            ret, frame = vid.read()

        for i in params:
            
            #use your own function
            gen_masks = process_video(frames, 0)
            
            tp = true_pos(masks, gen_masks)
            fp = false_pos(masks, gen_masks)
            
            plt.plot(tp, fp)
            
            m1, m2 = np.mean(tp), np.mean(fp)
            var1, var2 = np.var(tp), np.var(fp)
            
            features.append([m1, var1, m2, var2])
            
        features = np.array(features)
        
        np.save("method_name.npy", features)
        
if __name__ == "__main__":
    
    main()