
import torch
import numpy as np

class GenerateHeatmap():
    def __init__(self, num_parts, sigma):
        self.num_parts = num_parts
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints, output_res_size):
        [output_res_h, output_res_w] = output_res_size
        hms = np.zeros(shape = (self.num_parts, output_res_h, output_res_w), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=output_res_w or y>=output_res_h:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], output_res_w) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], output_res_h) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], output_res_w)
                    aa,bb = max(0, ul[1]), min(br[1], output_res_h)
                    
                    # print(idx, hms.shape, self.g.shape, aa,bb,cc,dd, a,b,c,d)
                    
                    cropped_hm = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
                    hms[idx, aa:bb,cc:dd] = cropped_hm
        return hms