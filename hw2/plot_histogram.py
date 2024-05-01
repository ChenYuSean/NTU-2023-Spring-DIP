import cv2
import numpy as np
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class Histogram:
    def __init__(self, image):
        self.hist, _ = np.histogram(image.flatten(), bins=256, range=(0,256))
        self.cdf = self.hist.cumsum()
        self.cdf_normalized = self.cdf / self.cdf.max()
    
    def plot_histogram(self, output_path):
        plt.figure()
        plt.hist(self.hist, 256, [0,256])
        plt.savefig(output_path)
    
    def plot_cdf(self, output_path):
        plt.figure()
        plt.plot(self.cdf_normalized, color='b')
        plt.savefig(output_path)
        
    def InverseCDF(self, alpha):
        for i in range(256):
            if self.cdf_normalized[i] >= alpha:
                return i
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    input_path = './OutputImages'
    output_path = './OutputImages'
    
    # image1 = cv2.imread(os.path.join(input_path, 'result1.png'), cv2.IMREAD_GRAYSCALE)
    # hist1 = Histogram(image1)
    # hist1.plot_cdf(os.path.join(output_path, 'cdf1.png'))
    # print(hist1.InverseCDF(0.9))
    
    # image2 = cv2.imread(os.path.join(input_path, 'phase2.png'), cv2.IMREAD_GRAYSCALE)
    # hist2 = Histogram(image2)
    # hist2.plot_cdf(os.path.join(output_path, 'cdf2.png'))
    # for p in np.arange(0.5, 1.0, 0.025):
    #     print(f"p:{p:.3f}, v:{hist2.InverseCDF(p)}")
    
    