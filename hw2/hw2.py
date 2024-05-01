import cv2
import numpy as np
import argparse
import os

# # Comment out when uploading to the judge system
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    return parser.parse_args()

def Sobel_edge_detection(img):
    res = np.zeros(img.shape)
    orientation_map = np.zeros(img.shape)   
    
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    row_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    col_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    for row,col in np.ndindex(img.shape):
        # start_row = row + translate_to_padded_img - half_size
        # end_row = row + translate_to_padded_img + half_size + 1
        # half_size = size // 2, translate_to_padded_img = half_size
        # start_row = row, end_row = row + size
        row_grad = 1/4 * np.sum(padded_img[row:row+3, col:col+3] * row_kernel)
        col_grad = 1/4 * np.sum(padded_img[row:row+3, col:col+3] * col_kernel)
        res[row, col] = np.sqrt(row_grad**2 + col_grad**2)
        orientation_map[row, col] = np.arctan2(col_grad, row_grad)
    
    return res, orientation_map

def thresholding(img, threshold):
    res = np.zeros(img.shape)
    
    for row,col in np.ndindex(img.shape):
        res[row, col] = 255 if img[row, col] > threshold else 0
    
    return res

def Canny_edge_detection(img, guassian_size, guassian_sigma, threshold_low, threshold_high, max_iteration, save_each_phase=False, output_path="./"):
    phase1 = Gaussian_filter(img, guassian_size, guassian_sigma)
    phase2, orientation = Sobel_edge_detection(phase1)
    phase3 = non_maximum_suppression(phase2, orientation)
    phase4 = hysteresis_thresholding(phase3, threshold_low, threshold_high)
    phase5 = edge_connecting(phase4, iteration=max_iteration)
    if save_each_phase:
        cv2.imwrite(os.path.join(output_path, 'phase1.png'), phase1)
        cv2.imwrite(os.path.join(output_path, 'phase2.png'), phase2)
        cv2.imwrite(os.path.join(output_path, 'phase3.png'), phase3)
        cv2.imwrite(os.path.join(output_path, 'phase4.png'), phase4)
    return phase5

def Gaussian_filter(image, size, sigma):
    res = np.zeros(image.shape, dtype=np.uint8)
    
    kernel = np.zeros((size, size))
    for x,y in np.ndindex(kernel.shape):
        kernel[x, y] = np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2))
    kernel = kernel / np.sum(kernel)
    
    padded_img = cv2.copyMakeBorder(image, size//2, size//2, size//2, size//2, cv2.BORDER_REPLICATE)
    
    for row,col in np.ndindex(image.shape):
        res[row, col] = np.sum(padded_img[row:row+size, col:col+size] * kernel)
    return res

def non_maximum_suppression(img, orientation):
    res = np.zeros(img.shape)
    
    # Orientation Change: 0, 45, 90, 135
    # Original Orientation: -pi ~ pi
    # Change to 0-180, 45-215, 90-270, 135-315 four direction
    # [0-22.5,0+22.5), [45-22.5,45+22.5), [90-22.5,90+22.5), [135-22.5,135+22.5)]
    for r,c in np.ndindex(orientation.shape):
        if orientation[r,c] < 0:
            orientation[r,c] += np.pi
        orientation[r,c] = orientation[r,c] * 180 / np.pi
        orientation[r,c] = orientation[r,c] % 180
        if orientation[r,c] < 22.5 or orientation[r,c] >= 157.5:
            orientation[r,c] = 0
        elif orientation[r,c] < 67.5:
            orientation[r,c] = 45
        elif orientation[r,c] < 112.5:
            orientation[r,c] = 90
        else:
            orientation[r,c] = 135
    
    # Non-maximum suppression
    for r,c in np.ndindex(img.shape):
        if orientation[r,c] == 0:
            if c == 0:
                res[r,c] = img[r,c] if img[r,c] > img[r,c+1] else 0
            elif c == res.shape[1]-1:
                res[r,c] = img[r,c] if img[r,c] > img[r,c-1] else 0
            else:
                res[r,c] = img[r,c] if img[r,c] > img[r,c-1] and img[r,c] > img[r,c+1] else 0
        elif orientation[r,c] == 45:
            if r == 0 or c == 0:
                res[r,c] = img[r,c] if img[r,c] > img[r+1,c+1] else 0
            elif r == res.shape[0]-1 or c == res.shape[1]-1:
                res[r,c] = img[r,c] if img[r,c] > img[r-1,c-1] else 0
            else:
                res[r,c] = img[r,c] if img[r,c] > img[r-1,c-1] and img[r,c] > img[r+1,c+1] else 0
        elif orientation[r,c] == 90:
            if r == 0:
                res[r,c] = img[r,c] if img[r,c] > img[r+1,c] else 0
            elif r == res.shape[0]-1:
                res[r,c] = img[r,c] if img[r,c] > img[r-1,c] else 0
            else:
                res[r,c] = img[r,c] if img[r,c] > img[r-1,c] and img[r,c] > img[r+1,c] else 0
        else:
            if r == 0 or c == res.shape[1]-1:
                res[r,c] = img[r,c] if img[r,c] > img[r+1,c-1] else 0
            elif r == res.shape[0]-1 or c == 0:
                res[r,c] = img[r,c] if img[r,c] > img[r-1,c+1] else 0
            else:
                res[r,c] = img[r,c] if img[r,c] > img[r-1,c+1] and img[r,c] > img[r+1,c-1] else 0
    return res

def hysteresis_thresholding(img, threshold_low, threshold_high):
    res = np.zeros(img.shape)
    
    for r,c in np.ndindex(img.shape):
        if img[r,c] > threshold_high:
            res[r,c] = 255
        elif img[r,c] > threshold_low:
            res[r,c] = 128
    return res

def edge_connecting(img, iteration = None):
    res = img.copy()
    iteration = img.shape[0]*img.shape[1] if iteration == None else iteration
    # Do the algorithm until the result is not changed or reach the maximum iteration
    for i in range(iteration):
        prev_res = res.copy()
        for r in range(1, res.shape[0]-1):
            for c in range(1, res.shape[1]-1):
                if res[r,c] == 128:
                    if res[r-1,c-1] == 255 or res[r-1,c] == 255 or res[r-1,c+1] == 255 or res[r,c-1] == 255 or res[r,c+1] == 255 or res[r+1,c-1] == 255 or res[r+1,c] == 255 or res[r+1,c+1] == 255:
                        res[r,c] = 255
        if np.array_equal(res, prev_res):
            break
    # Clear the rest candidate pixels
    for r in range(1, res.shape[0]-1):
        for c in range(1, res.shape[1]-1):
            if res[r,c] == 128:
                res[r,c] = 0
    return res

def Laplacian_of_Gaussian_edge_detection(img, size, sigma, threshold, save_each_phase=False, output_path="./"):
    phase1 = Gaussian_filter(img, size, sigma)
    phase2 = Laplacian(phase1)
    phase3 = zero_crossing(phase2, threshold)
    if save_each_phase:
        cv2.imwrite(os.path.join(output_path, 'phase1.png'), phase1)
        cv2.imwrite(os.path.join(output_path, 'phase2.png'), phase2+64)
        
        # comment out 
        # plt.figure()
        # plt.hist(phase2, np.arange(-2.0,2.0,0.1), [-2.0,2.0])
        # plt.savefig(os.path.join(output_path, 'hist.png'))
        
    return phase3

def Laplacian(img):
    res = np.zeros(img.shape)
    
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel = kernel / 8
    
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    for row,col in np.ndindex(img.shape):
        res[row, col] = np.sum(padded_img[row:row+3, col:col+3] * kernel)
    
    return res

def zero_crossing(img, threshold):
    res = np.zeros(img.shape)
    for r in range(1,img.shape[0]-1):
        for c in range(1,img.shape[1]-1):
            if np.absolute(img[r,c]) < threshold:
                if img[r-1,c] * img[r+1,c] < 0 or img[r,c-1] * img[r,c+1] < 0 or img[r-1,c-1] * img[r+1,c+1] < 0 or img[r-1,c+1] * img[r+1,c-1] < 0:
                    res[r,c] = 255
    return res

def edge_crispening(img, L, sigma, c):
    # c =[0.6~0.833]
    res = np.zeros(img.shape)
    low_pass_img = Gaussian_filter(img, L, sigma)    
    for row,col in np.ndindex(img.shape):
            res[row, col] = c/(2*c-1) * img[row, col] - (1-c)/(2*c-1) * low_pass_img[row, col]
    
    return res

class GeometricalModification:
    def __init__(self, img):
        self.img = img
        self.matrix = None
        
    @staticmethod
    def to_cartesian(p, q, P, Q):
        x = q - 1/2
        y = P + 1/2 - p
        return x,y
    @staticmethod
    def to_image(x, y, P, Q):
        p = P + 1/2 - y
        q = x + 1/2
        return p,q
    @staticmethod
    def bilinear_interpolation(img, p, q, padding_value = 255):
        if p < 0 or p >= img.shape[0]-1 or q < 0 or q >= img.shape[1]-1:
            return padding_value
        a, b = p - int(p), q - int(q)
        return img[int(p),int(q)] * (1-a)*(1-b) + img[int(p),int(q+1)] * (1-a)*b + img[int(p+1),int(q)] * a*(1-b) + img[int(p+1),int(q+1)] * a*b 
    

class Transform(GeometricalModification):
    def __init__(self, img):
        super().__init__(img)
        self.matrix = np.array([[1,0,0], [0,1,0], [0,0,1]])

    def translate(self, tx = 0, ty = 0):
        self.matrix = self.matrix @ np.array([[1,0,-tx], [0,1,-ty], [0,0,1]])

    def scale(self, sx = 1, sy = 1):
        self.matrix = self.matrix @ np.array([[1/sx,0,0], [0,1/sy,0], [0,0,1]])

    def rotate(self, angle = 0):
        rad = np.deg2rad(angle)
        self.matrix = self.matrix @ np.array([[np.cos(rad),np.sin(rad),0], [-np.sin(rad),np.cos(rad),0], [0,0,1]])
    
    def create_new_image(self):
        '''
        Create a new image after geometrical modification.
        '''  
        res = np.zeros(self.img.shape, dtype=np.uint8)
        for j,k in np.ndindex(res.shape):
            x_k, y_j = self.to_cartesian(j,k,self.img.shape[0],self.img.shape[1])
            u_p, v_q, _ = self.matrix @ np.array([[x_k],[y_j],[1]])
            p, q = self.to_image(u_p.item(), v_q.item(), self.img.shape[0], self.img.shape[1])
            res[j,k] = self.bilinear_interpolation(self.img, p, q)
        return res.astype(np.uint8)

class Warp(GeometricalModification):
    @staticmethod
    def distortion_effect(img, k1):
        res = np.zeros(img.shape, dtype=np.uint8)
        cx, cy = Warp.to_cartesian(img.shape[0]//2,img.shape[1]//2,img.shape[0],img.shape[1])
        for row,col in np.ndindex(res.shape):
            x, y = Warp.to_cartesian(row,col,img.shape[0],img.shape[1])
            dx = x - cx 
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            r_undistorted = r * (1 + k1*r**2)
            x_undistorted = r_undistorted * np.cos(theta) + cx
            y_undistorted = r_undistorted * np.sin(theta) + cy
            j,k = Warp.to_image(x_undistorted, y_undistorted, img.shape[0], img.shape[1])
            res[row,col] = Warp.bilinear_interpolation(img, j, k)
        return res
    @staticmethod
    def wavy_function(x, amplitude, frequency):
        return amplitude * np.sin(2 * np.pi * frequency * x)
    @staticmethod
    def wavy_effect(image, amplitude, frequency):
        res = np.copy(image)
        for row in range(image.shape[0]):
            displacement = Warp.wavy_function(row, amplitude, frequency)
            res[row] = np.roll(image[row], int(displacement), axis=0)
        
        return res

def main():
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Problem 1
    sample1 = cv2.imread(os.path.join(args.input_path, 'sample1.png'), cv2.IMREAD_GRAYSCALE)
    sample2 = cv2.imread(os.path.join(args.input_path, 'sample2.png'), cv2.IMREAD_GRAYSCALE)
    
    # (a) Sobel edge detection
    result1, _ = Sobel_edge_detection(sample1)
    cv2.imwrite(os.path.join(args.output_path, 'result1.png'), result1)
    result2 = thresholding(result1, 26) 
    cv2.imwrite(os.path.join(args.output_path, 'result2.png'), result2)
    
    # (b) Canney edge detection
    result3 = Canny_edge_detection(sample1, guassian_size=7, guassian_sigma=1.2, threshold_low=5, threshold_high=18, 
                                   max_iteration=20000, save_each_phase=False, output_path=args.output_path)
    cv2.imwrite(os.path.join(args.output_path, 'result3.png'), result3)
    
    # (c) Laplacian of Gaussian edge detection
    result4 = Laplacian_of_Gaussian_edge_detection(sample1, size=31, sigma=1.2, threshold=0.1, save_each_phase=False, output_path=args.output_path)
    cv2.imwrite(os.path.join(args.output_path, 'result4.png'), result4)
    
    # (d) Edge Crispening
    result5 = edge_crispening(sample2, L=31, sigma=1.2, c=0.6)
    cv2.imwrite(os.path.join(args.output_path, 'result5.png'), result5)
    
    # Problem 2
    sample3 = cv2.imread(os.path.join(args.input_path, 'sample3.png'), cv2.IMREAD_GRAYSCALE)
    sample5 = cv2.imread(os.path.join(args.input_path, 'sample5.png'), cv2.IMREAD_GRAYSCALE)
    
    # (a) 
    transform = Transform(sample3)
    transform.translate(-300, -300)
    transform.rotate(-10)
    transform.scale(1.7,2.0)
    transform.translate(335, 250)
    result8_1 = transform.create_new_image()
    # cv2.imwrite(os.path.join(args.output_path, 'result8_1.png'), result8_1)
    
    result8_2 = Warp.distortion_effect(result8_1, 0.0005)
    # cv2.imwrite(os.path.join(args.output_path, 'result8_2.png'), result8_2)
    
    transform = Transform(result8_2)
    transform.translate(-300, -300)
    transform.rotate(-40)
    transform.scale(2.0,2.0)
    transform.translate(360, 350)
    result8 = transform.create_new_image()
    cv2.imwrite(os.path.join(args.output_path, 'result8.png'), result8)
    
    # (b)
    sample5 = cv2.imread(os.path.join(args.input_path, 'sample5.png'), cv2.IMREAD_GRAYSCALE)
    result9 = Warp.wavy_effect(sample5, amplitude=20, frequency=0.007)
    cv2.imwrite(os.path.join(args.output_path, 'result9.png'), result9)
    
    
if __name__ == '__main__':
    main()
    
    