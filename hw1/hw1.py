import cv2
import numpy as np
import argparse
import os

# # Comment out when uploading to the judge system
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')

# def plot_histogram(image, output_path):
#     plt.figure()
#     plt.hist(image.flatten(), 256, [0,256])
#     plt.savefig(output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    return parser.parse_args()

def global_equalize(image):
    result_img = np.zeros(image.shape, dtype=np.uint8)
    count = np.zeros(256)
    for pixel in image.flatten():
        count[pixel] += 1
    cdf = np.cumsum(count).astype(np.float32)
    cdf = cdf / cdf[-1]
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            result_img[row, col] = 255 * cdf[image[row, col]]
            
    return result_img

def local_equalize(image, kernel_size):
    result_img = np.zeros(image.shape, dtype=np.uint8)
    half_kernel = kernel_size // 2
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            start_row = max(0, row - half_kernel)
            end_row = min(image.shape[0], row + half_kernel + 1)
            start_col = max(0, col - half_kernel)
            end_col = min(image.shape[1], col + half_kernel + 1)
            count = np.zeros(256)
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    count[image[i, j]] += 1
            cdf = np.cumsum(count).astype(np.float32)
            cdf = cdf / cdf[-1]
            result_img[row, col] = 255 * cdf[image[row, col]]
            
    return result_img

def Gaussian_3x3_filter(image, sigma):
    result_img = np.zeros(image.shape, dtype=np.uint8)
    kernel = np.array([[1,sigma,1],[sigma,sigma**2,sigma],[1,sigma,1]]) / (sigma+2)**2
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            val = 0
            start_row = max(0, row - 1)
            end_row = min(image.shape[0], row + 1)
            start_col = max(0, col - 1)
            end_col = min(image.shape[1], col + 1)
            for kr,i in enumerate(range(start_row, end_row)):
                for kc,j in  enumerate(range(start_col, end_col)):
                    val += image[i,j] * kernel[kr,kc]
            result_img[row,col] = val
    return result_img.astype(np.uint8)

def Median_filter(image, kernel_size):
    result_img = np.zeros(image.shape, dtype=np.uint8)
    half_kernel = kernel_size // 2
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            val = []
            start_row = max(0, row - half_kernel)
            end_row = min(image.shape[0], row + half_kernel + 1)
            start_col = max(0, col - half_kernel)
            end_col = min(image.shape[1], col + half_kernel + 1)
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    val.append(image[i,j])
            result_img[row,col] = np.median(val)
    return result_img.astype(np.uint8)

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    PIXEL_MAX = 255.0
    return 10 * np.log10(PIXEL_MAX**2 / mse)


def main():
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Problem 0
    image1 = cv2.imread(os.path.join(args.input_path, 'sample1.png'))

    ## Vertical flip
    result1 = np.zeros(image1.shape, dtype=np.uint8)
    for row in range(image1.shape[0]):
        for col in range(image1.shape[1]):
            result1[row, col] = image1[image1.shape[0]-1-row, col]
    cv2.imwrite(os.path.join(args.output_path,'result1.png'), result1)

    ## Gray Scale
    result2 = np.zeros(image1.shape, dtype=np.uint8)
    result2 = image1[:,:,0]*0.114 + image1[:,:,1]*0.587 + image1[:,:,2]*0.299
    cv2.imwrite(os.path.join(args.output_path,'result2.png'), result2)


    # Problem 1
    image2 = cv2.imread(os.path.join(args.input_path, 'sample2.png'), cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(os.path.join(args.input_path, 'sample3.png'), cv2.IMREAD_GRAYSCALE)

    ##  (a)
    result3 = np.zeros(image2.shape, dtype=np.uint8)
    result3 = image2 // 3
    result3 = result3.astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_path,'result3.png'), result3)

    ## (b)
    result4 = np.zeros(result3.shape, dtype=np.uint16)
    result4 = result3 * 3
    result4 = np.clip(result4, 0, 255)
    result4 = result4.astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_path,'result4.png'), result4)
    
    ## (c) Comment out this section
    # plot_histogram(image2, os.path.join(args.output_path,'hist1.png'))
    # plot_histogram(result3, os.path.join(args.output_path,'hist2.png'))
    # plot_histogram(result4, os.path.join(args.output_path,'hist3.png'))
    
    # ## (d)
    result5 = global_equalize(image2)
    result6 = global_equalize(result3)
    result7 = global_equalize(result4)
    cv2.imwrite(os.path.join(args.output_path,'result5.png'), result5)
    cv2.imwrite(os.path.join(args.output_path,'result6.png'), result6)
    cv2.imwrite(os.path.join(args.output_path,'result7.png'), result7)
    # comment out
    # plot_histogram(result5, os.path.join(args.output_path,'hist4.png'))
    # plot_histogram(result6, os.path.join(args.output_path,'hist5.png'))
    # plot_histogram(result7, os.path.join(args.output_path,'hist6.png'))
    
    ## (e)
    result8 = np.zeros(image2.shape, dtype=np.uint8)
    result8 = local_equalize(image2, 35)
    cv2.imwrite(os.path.join(args.output_path,'result8.png'), result8)
    # plot_histogram(result8, os.path.join(args.output_path,'hist7.png'))
    
    ## (f)
    result9 = np.zeros(image3.shape, dtype=np.uint8)
    result9 = image3 * 2
    result9 = np.clip(result9, 0, 255)
    result9 = global_equalize(result9)
    cv2.imwrite(os.path.join(args.output_path,'result9.png'), result9)
    # plot_histogram(result9, os.path.join(args.output_path,'hist8.png'))
    
    # Problem 2
    
    image4 = cv2.imread(os.path.join(args.input_path, 'sample4.png'), cv2.IMREAD_GRAYSCALE)
    ## (a)
    result10 = Gaussian_3x3_filter(image4, 2)
    cv2.imwrite(os.path.join(args.output_path,'result10.png'), result10)
    
    result11 = Median_filter(image4, 5)
    cv2.imwrite(os.path.join(args.output_path,'result11.png'), result11)
    
    ## (b)
    print(PSNR(image4, result10))
    print(PSNR(image4, result11))
    
if __name__ == '__main__':
    main()