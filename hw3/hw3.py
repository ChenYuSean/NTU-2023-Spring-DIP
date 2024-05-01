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
    parser.add_argument('--input_path',default="./SampleImages",type=str)
    parser.add_argument('--output_path',default="./OutputImages",type=str)
    return parser.parse_args()

class Morphology:
    BOUNDARY_KERNEL = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.uint8)
    HOLE_FILLING_KERNEL = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    COMPONENT_LABELING_KERNEL = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.uint8)
    @staticmethod
    def dilation(img, kernel):
        res_img = np.zeros(shape=img.shape,dtype=np.uint8)
        shift_indexs = np.argwhere(kernel == 1)
        shift_imgs = []
        center = (kernel.shape[0]//2,kernel.shape[1]//2)
        for shift_index in shift_indexs:
            top_bottom = shift_index[0]-center[0]
            left_right = shift_index[1]-center[1]
            if(top_bottom > 0):
                shift_img = cv2.copyMakeBorder(img,top_bottom,0,0,0,cv2.BORDER_CONSTANT,value=0) #shift down
                shift_img = shift_img[0:img.shape[0],0:img.shape[1]]
            else:
                shift_img = cv2.copyMakeBorder(img,0,-top_bottom,0,0,cv2.BORDER_CONSTANT,value=0) # shift up
                shift_img = shift_img[-top_bottom:img.shape[0]-top_bottom,0:img.shape[1]]
            if(left_right > 0):
                shift_img = cv2.copyMakeBorder(shift_img,0,0,left_right,0,cv2.BORDER_CONSTANT,value=0) # shift right
                shift_img = shift_img[0:img.shape[0],0:img.shape[1]]
            else:
                shift_img = cv2.copyMakeBorder(shift_img,0,0,0,-left_right,cv2.BORDER_CONSTANT,value=0) # shift left
                shift_img = shift_img[0:img.shape[0],-left_right:img.shape[1]-left_right]
            shift_imgs.append(shift_img)
        shift_imgs = np.stack(shift_imgs,axis=0)
        res_img = shift_imgs.max(axis=0)
        return res_img
    @staticmethod
    def erosion(img, kernel):
        res_img = np.zeros(shape=img.shape,dtype=np.uint8)
        shift_indexs = np.argwhere(kernel == 1)
        shift_imgs = []
        center = (kernel.shape[0]//2,kernel.shape[1]//2)
        for shift_index in shift_indexs:
            top_bottom = shift_index[0]-center[0]
            left_right = shift_index[1]-center[1]
            if(top_bottom > 0):
                shift_img = cv2.copyMakeBorder(img,top_bottom,0,0,0,cv2.BORDER_CONSTANT,value=0) #shift down
                shift_img = shift_img[0:img.shape[0],0:img.shape[1]]
            else:
                shift_img = cv2.copyMakeBorder(img,0,-top_bottom,0,0,cv2.BORDER_CONSTANT,value=0) # shift up
                shift_img = shift_img[-top_bottom:img.shape[0]-top_bottom,0:img.shape[1]]
            if(left_right > 0):
                shift_img = cv2.copyMakeBorder(shift_img,0,0,left_right,0,cv2.BORDER_CONSTANT,value=0) # shift right
                shift_img = shift_img[0:img.shape[0],0:img.shape[1]]
            else:
                shift_img = cv2.copyMakeBorder(shift_img,0,0,0,-left_right,cv2.BORDER_CONSTANT,value=0) # shift left
                shift_img = shift_img[0:img.shape[0],-left_right:img.shape[1]-left_right]
            shift_imgs.append(shift_img)
        shift_imgs = np.stack(shift_imgs,axis=0)
        # print(shift_imgs)
        res_img = shift_imgs.min(axis=0)
        return res_img
    @staticmethod
    def opening(img, kernel):
        return Morphology.dilation(Morphology.erosion(img,kernel),kernel)
    @staticmethod
    def closing(img, kernel):
        return Morphology.erosion(Morphology.dilation(img,kernel),kernel)
    @classmethod
    def boundary_extraction(cls, img, kernel = BOUNDARY_KERNEL):
        return img - Morphology.erosion(img,kernel)
    @classmethod
    def hole_filling(cls, img, start_positions, kernel = HOLE_FILLING_KERNEL, max_iteration = 10000):
        complement = 255 - img
        res = img.copy()
        for start_position in start_positions:
            for i in range(max_iteration):
                if i == 0:
                    G_0 = np.zeros(shape=img.shape,dtype=np.uint8)
                    G_0[start_position[0],start_position[1]] = 255
                    G_1 = G_0
                G_0 = G_1
                G_1 = Morphology.dilation(G_0,kernel)
                G_1 = np.minimum(G_1,complement)
                if(G_1 == G_0).all():
                    break       
            res = np.maximum(res,G_1)
        return res
    @classmethod
    def connected_component_labeling(cls, img, kernel = COMPONENT_LABELING_KERNEL, max_iteration = 10000):
        label_count = 0
        label_map = img.copy() * (-1)
        start_positions = np.argwhere(img == 255)
        for start_position in start_positions:
            if label_map[start_position[0],start_position[1]] >= 0:
                continue
            label_count += 1
            for i in range(max_iteration):
                if i == 0:
                    G_0 = np.zeros(shape=img.shape,dtype=np.uint8)
                    G_0[start_position[0],start_position[1]] = 255
                    G_1 = G_0
                G_0 = G_1
                G_1 = Morphology.dilation(G_0,kernel)
                G_1 = np.minimum(G_1,img)
                if(G_1 == G_0).all():
                    break
            label_map[G_1 == 255] = label_count
            
        return label_count,label_map
        
def fill(img):
    border = Morphology.boundary_extraction(img)
    fill_background = Morphology.hole_filling(border, [(0, 0)])
    reversed = 255 - fill_background
    res = np.maximum(reversed,border)
    
    return res

def Median_filter(image, kernel_size):
    result_img = np.zeros(image.shape, dtype=np.uint8)
    half_kernel = kernel_size // 2
    for row, col in np.ndindex(image.shape):
        start_row = max(0, row - half_kernel)
        end_row = min(image.shape[0], row + half_kernel + 1)
        start_col = max(0, col - half_kernel)
        end_col = min(image.shape[1], col + half_kernel + 1)
        result_img[row,col] = np.median(image[start_row:end_row, start_col:end_col])
    return result_img.astype(np.uint8)

def Laws_method(image, window_size=13, path = None , save = False):
    # Laws' method
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    # 1D filter
    L3 = (1/6) * np.array([1, 2, 1])
    E3 = (1/2) * np.array([-1, 0, 1])
    S3 = (1/2) * np.array([1, -2, 1])
    
    # 2D filter
    filters = [L3, E3, S3]
    filters_2d = []
    for f1 in filters:
        for f2 in filters:
            filters_2d.append(np.outer(f1, f2))
    
    # Convolution
    microstructures = []
    for f in filters_2d:
        result_img = np.zeros(image.shape)
        for row,col in np.ndindex(image.shape):
            result_img[row, col] = np.sum(padded_image[row:row+3, col:col+3] * f)
        microstructures.append(result_img)
    
    # Energy
    features = []
    for micro_image in microstructures:
        d = window_size // 2
        padded_micro = cv2.copyMakeBorder(micro_image, d, d, d, d, cv2.BORDER_REFLECT)
        feature = np.zeros(image.shape)
        for row, col in np.ndindex(image.shape):
            feature[row, col] = np.sum(padded_micro[row:row+window_size, col:col+window_size] ** 2)
        features.append(feature)
        
    if path is not None and save:
        for i, saved_img in enumerate(microstructures):
            cv2.imwrite(os.path.join(path, f'middle_{i}.png'), saved_img.astype(np.uint8))
        for i, feature in enumerate(features):
            saved_img = feature / np.max(feature) * 255
            cv2.imwrite(os.path.join(path, f'feature_{i}.png'), saved_img.astype(np.uint8))
    
    return features, microstructures

def k_means(image, features: np.ndarray, k, max_iter = 100):
    # features shape: (num_features, height, width)
    num_features = features.shape[0]
    res = np.zeros((image.shape[0],image.shape[1],3), dtype=np.uint8) # color image
    
    # Initialize centroids
    c_row = np.random.randint(0, features.shape[1], k)
    c_col = np.random.randint(0, features.shape[2], k)
    centroid_features = [features[:, r, c] for r, c in zip(c_row, c_col)] # (k,9)
        
    # K-means
    labels = np.zeros(image.shape, dtype=np.uint8) # label value = 0,1,2,...,k-1
    for _ in range(max_iter):
        # Assign each pixel to the nearest centroid
        for row, col in np.ndindex(image.shape):
            pixel = features[:, row, col]
            dists = [np.linalg.norm(pixel - cf) for cf in centroid_features]
            labels[row, col] = np.argmin(dists)
            
        # Update centroids
        new_centroids = []
        for cluster_idx in range(k):
            new_centroid = np.zeros(num_features)
            pixel_idx = np.argwhere(labels == cluster_idx)
            for r,c in pixel_idx:
                new_centroid += features[:, r, c]
            new_centroid /= len(pixel_idx)
            new_centroids.append(new_centroid)

        # Check convergence
        if np.allclose(centroid_features, new_centroids):
            break
        centroid_features = new_centroids
    
    # return texture classiflication result
    for i in range(k):
        res[labels == i] = np.random.randint(0, 256, 3)
    
    return res, labels

def add_texture(label_map, textures):
    res = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for row, col in np.ndindex(label_map.shape):
        texture = textures[label_map[row,col]]
        res[row, col] = texture[row % texture.shape[0], col % texture.shape[1]]
    return res

def main():
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    np.random.seed(0)
    
    # Problem 1
    sample1 = cv2.imread(os.path.join(args.input_path, 'sample1.png'), cv2.IMREAD_GRAYSCALE)
    # (a)
    result1 = Morphology.boundary_extraction(sample1)
    cv2.imwrite(os.path.join(args.output_path, 'result1.png'), result1)
    # (b)
    result2 = fill(sample1)
    cv2.imwrite(os.path.join(args.output_path, 'result2.png'), result2)
    # (c)
    result3 = Morphology.opening(sample1, np.array([[0,1,0],[1,1,1],[0,1,0]]))
    cv2.imwrite(os.path.join(args.output_path, 'result3.png'), result3)
    # cv2.imwrite(os.path.join(args.output_path, 'median_3.png'), Median_filter(sample1, 3))
    # cv2.imwrite(os.path.join(args.output_path, 'median_5.png'), Median_filter(sample1, 5))
    # (d)
    label_count, label_map = Morphology.connected_component_labeling(sample1)
    print(label_count)
    label_img = cv2.imread(os.path.join(args.input_path, 'sample1.png'), cv2.IMREAD_COLOR)
    for i in range(1, label_count+1):
        label_img[label_map == i] = np.random.randint(0, 256, 3)
    cv2.imwrite(os.path.join(args.output_path, 'label_img.png'), label_img)
    
    # Problem 2
    sample2 = cv2.imread(os.path.join(args.input_path, 'sample2.png'), cv2.IMREAD_GRAYSCALE)
    # (a)
    features, _ = Laws_method(sample2, path = args.output_path, save = False)
    # (b)
    result4, label_map = k_means(sample2, np.array(features), k = 5, max_iter=50)
    cv2.imwrite(os.path.join(args.output_path, 'result4.png'), result4)
    # (c)
    textures = []
    textures.append(cv2.imread(os.path.join(args.input_path, '64x', 'WoolSnow.png'), cv2.IMREAD_COLOR))
    textures.append(cv2.imread(os.path.join(args.input_path, '64x', 'Grass.png'), cv2.IMREAD_COLOR))
    textures.append(cv2.imread(os.path.join(args.input_path, '64x', 'LeavesTS.png'), cv2.IMREAD_COLOR))
    textures.append(cv2.imread(os.path.join(args.input_path, '64x', 'Ice.png'), cv2.IMREAD_COLOR))
    textures.append(cv2.imread(os.path.join(args.input_path, '64x', 'Glass.png'), cv2.IMREAD_COLOR))
    result5 = add_texture(label_map, textures)
    cv2.imwrite(os.path.join(args.output_path, 'result5.png'), result5)

    
if __name__ == '__main__':
    main()
    