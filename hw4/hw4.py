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
    SCAN_TYPE_ROW = 'row'
    SCAN_TYPE_COL = 'col'
    @staticmethod
    def thresholding(img, threshold = 128):
        ret = img.copy()
        ret[ret >= threshold] = 255
        ret[ret < threshold] = 0
        return ret
    @staticmethod
    def complement(img):
        return 255 - img
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
    def connected_component_labeling(cls, img, kernel = COMPONENT_LABELING_KERNEL, max_iteration = 10000, discard_size = 0,
                                     scan_type = None, save_image = False, path = './', filename = 'label_img.png'):
        label_count = 0
        label_map = img.copy() * (-1)
        start_positions = []
        if scan_type == cls.SCAN_TYPE_ROW or scan_type == None:
            for r in range(img.shape[0]):
                for c in range(img.shape[1]):
                    if img[r,c] == 255:
                        start_positions.append((r,c))
        elif scan_type == cls.SCAN_TYPE_COL:
            for c in range(img.shape[1]):
                for r in range(img.shape[0]):
                    if img[r,c] == 255:
                        start_positions.append((r,c))
        for start_position in start_positions:
            if label_map[start_position[0],start_position[1]] >= 0:
                continue
            label_count += 1
            # print(label_count)
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

            if G_1[G_1 == 255].shape[0] < discard_size:
                label_count -= 1
                label_map[G_1 == 255] = 0
            else:
                label_map[G_1 == 255] = label_count
            
        if save_image:
            label_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for label in range(1, label_count+1):
                label_img[label_map == label] = np.random.randint(0, 256, 3)
            print(f'total label:{label_count}')
            cv2.imwrite(os.path.join(path, filename), label_img)
        
        return label_count,label_map

class Dithering:
    def __init__(self, img, matrix, size = 2):
        self.img = img
        self.matrix = self.upsample(matrix, size)
        self.threshold = 255 * (self.matrix + 0.5) / (self.matrix.shape[0] * self.matrix.shape[1])
        
    def upsample(self, matrix: np.array, size):
        dither = matrix.copy()
        while dither.shape[0] < size:
            old_size = dither.shape[0]
            new_size = old_size * 2
            new_dither = np.zeros((new_size, new_size))
            for i, j in np.ndindex(dither.shape):
                new_dither[i, j] = 4 * dither[i, j] + matrix[0,0]
                new_dither[i, old_size + j] =  4 * dither[i, j] + matrix[0,1]
                new_dither[old_size + i, j] = 4 * dither[i, j] + matrix[1,0]
                new_dither[old_size + i, old_size + j] = 4 * dither[i, j] + matrix[1,1]
            dither = new_dither
        
        return dither
    def dither(self):
        ret = self.img.copy()
        for r,c in np.ndindex(ret.shape):
            ret[r,c] = 255 if ret[r,c] > self.threshold[r % self.threshold.shape[0], c % self.threshold.shape[1]] else 0
        return ret
    
class ErrorDiffusion:
    def __init__(self, img, filter, threshold = 0.5, path = './'):
        self.img = img
        self.filter = filter.astype(np.float64) / np.sum(filter)
        self.flip_filter = np.flip(self.filter, axis = 1)
        self.threshold = threshold if threshold <= 1.0 else threshold / 255 
        self.path = path
        
    def error_diffusion(self):
        center = (self.filter.shape[0] // 2, self.filter.shape[1] // 2)
        H, W = self.img.shape
        ret = np.zeros((H,W) , dtype=np.float64)
        normalized = self.img.copy().astype(np.float64) / 255
        for r in range(center[0], H-1-center[0]):
            for c in range(center[1], W-1-center[1]):
                # serpentine scanning
                if r % 2 == 0:
                    filter = self.filter
                else:
                    c = ret.shape[1] - c - 1
                    filter = self.flip_filter
                # thresholding
                ret[r,c] = 1.0 if normalized[r,c] >= self.threshold else 0.0
                # error diffusion
                error = normalized[r,c] - ret[r,c]
                row_slice = slice(r - center[0], r + center[0] + 1)
                col_slice = slice(c - center[1], c + center[1] + 1)
                patch = normalized[row_slice, col_slice]
                patch = patch + error * np.ones(filter.shape) * filter
                patch = np.clip(patch, 0, 1)
                normalized[row_slice, col_slice] = patch
        return ret * 255

class ShapeProperties:
    Q = {}
    Q[0] = [np.array([[0,0],[0,0]])]
    Q[1] = [np.array([[0,0],[0,255]]), np.array([[0,0],[255,0]]), np.array([[0,255],[0,0]]), np.array([[255,0],[0,0]])]
    Q[2] = [np.array([[255,255],[0,0]]), np.array([[255,0],[255,0]]), np.array([[0,255],[0,255]]), np.array([[0,0],[255,255]])]
    Q[3] = [np.array([[255,255],[255,0]]), np.array([[255,255],[0,255]]), np.array([[0,255],[255,255]]), np.array([[255,0],[255,255]])]
    Q[4] = [np.array([[255,255],[255,255]])]
    Q['D'] = [np.array([[255,0],[0,255]]), np.array([[0,255],[255,0]])]
    def __init__(self,image):
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.bb_area = self.height * self.width
        self.area = np.sum(image == 255) / self.bb_area
        self.moment = self.calculate_moment(3,3)
        self.invariant = self.calculate_invariant()
        self.bit_quad_count = self.calculate_bit_quad()
    
    def print_properties(self):
        print(f'Area: {self.area}')
        print(f'Invariant: {self.invariant}')
    
    def calculate_moment(self,max_p,max_q):
        moment = np.zeros((max_p+1,max_q+1))
        for p in range(max_p+1):
            for q in range(max_q+1):
                for r,c in np.ndindex(self.image.shape):
                    moment[p,q] += (r/self.height)**p * (c/self.width)**q * self.image[r,c]
        return moment
    
    def calculate_invariant(self):
        M = self.moment
        invariant = np.zeros(7)
        invariant[0] = M[2,0] + M[0,2]
        invariant[1] = (M[2,0] - M[0,2])**2 + 4*M[1,1]**2
        invariant[2] = (M[3,0] - 3*M[1,2])**2 + (3*M[2,1] - M[0,3])**2
        invariant[3] = (M[3,0] + M[1,2])**2 + (M[2,1] + M[0,3])**2
        invariant[4] = (M[3,0] - 3*M[1,2]) * (M[3,0] + M[1,2]) * ((M[3,0] + M[1,2])**2 - 3*(M[2,1] + M[0,3])**2) + (3*M[2,1] + M[0,3])*(M[2,1] + M[0,3])*(3*(M[3,0] + M[1,2])**2 - (M[2,1] + M[0,3])**2) - (M[3,0] + M[1,2])*(M[2,1] + M[0,3])*((M[3,0] + M[1,2])**2 - (M[2,1] + M[0,3])**2)
        invariant[5] = (M[2,0] - M[0,2]) * ((M[3,0] + M[1,2])**2 - (M[2,1] + M[0,3])**2) + 4*(M[1,1]**2)*(M[3,0] + M[1,2])*(M[2,1] + M[0,3])
        invariant[6] = (3*M[2,1] - M[0,3])*(M[3,0] + M[1,2])*((M[3,0] + M[1,2])**2 - 3*(M[2,1] + M[0,3])**2) + (3*M[1,2] - M[3,0]) * (M[2,1] + M[0,3]) *(3*(M[3,0] + M[1,2])**2 - (M[2,1] + M[0,3])**2)
        return invariant

    def calculate_bit_quad(self):
        bit_quad_count = [0,0,0,0,0,0] # q0 q1 q2 q3 q4 qd
        for r in range(0,self.height-1):
            for c in range(0,self.width-1):
                quad = self.image[r:r+2,c:c+2]
                for key in self.Q:
                    for q in self.Q[key]:
                        if (quad == q).all():
                            if key == 'D':
                                bit_quad_count[5] += 1
                            else:
                                bit_quad_count[int(key)] += 1
        return bit_quad_count

class ShapeAnalyzer:
    def __init__(self, training_set):
        self.training_set = training_set
        self.property_set = self.calculate_properties()
        
    def calculate_properties(self):
        properties = {}
        for key in self.training_set:
            properties[key] = ShapeProperties(self.training_set[key])
        return properties
    
    def print_properties_set(self):
        for key in self.property_set:
            print(key)
            self.property_set[key].print_properties()
            print("--------------------------------------")
    
    def shape_analyze(self, image, discard_size = 0 , path = './', filename = 'label.png'):
        label_count, label_map = Morphology.connected_component_labeling(image, discard_size=discard_size, scan_type=Morphology.SCAN_TYPE_COL,
                                                                        save_image=True, path=path, filename=filename)
        for label in range(1,label_count+1):
            component = np.argwhere(label_map == label)
            bbox = [np.min(component[:,0]), np.min(component[:,1]), np.max(component[:,0]), np.max(component[:,1])]
            sub_image = image[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1]
            predict, distance = self.match_properties(sub_image)
            print(((bbox[0]+bbox[2]+1)/2,(bbox[1]+bbox[3]+1)),predict, distance)
            
    def match_properties(self, image):
        properties = ShapeProperties(image)
        properties.print_properties()
        min_distance = float('inf')
        min_key = None
        # for key in self.property_set:
        #     distance = np.linalg.norm(self.property_set[key].invariant - properties.invariant)
        #     if distance < min_distance:
        #         min_distance = distance
        #         min_key = key
        return min_key, min_distance
        
def split_training(image, path = './'):
    samples = {}
    truth = ['C', 'G', 'A', 'B', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'O',
             'Q', 'S', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
             '0', '1', '2', '3', '6', '8', '9', '4', '5', '7']
    processed = Morphology.thresholding(image, 128)
    processed = Morphology.complement(processed)
    label_count, label_map = Morphology.connected_component_labeling(processed, save_image=True, path=path)
    for label in range(1,label_count+1):
        component = np.argwhere(label_map == label)
        bbox = [np.min(component[:,0]), np.min(component[:,1]), np.max(component[:,0]), np.max(component[:,1])]
        sub_image = processed[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1]
        samples[truth[label-1]] = sub_image
    
    path = os.path.join(path, 'SplitTraingSet')
    os.makedirs(path, exist_ok=True)
    # padding "I" side ways
    samples['I'] = cv2.copyMakeBorder(samples['I'], 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=0)
    for key in samples:
        cv2.imwrite(os.path.join(path, f'{key}.png'), samples[key])
    return samples

def outer_hole_filling(img):
    start_positions = []
    for r in range(img.shape[0]):
        if img[r,0] == 0:
            start_positions.append((r,0))
        if img[r,img.shape[1]-1] == 0:
            start_positions.append((r,img.shape[1]-1))
    for c in range(img.shape[1]):
        if img[0,c] == 0:
            start_positions.append((0,c))
        if img[img.shape[0]-1,c] == 0:
            start_positions.append((img.shape[0]-1,c))
    return Morphology.hole_filling(img, start_positions)

def main():
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    np.random.seed(0)
    
    # # Problem 1
    # sample1 = cv2.imread(os.path.join(args.input_path, 'sample1.png'), cv2.IMREAD_GRAYSCALE)
    # # (a)
    # result1 = Dithering(sample1, np.array([[0, 2], [3, 1]])).dither()
    # cv2.imwrite(os.path.join(args.output_path, 'result1.png'), result1)
    # # (b)
    # result2 = Dithering(sample1, np.array([[0, 2], [3, 1]]), 256).dither()
    # cv2.imwrite(os.path.join(args.output_path, 'result2.png'), result2)
    # # (c)
    # result3 = ErrorDiffusion(sample1, np.array([[0, 0, 0], [0, 0, 7], [3, 5, 1]]), path=args.output_path).error_diffusion()
    # cv2.imwrite(os.path.join(args.output_path, 'result3.png'), result3)
    # result4 = ErrorDiffusion(sample1, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]])).error_diffusion()
    # cv2.imwrite(os.path.join(args.output_path, 'result4.png'), result4)
    
    # Problem 2
    raw_training_set = cv2.imread(os.path.join(args.input_path, 'TrainingSet.png'), cv2.IMREAD_GRAYSCALE)
    
    training_set = {}
    folder_path = os.path.join(args.input_path, 'SplitTraingSet')
    if not os.path.isdir(folder_path):
        training_set = split_training(raw_training_set, args.output_path)
    else:
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
                training_set[filename.split('.')[0]] = img
                
    analyzer = ShapeAnalyzer(training_set)
    
    analyzer.print_properties_set()
    
    # preprocess sample
    # sample2 = cv2.imread(os.path.join(args.input_path, 'sample2.png'), cv2.IMREAD_GRAYSCALE)
    # sample2 = Morphology.complement(Morphology.thresholding(sample2, 128))
    # sample2 = Morphology.complement(Morphology.opening(sample2, Morphology.BOUNDARY_KERNEL))
    # sample2 = Morphology.complement(outer_hole_filling(sample2))
    # analyzer.shape_analyze(sample2, discard_size=70, path=args.output_path, filename='sample2_label.png')
    
    # sample3 = cv2.imread(os.path.join(args.input_path, 'sample3.png'), cv2.IMREAD_GRAYSCALE)
    # sample3 = Morphology.thresholding(sample3, 128)
    # sample3 = Morphology.complement(Morphology.opening(sample3, Morphology.BOUNDARY_KERNEL))
    # sample3 = Morphology.complement(outer_hole_filling(sample3))
    # analyzer.shape_analyze(sample3, discard_size=70, path=args.output_path, filename='sample3_label.png')
    
    
    # sample4 = cv2.imread(os.path.join(args.input_path, 'sample4.png'), cv2.IMREAD_GRAYSCALE)
    # sample4 = Morphology.complement(Morphology.thresholding(sample4, 128))
    # sample4 = Morphology.complement(Morphology.opening(sample4, Morphology.BOUNDARY_KERNEL))
    # sample4 = Morphology.complement(outer_hole_filling(sample4))
    # analyzer.shape_analyze(sample4, discard_size=70, path=args.output_path, filename='sample4_label.png')
    

if __name__ == '__main__':
    main()