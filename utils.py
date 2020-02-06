import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform as sktsf
from torchvision import transforms as tvtsf

def collate_fn(batch):
    return tuple(zip(*batch))

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:

                tensor = getattr(n[0], 'variable')
                print(n[0])
                # print('tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print(' - grad_fn:', tensor.grad_fn)
                print()
            except AttributeError as e:
                print(e)
                getBack(n[0])


def normalize_numpy_image(image):
    # Normalizing image
    norm = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    image = norm(torch.from_numpy(image))

    return image

def resize_image(image, min_size=600, max_size=1024, fix_resize=False):
    # Rescaling Images
    C, H, W = image.shape
    min_size = min_size
    max_size = max_size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = image / 255.

    if fix_resize:
        image = sktsf.resize(image, (C, min_size, max_size), mode='reflect',anti_aliasing=False)
    else:
        image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
        
    return image

def get_column_separators(image, smoothing=2, is_row=True):
    if is_row:
        cols = np.where(np.sum(image, axis=1)!=0)[0]
    else:
        cols = np.where(np.sum(image, axis=0)!=0)[0]

    if len(cols) == 0:
        return []

    adjacent_cols = [cols[0]]
    final_seperators = []
    for i in range(1, len(cols)):

        if cols[i] - cols[i - 1] < smoothing:
            adjacent_cols.append(cols[i])
            
        elif len(adjacent_cols) > 0:
            final_seperators.append(sum(adjacent_cols) // len(adjacent_cols))
            adjacent_cols = [cols[i]]

    if len(adjacent_cols) > 0:
        final_seperators.append(sum(adjacent_cols) // len(adjacent_cols))
    
    return final_seperators

def get_midpoints_from_grid(grid):

    row_sep = np.where(np.sum(grid, axis=1) == grid.shape[1])[0]
    col_sep = np.where(np.sum(grid, axis=0) == grid.shape[0])[0]

    def find_midpoint(indices):

        adj_indices = [indices[0]]
        midpoints = []

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] == 1:
                adj_indices.append(indices[i])
            
            elif len(adj_indices) > 0:
                midpoints.append(sum(adj_indices) // len(adj_indices))
                adj_indices = [indices[i]]

        if len(adj_indices) > 0:
            midpoints.append(sum(adj_indices) // len(adj_indices))

        return midpoints

    col_midpoints, row_midpoints = [], []

    if len(row_sep):
        row_midpoints = find_midpoint(row_sep)

    if len(col_sep):
        col_midpoints = find_midpoint(col_sep)
        
    return row_midpoints, col_midpoints


def tensor_to_numpy_image(tensor, display=False, write_path=None, threshold=0.7):
    tensor = tensor.squeeze(0)  #1,c,h,w -> c,h,w
    c, h, w = tensor.shape
    np_array = np.array(tensor.view(h,w,c).detach())
    np_array[np_array > threshold] = 255
    np_array[np_array <= threshold] = 0

    if display:
        cv2.imshow("image"+str(torch.rand(3)), np_array)
        cv2.waitKey(0)
    if write_path:
        cv2.imwrite(write_path, np_array)

    return np_array

def probs_to_image(tensor, image_shape, axis):
    """this converts probabilities tensor to image"""
    # (1, 1, n) = tensor.shape
    # b,c,h,w = image_shape
    b, c, h, w = image_shape
    if axis == 0:
        tensor_image = tensor.view(1,1,tensor.shape[2]).repeat(1,h,1)
    
    elif axis == 1:
        tensor_image = tensor.view(1,tensor.shape[2],1).repeat(1,1,w)
    
    else:
        print("Error: invalid axis.")

    return tensor_image.unsqueeze(0) 

def binary_grid_from_prob_images(row_prob_img, col_prob_img, thresh=0.7, row_smooth=5, col_smooth=15):
    
    row_prob_img[row_prob_img > thresh] = 1
    row_prob_img[row_prob_img <= thresh] = 0
    
    col_prob_img[col_prob_img > thresh] = 1
    col_prob_img[col_prob_img <= thresh] = 0
        
    row_indices = get_column_separators(row_prob_img.squeeze(0).squeeze(0).detach().numpy(), smoothing=row_smooth, is_row=True)
    col_indices = get_column_separators(col_prob_img.squeeze(0).squeeze(0).detach().numpy(), smoothing=col_smooth, is_row=False)

    col_smooth_image = torch.zeros(col_prob_img.shape)
    row_smooth_image = torch.zeros(row_prob_img.shape)

    for i in col_indices:
        col_img = col_smooth_image[0][0].transpose(1,0)
        if (i > 0):
            col_img[i+1:i+4] = 1.0
            col_img[max(0,i-3):i+1] = 1.0

    row_img = row_smooth_image[0][0]
    for i in row_indices:
        if (i > 0):
            row_img[i+1:i+4] = 1.0
            row_img[max(0,i-3):i+1] = 1.0

    row_img = row_img.unsqueeze(0).unsqueeze(0) 

    grid = row_img.int() | col_smooth_image.int()
    grid = grid.float()
    
    return grid, row_img, col_smooth_image

def create_merge_gt(row_image, col_image, merge_file):
    rows = np.array(get_column_separators(row_image, smoothing=2, is_row=True))
    cols = np.array(get_column_separators(col_image, smoothing=2, is_row=False))

    with open(merge_file, "rb") as f:
        merges = pickle.load(f)

    gt_down = np.zeros((rows.shape[0], cols.shape[0] + 1))
    gt_right = np.zeros((rows.shape[0] + 1, cols.shape[0]))

    for rect in merges["row"]:
        start_col = np.amax([0] + list(np.where(rect[0] > cols)[0] + 1))
        end_col = np.amax([0] + list(np.where(rect[2] > cols)[0] + 1))

        start_row = np.amax([0] + list(np.where(rect[1] > rows)[0] + 1))
        end_row = np.amax([0] + list(np.where(rect[3] > rows)[0] + 1))

        for i in range(start_col, end_col + 1):
            for j in range(start_row, end_row):
                gt_down[j, i] = 1

        for i in range(start_col, end_col):
            for j in range(start_row, end_row + 1):
                gt_right[j, i] = 1

    for rect in merges["col"]:
        start_col = np.amax([0] + list(np.where(rect[0] > cols)[0] + 1))
        end_col = np.amax([0] + list(np.where(rect[2] > cols)[0] + 1))

        start_row = np.amax([0] + list(np.where(rect[1] > rows)[0] + 1))
        end_row = np.amax([0] + list(np.where(rect[3] > rows)[0] + 1))

        for i in range(start_col, end_col + 1):
            for j in range(start_row, end_row):
                gt_down[j, i] = 1

        for i in range(start_col, end_col):
            for j in range(start_row, end_row + 1):
                gt_right[j, i] = 1

    return torch.Tensor(gt_down), torch.Tensor(gt_right)

def draw_merge_output(image, grid_img, col_merge, row_merge, colors=((0,0,255),(255,0,0))):

    grid = grid_img.squeeze(0).squeeze(0)
    image = cv2.resize(image, (grid.shape[1], grid.shape[0]))
    
    image_cpy = image.copy()

    row_mids, col_mids = [0], [0]
    np_grid = grid.cpu().numpy()
    r_mids, c_mids = get_midpoints_from_grid(np_grid)

    row_mids.extend(r_mids)
    row_mids.append(grid.shape[0])
    col_mids.extend(c_mids)
    col_mids.append(grid.shape[1])

    for row_id in range(1, len(row_mids)):
        for col_id in range(1, len(col_mids)):

            if row_id <= row_merge.shape[0]:
                # print("row:",row_merge.shape, row_id-1, col_id-1)
                if row_merge[row_id-1][col_id-1].item():
                    x1, y1 = (col_mids[col_id-1]+col_mids[col_id])//2, (row_mids[row_id-1]+row_mids[row_id])//2
                    x2, y2 = (col_mids[col_id-1]+col_mids[col_id])//2, (row_mids[row_id]+row_mids[row_id+1])//2
                    cv2.line(image_cpy, (x1,y1), (x2,y2), colors[0], 10) 
                    
            if col_id <= col_merge.shape[1]:
                # print("col:",col_merge.shape, row_id-1, col_id-1)
                if col_merge[row_id-1][col_id-1].item() == 1:
                    x1, y1 = (col_mids[col_id-1]+col_mids[col_id])//2, (row_mids[row_id-1]+row_mids[row_id])//2
                    x2, y2 = (col_mids[col_id]+col_mids[col_id+1])//2, (row_mids[row_id-1]+row_mids[row_id])//2
                    cv2.line(image_cpy, (x1,y1), (x2,y2), colors[1], 10)
                    
    alpha = 0.4
    image = cv2.addWeighted(image, alpha, image_cpy, 1-alpha, gamma=0)
    return image