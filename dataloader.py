import os
import torch

import numpy as np
import cv2

from skimage import transform as sktsf
from torchvision import transforms as tvtsf

def preprocess_image(image, min_size=600, max_size=600):
    # Rescaling Images
    C, H, W = image.shape
    min_size = min_size
    max_size = max_size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = image / 255.
    # image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    image = sktsf.resize(image, (C, min_size, max_size), mode='reflect',anti_aliasing=False)

    # Normalizing image
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    image = normalize(torch.from_numpy(image))

    return image

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, root, train_images_path, train_labels_path, transforms=None):
        
        self.root = root
        self.transforms = transforms
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        
        self.img_paths = list(sorted(os.listdir(os.path.join(self.root, self.train_images_path ))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.train_images_path , self.img_paths[idx])
        img_name = img_path.split("/")[-1][:-4]

        row_label_path = os.path.join(self.root, self.train_labels_path , img_name+"_row.txt")
        col_label_path = os.path.join(self.root, self.train_labels_path , img_name+"_col.txt")
        
        image = cv2.imread(img_path)        
        image = image.astype('float32')

        if image.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            image = image[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))

        C, H, W = image.shape
        image = preprocess_image(image)
        image = image.numpy()

        if not (os.path.isfile(row_label_path) and os.path.isfile(col_label_path)):
            print('[error] label file not found')
            exit(0)

        _, o_H, o_W = image.shape
        scale = o_H / H
        
        with open(row_label_path, "r") as f:
            row_label = f.read().split("\n")

        with open(col_label_path, "r") as f:
            col_label = f.read().split("\n")

        row_label = np.array([[int(x) for x in row_label if x != ""]])
        col_label = np.array([[int(x) for x in col_label if x != ""]])

        row_target = cv2.resize(row_label, (o_H, 1), interpolation=cv2.INTER_NEAREST)
        col_target = cv2.resize(col_label, (o_W, 1), interpolation=cv2.INTER_NEAREST)
        
        row_target[row_target==255] = 1
        col_target[col_target==255] = 1

        row_target = torch.tensor(row_target[0])
        col_target = torch.tensor(col_target[0])

        target = [row_target, col_target]
        # print(target[0].shape, target[1].shape)
        image = image.transpose((1,2,0))

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target, img_path

    def __len__(self):
        return len(self.img_paths)

