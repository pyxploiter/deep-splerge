import os
import pickle
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils

from transforms import get_transform
from dataloader import SplitTableDataset
from merge import MergeModel

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--test_images_dir", dest="train_images_dir", help="Path to training data images.", default="data/org_images")
parser.add_argument("-l", "--test_labels_dir", dest="train_labels_dir", help="Path to training data labels.", default="data/labels")
parser.add_argument("-e","--eval", dest="eval", action="store_true", help="evaluation flag")
parser.add_argument("-w","--weight_path", dest="weight_path", help="path for model weights.", default="model")
parser.add_argument("-t","--thresh", dest="thresh", help="threshold for true positive", default=0.70)
parser.add_argument("--vs","--validation_split", type=float, dest="validation_split", help="validation split in data", default=0.008)

configs = parser.parse_args()

print(25*"=", "Configuration", 25*"=")
print("Data Images Directory:", configs.train_images_dir)
print("Data Labels Directory:", configs.train_labels_dir)
print("Evaluation Mode:", configs.eval)
print(65*"=")

train_images_path = configs.train_images_dir
train_labels_path = configs.train_labels_dir
merges_path = "data/merges"

print("Loading dataset...")
dataset = SplitTableDataset(os.getcwd(), 
                        train_images_path, 
                        train_labels_path, 
                        transforms=get_transform(train=True), 
                        fix_resize=False)

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

test_split = int(configs.validation_split * len(indices))

train_dataset = torch.utils.data.Subset(dataset, indices[test_split:])
val_dataset = torch.utils.data.Subset(dataset, indices[:test_split])

# define training and validation data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Creating merge model...")
model = MergeModel().to(device)

print("loading weights...")
checkpoint = torch.load("model/adamax/merge_model_98000.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Train the model
total_step = len(train_loader)

print(27*"=", "Testing", 27*"=")

merge_list = open("data/merge_images.txt").read().split("\n")[:-1]
model.eval()

if not os.path.exists("outputs"):
      os.mkdir("outputs")

row_tp, row_tn, row_fn, row_fp = 0,0,0,0
col_tp, col_tn, col_fn, col_fp = 0,0,0,0

thresh = configs.thresh
with torch.no_grad():
    for i, (image, target, img_path, W, H) in enumerate(val_dataset):
        try:
            image = image.unsqueeze(0)

            img_name = img_path.split("/")[-1][:-4]
            print(i, len(val_dataset), img_name)

            with open("data/split_outs/"+img_name+".pkl", "rb") as f:
                split_outputs = pickle.load(f)

            row_prob = split_outputs["row_prob"]
            col_prob = split_outputs["col_prob"]

            image_shape = image.shape
            col_prob_img = utils.probs_to_image(col_prob.detach().clone(), image_shape, axis=0)
            row_prob_img = utils.probs_to_image(row_prob.detach().clone(), image_shape, axis=1)

            col_region = col_prob_img.detach().clone()
            col_region[col_region > thresh] = 1 
            col_region[col_region <= thresh] = 0
            col_region = (~col_region.bool()).float()

            row_region = row_prob_img.detach().clone()
            row_region[row_region > thresh] = 1
            row_region[row_region <= thresh] = 0
            row_region = (~row_region.bool()).float()    

            grid_img, row_img, col_img = utils.binary_grid_from_prob_images(row_prob_img, col_prob_img)

            # utils.tensor_to_numpy_image(row_img, write_path="../deep-splerge/eval/row_out/"+img_name+".png")
            # utils.tensor_to_numpy_image(col_img, write_path="../deep-splerge/eval/col_out/"+img_name+".png")   
            # continue         

            row_img = cv2.resize(row_img[0,0].numpy(), (W, H))
            col_img = cv2.resize(col_img[0,0].numpy(), (W, H))

            gt_down, gt_right = utils.create_merge_gt(row_img, col_img, os.path.join(merges_path, img_name + ".pkl"))
            
            input_feature = torch.cat((image, 
                                    row_prob_img, 
                                    col_prob_img,
                                    row_region, 
                                    col_region, 
                                    grid_img), 
                                1)

            outputs = model(input_feature.to(device))

            row_merge = outputs[1].squeeze(0).squeeze(0)
            row_merge[row_merge > thresh] = 1
            row_merge[row_merge <= thresh] = 0
            
            col_merge = outputs[3].squeeze(0).squeeze(0)
            col_merge[col_merge > thresh] = 1
            col_merge[col_merge <= thresh] = 0

            if configs.eval:
                  row_tp += np.count_nonzero(((row_merge.cpu() == 1) & (gt_down == 1)).numpy())
                  row_tn += np.count_nonzero(((row_merge.cpu() == 0) & (gt_down == 0)).numpy())
                  row_fn += np.count_nonzero(((row_merge.cpu() == 0) & (gt_down == 1)).numpy())
                  row_fp += np.count_nonzero(((row_merge.cpu() == 1) & (gt_down == 0)).numpy())

                  col_tp += np.count_nonzero(((col_merge.cpu() == 1) & (gt_right == 1)).numpy())
                  col_tn += np.count_nonzero(((col_merge.cpu() == 0) & (gt_right == 0)).numpy())
                  col_fn += np.count_nonzero(((col_merge.cpu() == 0) & (gt_right == 1)).numpy())
                  col_fp += np.count_nonzero(((col_merge.cpu() == 1) & (gt_right == 0)).numpy())

            grid_np_img = utils.tensor_to_numpy_image(grid_img)
            grid_np_img = cv2.resize(grid_np_img, (W,H))
            grid_np_img = cv2.cvtColor(grid_np_img, cv2.COLOR_GRAY2BGR)
            test_image = cv2.imread(img_path)
            test_image[np.where((grid_np_img == [255, 255, 255]).all(axis = 2))] = [0, 255, 0]

            out_img = utils.draw_merge_output(test_image, grid_img, col_merge, row_merge)
            gt_img = utils.draw_merge_output(test_image, grid_img, gt_right, gt_down, colors=((0,100,255), (255,100,0)))
            out_img = cv2.copyMakeBorder(out_img, 0, 0, 10, 0, cv2.BORDER_CONSTANT)

            compare = np.concatenate((gt_img, out_img), axis=1)
            cv2.imwrite("outputs/"+img_name+".png", compare)
            # cv2.imshow("img", compare)
            # cv2.waitKey(0)
            # exit(0)
        except Exception as e:
            print(e)

row_precision = row_tp / (row_tp + row_fp)
row_recall = row_tp / (row_tp + row_fn)
row_accuracy = (row_tp + row_tn) / (row_tp + row_fp + row_tn + row_fn)  
row_f1 = 2 * (row_precision * row_recall) / (row_precision + row_recall)

col_precision = col_tp / (col_tp + col_fp)
col_recall = col_tp / (col_tp + col_fn)
col_accuracy = (col_tp + col_tn) / (col_tp + col_fp + col_tn + col_fn)  
col_f1 = 2 * (col_precision * col_recall) / (col_precision + col_recall)

print("Row:")
print("Precision:", row_precision)
print("Recall:", row_recall)
print("Accuracy:", row_accuracy)
print("F1 Score:", row_f1)

print("Column:")
print("Precision:", col_precision)
print("Recall:", col_recall)
print("Accuracy:", col_accuracy)
print("F1 Score:", col_f1)