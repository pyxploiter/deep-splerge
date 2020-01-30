import os
import pickle
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils

from transforms import get_transform
from dataloader import SplitTableDataset
from losses import merge_loss
from merge import MergeModel

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--train_images_dir", dest="train_images_dir", help="Path to training data images.", default="data/org_images")
parser.add_argument("-l", "--train_labels_dir", dest="train_labels_dir", help="Path to training data labels.", default="data/labels")
parser.add_argument("-o","--output_weight_path", dest="output_weight_path", help="Output path for weights.", default="model")
parser.add_argument("-e","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=50)
parser.add_argument("-s","--save_every", type=int, dest="save_every", help="Save checkpoints after given epochs", default=1000)
parser.add_argument("--log_every", type=int, dest="log_every", help="Print logs after every given steps", default=1)
parser.add_argument("--val_every", type=int, dest="val_every", help="perform validation after given steps", default=1000)
parser.add_argument("-b","--batch_size", type=int, dest="batch_size", help="batch size of training samples", default=1)
parser.add_argument("--lr","--learning_rate", type=float, dest="learning_rate", help="learning rate", default=0.00075)
parser.add_argument("--dr","--decay_rate", type=float, dest="decay_rate", help="weight decay rate", default=0.75)
parser.add_argument("--vs","--validation_split", type=float, dest="validation_split", help="validation split in data", default=0.008)

configs = parser.parse_args()

print(25*"=", "Configuration", 25*"=")
print("Train Images Directory:", configs.train_images_dir)
print("Train Labels Directory:", configs.train_labels_dir)
print("Validation Split:", configs.validation_split)
print("Output Weights Path:", configs.output_weight_path)
print("Number of Epochs:", configs.num_epochs)
print("Save Checkpoint Frequency:", configs.save_every)
print("Display logs after steps:", configs.log_every)
print("Perform validation after steps:", configs.val_every)
print("Batch Size:", configs.batch_size)
print("Learning Rate:", configs.learning_rate)
print("Decay Rate:", configs.decay_rate)
print(65*"=")

batch_size = configs.batch_size
learning_rate = configs.learning_rate
num_epochs = configs.num_epochs

MODEL_STORE_PATH = configs.output_weight_path

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Creating merge model...")
model = MergeModel().to(device)

print("loading weights...")

checkpoint = torch.load("model/merge_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']

criterion = merge_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=configs.decay_rate)

# create the summary writer
writer = SummaryWriter()

# Train the model
total_step = len(train_loader)

print(27*"=", "Training", 27*"=")

model.eval()
step = 0
for epoch in range(1):
    with torch.no_grad():
        for i, (image, target, img_path, W, H) in enumerate(val_dataset):
                image = image.unsqueeze(0)
                # incrementing step
                step -=- 1
                img_name = img_path.split("/")[-1][:-4]

                with open("data/split_outs/"+img_name+".pkl", "rb") as f:
                    split_outputs = pickle.load(f)

                row_prob = split_outputs["row_prob"]
                col_prob = split_outputs["col_prob"]

                thresh = 0.7
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

                # print(image.shape)
                # print(row_img.shape)
                # 
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

                # writer.add_graph(model, input_feature.to(device))

                optimizer.zero_grad()
                outputs = model(input_feature.to(device))


                loss, d_loss, r_loss = merge_loss(outputs, (gt_down, gt_right))

                print("Down:", outputs[1])
                print("GT DOWN:",gt_down)
                print("\nRight", outputs[3])
                print("GT Right", gt_right)
                torch.cuda.empty_cache()
