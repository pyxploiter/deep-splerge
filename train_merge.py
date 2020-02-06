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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Creating merge model...")
model = MergeModel().to(device)

criterion = merge_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=configs.decay_rate)

# create the summary writer
writer = SummaryWriter()
torch.autograd.set_detect_anomaly(True)
# Train the model
total_step = len(train_loader)

print(27*"=", "Training", 27*"=")
                    
merge_list = open("data/merge_images.txt").read().split("\n")[:-1]
total_loss = 0
total_d_loss = 0
total_r_loss = 0

restore = False
if restore:
    checkpoint = torch.load("model/merge_model_hr.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    step = checkpoint['iteration']
    print("Restoring model from:")
    print("Iterations:", step)
    print("Epochs:", epoch)
    print("Loss:", loss)
else:
    step = 0

# MERGE DATA, Adamax 0.002, [0.5,1]
for epoch in range(num_epochs):

    for i, (image, _, img_path, W, H) in enumerate(train_loader):
        try:
            model.train()

            img_name = img_path[0].split("/")[-1][:-4]

            if img_name+".png" in merge_list: 
                # incrementing step
                step -=- 1

                with open("data/split_outs/"+img_name+".pkl", "rb") as f:
                    split_outputs = pickle.load(f)

                row_prob = split_outputs["row_prob"]
                col_prob = split_outputs["col_prob"]

                thresh = 0.7

                col_prob_img = utils.probs_to_image(col_prob.detach().clone(), image.shape, axis=0)
                row_prob_img = utils.probs_to_image(row_prob.detach().clone(), image.shape, axis=1)

                col_region = col_prob_img.detach().clone()
                col_region[col_region > thresh] = 1
                col_region[col_region <= thresh] = 0
                col_region = (~col_region.bool()).float()

                row_region = row_prob_img.detach().clone()
                row_region[row_region > thresh] = 1
                row_region[row_region <= thresh] = 0
                row_region = (~row_region.bool()).float()    

                grid_img, row_img, col_img = utils.binary_grid_from_prob_images(row_prob_img, col_prob_img)

                row_img = cv2.resize(row_img[0,0].numpy(), (W[0].item(), H[0].item()))
                col_img = cv2.resize(col_img[0,0].numpy(), (W[0].item(), H[0].item()))

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

                loss, d_loss, r_loss = merge_loss(outputs, (gt_down, gt_right), [0.5,1])

                loss.backward()
                # d_loss.backward(retain_graph=True)
                # r_loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_d_loss += d_loss.item()
                total_r_loss += r_loss.item()

                if (step) % configs.log_every == 0:
                    # writing loss to tensorboard
                    writer.add_scalar("training losses/combined loss", loss.item(), step)
                    writer.add_scalar("training losses/down loss", d_loss.item(), step)
                    writer.add_scalar("training losses/right loss", r_loss.item(), step)

                    writer.add_scalar("training average losses/combined loss", total_loss/step, step)
                    writer.add_scalar("training average losses/down loss", total_d_loss/step, step)
                    writer.add_scalar("training average losses/right loss", total_r_loss/step, step)

                    print("Iteration:", step, "Learning Rate:", lr_scheduler.get_lr()[0])
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Down Loss: {:.4f}, Right Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), d_loss.item(), r_loss.item()))
                    print("---")

                if (step) % configs.val_every == 0:

                    print(26*"~", "Validation", 26*"~")
                    model.eval()
                    with torch.no_grad():
                        val_loss_list = list()
                        val_down_loss_list = list()
                        val_right_loss_list = list()

                        for x, (val_image, _, val_img_path, Wv, Hv) in enumerate(val_loader):
                                        
                            val_img_name = val_img_path[0].split("/")[-1][:-4]

                            with open("data/split_outs/"+val_img_name+".pkl", "rb") as f:
                                val_split_outputs = pickle.load(f)

                            val_row_prob = val_split_outputs["row_prob"]
                            val_col_prob = val_split_outputs["col_prob"]

                            thresh = 0.7

                            vcol_prob_img = utils.probs_to_image(val_col_prob.detach().clone(), val_image.shape, axis=0)
                            vrow_prob_img = utils.probs_to_image(val_row_prob.detach().clone(), val_image.shape, axis=1)

                            vcol_region = vcol_prob_img.detach().clone()
                            vcol_region[vcol_region > thresh] = 1 
                            vcol_region[vcol_region <= thresh] = 0
                            vcol_region = (~vcol_region.bool()).float()

                            vrow_region = vrow_prob_img.detach().clone()
                            vrow_region[vrow_region > thresh] = 1
                            vrow_region[vrow_region <= thresh] = 0
                            vrow_region = (~vrow_region.bool()).float()    

                            vgrid_img, vrow_img, vcol_img = utils.binary_grid_from_prob_images(vrow_prob_img, vcol_prob_img)

                            vrow_img = cv2.resize(vrow_img[0,0].numpy(), (Wv[0].item(), Hv[0].item()))
                            vcol_img = cv2.resize(vcol_img[0,0].numpy(), (Wv[0].item(), Hv[0].item()))

                            vgt_down, vgt_right = utils.create_merge_gt(vrow_img, vcol_img, os.path.join(merges_path, img_name + ".pkl"))
                            
                            vinput_feature = torch.cat((val_image, 
                                                    vrow_prob_img, 
                                                    vcol_prob_img,
                                                    vrow_region, 
                                                    vcol_region, 
                                                    vgrid_img), 
                                                1)

                            val_outputs = model(vinput_feature.to(device))
                            vloss, vd_loss, vr_loss = merge_loss(val_outputs, (vgt_down, vgt_right))
                            
                            val_loss_list.append(vloss.item())
                            val_down_loss_list.append(vd_loss.item())
                            val_right_loss_list.append(vr_loss.item())

                            print('Step [{}/{}], Val Loss: {:.4f}, Down Loss: {:.4f}, Right Loss: {:.4f}'
                                .format(x + 1, len(val_loader), vloss, vd_loss, vr_loss))

                        avg_val_loss = np.mean(np.array(val_loss_list))
                        avg_down_val_loss = np.mean(np.array(val_down_loss_list))
                        avg_right_val_loss = np.mean(np.array(val_right_loss_list))

                        writer.add_scalar("validation losses/combined loss val", avg_val_loss, (epoch*total_step + i))
                        writer.add_scalar("validation losses/down loss val", avg_down_val_loss, (epoch*total_step + i))
                        writer.add_scalar("validation losses/right loss val", avg_right_val_loss, (epoch*total_step + i))
                      
                    print(64*"~")
                    for name, param in model.named_parameters():
                        if "bias" in name:
                            writer.add_histogram("bias/"+name, param, step)
                        else:
                            writer.add_histogram("weights/"+name, param, step)

                if ((step+1) % configs.save_every == 0):
                    print(65*"=")
                    print("Saving model weights at iteration", step+1)
                    
                    torch.save({
                        'epoch': epoch+1,
                        'iteration': step+1,
                        'learning_rate': 0.01,
                        'optimizer': 'Adamax',
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss,
                        'd_loss': total_d_loss,
                        'r_loss': total_r_loss
                    }, MODEL_STORE_PATH+'/merge_model_'+str(step+1)+'.pth')

                    print(65*"=")

                torch.cuda.empty_cache()
        except Exception as e:
            print(e)


    lr_scheduler.step()
