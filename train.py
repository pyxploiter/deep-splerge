import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transforms import get_transform
from dataloader import TableDataset
from splerge import Splerge
from utils import splerge_loss

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--train_images_dir", dest="train_images_dir", help="Path to training data images.")
parser.add_argument("-l", "--train_labels_dir", dest="train_labels_dir", help="Path to training data labels.")
parser.add_argument("-o","--output_weight_path", dest="output_weight_path", help="Output path for weights.", default="model")
parser.add_argument("-e","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=10)
parser.add_argument("-s","--save_every", type=int, dest="save_every", help="Save checkpoints after given epochs", default=50)
parser.add_argument("--log_every", type=int, dest="log_every", help="Print logs after every given steps", default=10)
parser.add_argument("--val_every", type=int, dest="val_every", help="perform validation after given steps", default=100)
parser.add_argument("-b","--batch_size", type=int, dest="batch_size", help="batch size of training samples", default=2)
parser.add_argument("--lr","--learning_rate", type=float, dest="learning_rate", help="learning rate", default=0.00075)
parser.add_argument("--dr","--decay_rate", type=float, dest="decay_rate", help="decay rate", default=0.75)
parser.add_argument("--vs","--validation_split", type=float, dest="validation_split", help="validation split in data", default=0.2)

options = parser.parse_args()

print(25*"=", "Configuration", 25*"=")
print("Train Images Directory:", options.train_images_dir)
print("Train Labels Directory:", options.train_labels_dir)
print("Validation Split:", options.validation_split)
print("Output Weights Path:", options.output_weight_path)
print("Number of Epochs:", options.num_epochs)
print("Save Checkpoint Frequency:", options.save_every)
print("Display logs after steps:", options.log_every)
print("Perform validation after steps:", options.val_every)
print("Batch Size:", options.batch_size)
print("Learning Rate:", options.learning_rate)
print("Decay Rate:", options.decay_rate)
print(65*"=")

batch_size = options.batch_size
learning_rate = options.learning_rate

MODEL_STORE_PATH = options.output_weight_path

train_images_path = options.train_images_dir
train_labels_path = options.train_labels_dir

print("Loading dataset...")
dataset = TableDataset(os.getcwd(), train_images_path, train_labels_path, get_transform(train=True))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

test_split = int(options.validation_split * len(indices))

train_dataset = torch.utils.data.Subset(dataset, indices[test_split:])
test_dataset = torch.utils.data.Subset(dataset, indices[:test_split])

# define training and validation data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Creating splerge model...")
model = Splerge().to(device)
# print(model)

criterion = splerge_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = options.num_epochs

# create the summary writer
writer = SummaryWriter()

# Train the model
total_step = len(train_loader)

print(27*"=", "Training", 27*"=")

# TODO: decay learning rate by factor of 0.75 after every 80k steps

for epoch in range(num_epochs):
    if ((epoch+1) % options.save_every == 0):
        print(65*"=")
        print("Saving model weights at epoch", epoch+1)
        torch.save(model.state_dict(), MODEL_STORE_PATH+'/model_ep{}.pth'.format(epoch+1))
        print(65*"=")

    for i, (images, targets, img_path) in enumerate(train_loader):
        # write model architecture to tensorboard
        writer.add_graph(model, images)
        model.train()
        step -=- 1
        
        targets[0] = targets[0].long().to(device)
        targets[1] = targets[1].long().to(device)
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()        
        
        # Run the forward pass
        outputs = model(images.to(device))
        loss, rpn_loss, cpn_loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (i+1) % options.log_every == 0:
            #writing loss to tensorboard
            writer.add_scalar("total loss - train",loss.item(), (epoch*total_step + i))
            writer.add_scalar("rpn loss - train", rpn_loss.item(), (epoch*total_step + i))
            writer.add_scalar("cpn loss - train", cpn_loss.item(), (epoch*total_step + i))

            print('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, RPN Loss: {:.4f}, CPN Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), rpn_loss.item(), cpn_loss.item()))

        if (i+1) % options.val_every == 0:
            print(26*"~", "Validation", 26*"~")
            model.eval()
            with torch.no_grad():
                val_loss_list = list()
                val_rpn_loss_list = list()
                val_cpn_loss_list = list()

                for x, (val_images, val_targets, _) in enumerate(test_loader):
                    val_targets[0] = val_targets[0].long().to(device)
                    val_targets[1] = val_targets[1].long().to(device)

                    val_outputs = model(val_images.to(device))
                    val_loss, val_rpn_loss, val_cpn_loss = criterion(val_outputs, val_targets)
                    
                    val_loss_list.append(val_loss.item())
                    val_rpn_loss_list.append(val_rpn_loss.item())
                    val_cpn_loss_list.append(val_cpn_loss.item())

                avg_val_loss = np.mean(np.array(val_loss_list))
                avg_rpn_val_loss = np.mean(np.array(val_rpn_loss_list))
                avg_cpn_val_loss = np.mean(np.array(val_cpn_loss_list))

                writer.add_scalar("total loss - val", avg_val_loss, (epoch*total_step + i))
                writer.add_scalar("total loss - val", avg_rpn_val_loss, (epoch*total_step + i))
                writer.add_scalar("total loss - val", avg_cpn_val_loss, (epoch*total_step + i))

                print('Step [{}/{}], Validation Loss: {:.4f}'
                  .format(x + 1, len(test_loader), avg_val_loss))
            print(64*"~")

        

