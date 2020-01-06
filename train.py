import os
import argparse

import torch
from torch.utils.data import DataLoader

from transforms import get_transform
from dataloader import TableDataset
from splerge import Splerge
from utils import splerge_loss

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--train_images_dir", dest="train_images_dir", help="Path to training data images.")
parser.add_argument("-l", "--train_labels_dir", dest="train_labels_dir", help="Path to training data labels.")
parser.add_argument("-o","--output_weight_path", dest="output_weight_path", help="Output path for weights.", default="model")
parser.add_argument("-e","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=10)
parser.add_argument("--cf","--check_freq", type=int, dest="checkpoint_freq", help="Save checkpoints after given epochs", default=50)
parser.add_argument("-b","--batch_size", type=int, dest="batch_size", help="batch size of training samples", default=2)
parser.add_argument("--lr","--learning_rate", dest="learning_rate", help="learning rate", default=0.0005)
parser.add_argument("--vs","--validation_split", dest="validation_split", help="validation split in data", default=0.2)

options = parser.parse_args()

batch_size = options.batch_size
learning_rate = options.learning_rate
num_workers = 1

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("creating splerge model...")
model = Splerge().to(device)
print(model)

criterion = splerge_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

num_epochs = options.num_epochs

model.train()
print("starting training...")
for epoch in range(num_epochs):
    if ((epoch+1) % checkpoint_freq == 0):
        print("saving model weights at epoch", epoch+1)
        torch.save(model.state_dict(), MODEL_STORE_PATH+'/model_ep{}.pth'.format(epoch+1))

    for i, (images, targets, img_path) in enumerate(train_loader):
        # print(img_path)
        images = images.to(device)
        
        targets[0] = targets[0].long().to(device)
        targets[1] = targets[1].long().to(device)
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()        
        
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        # # Track the accuracy
        # total = labels.size(0)
        # _, predicted = torch.max(outputs.data, 1)
        # correct = (predicted == labels).sum().item()
        # acc_list.append(correct / total)

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

