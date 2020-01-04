import os
import torch
from torch.utils.data import DataLoader

from transforms import get_transform
from dataloader import TableDataset
from splerge import Splerge
from utils import splerge_loss
# from utils import collate_fn

batch_size = 2
num_workers = 1
learning_rate = 0.0001

MODEL_STORE_PATH = 'model'

train_images_path = "data/images"
train_labels_path = "data/labels"

print("Loading dataset...")
dataset = TableDataset(os.getcwd(), train_images_path, train_labels_path, get_transform(train=True))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

train_dataset = torch.utils.data.Subset(dataset, indices[:-20])
test_dataset = torch.utils.data.Subset(dataset, indices[-20:])

# define training and validation data loaders
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
   # collate_fn=collate_fn)

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

num_epochs = 300
save_model = False

model.train()
print("starting training...")
for epoch in range(num_epochs):
    if ((epoch+1) % 10 == 0) and save_model:
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

