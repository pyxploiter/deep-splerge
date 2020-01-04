import os
import cv2
import torch
from torch.utils.data import DataLoader

from dataloader import TableDataset
from transforms import get_transform
from splerge import Splerge

model_path = "model/model_ep300.pth"

train_images_path = "data/images"
train_labels_path = "data/labels"
output_path = "evaluations/"

batch_size = 1
num_workers = 1

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

print("loading weights...")
model.load_state_dict(torch.load(model_path))

model.eval()
print("starting evaluation...")
with torch.no_grad():
    for i, (images, targets, img_path) in enumerate(train_loader):

        images = images.to(device)

        targets[0] = targets[0].long().to(device)
        targets[1] = targets[1].long().to(device)

        output = model(images)
        rpn_o, cpn_o = output

        r3, r4, r5 = rpn_o
        c3, c4, c5 = cpn_o

        r3, r4, r5 = r3[0][0], r4[0][0], r5[0][0] 
        c3, c4, c5 = c3[0][0], c4[0][0], c5[0][0]

        tr5 = targets[0][0]
        tc5 = targets[1][0]

        cout = c5.clone()
        cout[cout > 0.7] = 255
        cout[cout <= 0.7] = 0
        cout = cout.view(1, cout.shape[0]).repeat(cout.shape[0], 1)
        cout = cout.cpu().numpy()

        rout = r5.clone()
        rout[rout > 0.7] = 255
        rout[rout <= 0.7] = 0
        rout = rout.view(rout.shape[0], 1).repeat(1, rout.shape[0])
        rout = rout.cpu().numpy()
        
        tcout = tc5.view(1, tc5.shape[0]).repeat(tc5.shape[0], 1)
        tcout = tcout.cpu().numpy()
        tcout[tcout == 1] = 255

        trout = tr5.view(tr5.shape[0], 1).repeat(1, tr5.shape[0])
        trout = trout.cpu().numpy()
        trout[trout == 1] = 255

        img_name = img_path[0].split("/")[-1][:-4]
        cv2.imwrite(output_path+img_name+"_col_out.png", cout)
        cv2.imwrite(output_path+img_name+"_row_out.png", rout)
        cv2.imwrite(output_path+img_name+"_col.png", tcout)
        cv2.imwrite(output_path+img_name+"_row.png", trout)
        
        print('Step [{}/{}] Image: {}'
          .format(i+1, len(train_loader), img_name))

        # cv2.imshow("col_out", cout.astype("uint8"))
        # cv2.imshow("col", tcout.astype("uint8"))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow("row_out", rout.astype("uint8"))
        # cv2.imshow("row", trout.astype("uint8"))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # print(torch.abs(tr5 - r5))
        # print(torch.sum(torch.abs(tr5 - r5)))
        # print(torch.sum(torch.abs(tc5 - c5)))
