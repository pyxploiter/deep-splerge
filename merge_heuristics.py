import cv2
import os
import glob
import utility as ut
import utilscp as ut1
import GTElement as gc
import numpy as np

import torch
from torch.utils.data import DataLoader
from dataloader import TableDataset
from transforms import get_transform
from splerge import Splerge
import utils


def col_merging(_table, ocr_data):
    for word in ocr_data:
        for col in _table.gtCols:
            if (word[2] < col.x1) and (word[4] > col.x2):
                merging_col = gc.Row(word[2], word[3], word[4])
                # print("x================x================x")
                # print("col merge happening at: ")
                # print(merging_col)
                # print(word)
                # print("x================x================x")
                _table.gtSpans.append(merging_col)
    _table.evaluateCells()


def row_merging(_table, ocr_data):
    for word in ocr_data:
        for row in _table.gtRows:
            if (word[3] < row.y1) and (word[5] > row.y2):
                merging_row = gc.Column(word[2], word[3], word[5])
                # print("x================x================x")
                # print("row merge happening at: ")
                # print(merging_row)
                # print(word)
                # print("x================x================x")

                _table.gtSpans.append(merging_row)
    _table.evaluateCells()


def get_table(_list_rows, _list_col, h, w):
    _list_row_obj = []
    _list_col_obj = []
    _table = gc.Table(0, 0, w, h)
    for row in _list_rows:
        _table.gtRows.append(gc.Row(0, row, w))
    for col in _list_col:
        _table.gtCols.append(gc.Column(col, 0, h))
    _table.evaluateCells()
    return _table


def get_grid_structure(img1, img2):
    _list_rows = ut1.get_column_separators(img1, smoothing=10, is_row=True)
    _list_col = ut1.get_column_separators(img2, smoothing=20, is_row=False)
    img3 = np.zeros_like(img1)
    img3[:, _list_col] = 255
    img4 = np.zeros_like(img1)
    img4[_list_rows, :] = 255
    # cv2.imwrite("merge-heuristic-data/col_final.png", img3)
    # cv2.imwrite("merge-heuristic-data/row_final.png", img4)
    return _list_rows, _list_col


def execute_pipeline(img1, img2, ocr_data, org_image):
    h, w, _ = org_image.shape
    _list_rows, _list_col = get_grid_structure(img1, img2)
    org_image[_list_rows, :] = (255, 0, 255)
    org_image[:, _list_col] = (255, 0, 255)
    _table = get_table(_list_rows, _list_col, h, w)
    col_merging(_table, ocr_data)
    row_merging(_table, ocr_data)
    return _table


def visualize_merging(_table, img, image_name):
    for item in _table.gtSpans:
        cv2.line(img, (item.x1, item.y1), (item.x2, item.y2), (0, 255, 0), 2)
    cv2.imwrite("merge-heuristic-data/" + image_name + ".png", img)


# if __name__ == "__main__":
#     img1, img2, ocr_data, org_image = prep()
#     final_table = execute_pipeline(img1, img2, ocr_data, org_image)
#     visualize_merging(final_table, org_image)

if __name__ == "__main__":
    model_path = "model/model_625k.pth"

    train_images_path = "data/images"
    train_labels_path = "data/labels"
    ocr_path = "data/augmented/ocr"
    output_path = "evaluations/"

    batch_size = 1
    num_workers = 1

    print("Loading dataset...")
    dataset = TableDataset(
        os.getcwd(), train_images_path, train_labels_path, get_transform(train=True)
    )

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    train_dataset = torch.utils.data.Subset(dataset, indices)

    # define training and validation data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("creating splerge model...")
    model = Splerge().to(device)
    # print(model)

    print("loading weights...")
    model.load_state_dict(torch.load(model_path))

    model.eval()

    errored = []

    print("starting evaluation...")
    with torch.no_grad():
        for i, (images, targets, img_path) in enumerate(train_loader):
            print(i, "/", len(train_loader), ":", img_path[0].split("/")[-1][:-4])

            try:
                img_name = img_path[0].split("/")[-1][:-4]

                images = images.to(device)

                output = model(images)
                rpn_o, cpn_o = output

                r3, r4, r5 = rpn_o
                c3, c4, c5 = cpn_o

                row_img = utils.probs_to_image(r5, images.shape, axis=1)
                col_img = utils.probs_to_image(c5, images.shape, axis=0)

                row_img = utils.tensor_to_numpy_image(row_img.cpu())
                col_img = utils.tensor_to_numpy_image(col_img.cpu())

                orig_image = cv2.imread(img_path[0])

                h, w, _ = orig_image.shape
                col_img = cv2.resize(col_img, (w, h))
                row_img = cv2.resize(row_img, (w, h))

                ocr_data = ut.read_ocr(os.path.join(ocr_path, img_name[:-2] + ".pkl"))

                final_table = execute_pipeline(row_img, col_img, ocr_data, orig_image)
                visualize_merging(final_table, orig_image, img_name)

            except Exception as e:
                print("Error Image:", img_name)
                print(e)
                errored.append(img_name)

    for i in errored:
        print(i)
