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

base_path = "/home/umar_visionx/Documents/Asad/umar-deep-splerge/deep-splerge/tetsing_data"

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
    _table.populate_ocr(ocr_data)
    _table.merge_header()
    return _table   

def visualize_merging(_table, img, image_name):
    for item in _table.gtSpans:
        cv2.line(img, (item.x1, item.y1), (item.x2, item.y2), (0, 255, 0), 2)
    cv2.imwrite("/home/umar_visionx/Documents/Asad/umar-deep-splerge/deep-splerge/tetsing_data/"+image_name + ".png", img)


if __name__ == "__main__":
    bpath_img = "images"
    bpath_ocr = "ocr"
    bpath_split = "split_images"
    img_path = os.path.join(base_path,bpath_img)
    ocr_path = os.path.join(base_path,bpath_ocr) 
    split_path = os.path.join(base_path,bpath_split)
    _ocr_list = glob.glob(os.path.join(ocr_path, "*.pkl"))
    _img_list = glob.glob(os.path.join(img_path, "*.png"))
    _split_img = glob.glob(os.path.join(split_path, "*.png"))
    img1 = ~cv2.imread(_split_img[1])
    img2 = ~cv2.imread(_split_img[0])
    org_img = cv2.imread(_img_list[0])
    h, w, _ = org_img.shape
    img1 = cv2.resize(img1, (w,h))
    img2 = cv2.resize(img2, (w,h))
    ocr_data = ut.read_ocr(_ocr_list[0])
    # for item in ocr_data:
    #     cv2.rectangle(org_img, (item[2], item[3]), (item[4], item[5]), (0,255,0), -1)
    final_table = execute_pipeline(img1, img2, ocr_data, org_img)
    visualize_merging(final_table, org_img, "testing.png")