import os
import cv2
import glob
import pickle
import functools


def read_ocr(path):
    with (open(path, "rb")) as openfile:
        ocr_data = pickle.load(openfile)
    return ocr_data


def read_splitout(path):
    return cv2.imread(path)


def compare(w1, w2):
    if max(w1[3], w2[3]) - min(w1[5], w2[5]) > 0.2 * (w1[5] - w1[3]):
        if w1[3] < w2[3]:
            return -1
        elif w1[3] > w2[3]:
            return 1
        else:
            return 0
    else:
        if w1[2] < w2[2]:
            return -1
        elif w1[2] > w2[2]:
            return 1
        else:
            return 0


def ocr_closing(bboxes_table):
    bboxes_table.sort(key=functools.cmp_to_key(compare))
    j = 0
    while j < len(bboxes_table):
        k = j + 1
        overlapping = []
        while k < len(bboxes_table):
            if max(bboxes_table[j][3], bboxes_table[k][3]) > min(
                bboxes_table[j][5], bboxes_table[k][5]
            ):
                break
            else:
                overlapping.append(bboxes_table[k])
            k += 1
        overlapping.sort(key=lambda x: x[2])

        k = 0
        while k < len(overlapping):
            h1 = bboxes_table[j][5] - bboxes_table[j][3]
            h2 = overlapping[k][5] - overlapping[k][3]
            dx = overlapping[k][2] - bboxes_table[j][4]
            if dx < h1:
                bboxes_table[j][1] += " " + overlapping[k][1]
                bboxes_table[j][2] = min(bboxes_table[j][2], overlapping[k][2])
                bboxes_table[j][3] = min(bboxes_table[j][3], overlapping[k][3])
                bboxes_table[j][4] = max(bboxes_table[j][4], overlapping[k][4])
                bboxes_table[j][5] = max(bboxes_table[j][5], overlapping[k][5])
                bboxes_table[j][0] = len(bboxes_table[j][1])
                bboxes_table.remove(overlapping[k])
            k += 1
        j += 1
