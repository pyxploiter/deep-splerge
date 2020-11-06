import os
import re
import cv2
import glob
import pickle
import functools
import numpy as np


def clean_ocr_data(ocr_data):
    clean_ocr_data = []
    for item in ocr_data:
        txt = item[1]
        txt = str(txt).replace("|", "")
        txt = str(txt).replace("_", "")
        if len(txt) == 0:
            continue
        else:
            clean_ocr_data.append(item)
    return clean_ocr_data


def compare(w1, w2):
    if max(w1[3], w2[3]) - min(w1[5], w2[5]) > 0. * (w1[5] - w1[3]):
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


def make_sentences(ocr_data):
    ocr_data = ocr_data.copy()
    ocr_data.sort(key=functools.cmp_to_key(compare))
    j = 0
    while j < len(ocr_data):
        k = j + 1
        overlapping = []
        while k < len(ocr_data):
            if max(ocr_data[j][3], ocr_data[k][3]) > min(
                ocr_data[j][5], ocr_data[k][5]
            ):
                break
            else:
                overlapping.append(ocr_data[k])
            k += 1
        overlapping.sort(key=lambda x: x[2])

        k = 0
        while k < len(overlapping):
            h1 = ocr_data[j][5] - ocr_data[j][3]
            h2 = overlapping[k][5] - overlapping[k][3]
            dx = overlapping[k][2] - ocr_data[j][4]
            if dx < h1 * 0.75 and dx > 0:
                ocr_data[j][1] += " " + overlapping[k][1]
                ocr_data[j][2] = min(ocr_data[j][2], overlapping[k][2])
                ocr_data[j][3] = min(ocr_data[j][3], overlapping[k][3])
                ocr_data[j][4] = max(ocr_data[j][4], overlapping[k][4])
                ocr_data[j][5] = max(ocr_data[j][5], overlapping[k][5])
                ocr_data[j][0] = len(ocr_data[j][1])
                ocr_data.remove(overlapping[k])
            k += 1
        j += 1
    return ocr_data