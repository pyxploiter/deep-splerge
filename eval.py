import numpy as np
import random
import os
import cv2
import pickle
import glob
import random
import string
import argparse
from xml.dom import minidom
from xml.etree import ElementTree

def get_column_seperators(image, smoothing=10, is_row=True):
    if is_row:
        cols = np.where(np.sum(image, axis=1)!=0)[0]
    else:
        cols = np.where(np.sum(image, axis=0)!=0)[0]

    if len(cols) == 0:
        return []

    adjacent_cols = [cols[0]]
    final_seperators = []
    for i in range(1, len(cols)):
        if cols[i] - cols[i - 1] < smoothing:
            adjacent_cols.append(cols[i])
        elif len(adjacent_cols) > 0:
            final_seperators.append(sum(adjacent_cols) // len(adjacent_cols))
            adjacent_cols = []
    if len(adjacent_cols) > 0:
        final_seperators.append(sum(adjacent_cols) // len(adjacent_cols))

    return final_seperators

def color_code_image(seperators, ocr_mask, is_row=True):
    colors1 = list(range(20))
    colors2 = list(range(90, 110))
    colors3 = list(range(160,180))

    colors = []
    for color_tuple in zip(colors1, colors2, colors3):
        colors += list(color_tuple)

    colors = colors[:len(seperators) + 1]

    color_encoded = np.zeros_like(ocr_mask)

    seperators = [0] + seperators + [ocr_mask.shape[0] if is_row else ocr_mask.shape[1]]
    for i in range(len(seperators) - 1):
        if is_row:
            cv2.rectangle(color_encoded, (0, seperators[i]), (ocr_mask.shape[1], seperators[i + 1]), colors[i], -1)
        else:
            cv2.rectangle(color_encoded, (seperators[i], 0), (seperators[i + 1], ocr_mask.shape[0]), colors[i], -1)

    color_encoded[ocr_mask==0] = 255
    return color_encoded

def rescale_seperators(seperators, org_max, new_max):
    for i in range(len(seperators)):
        seperators[i] = seperators[i] * new_max // org_max
    return seperators

def evaluate_color_encodings(gt_img, prediction, T=0.9):
    gt_colors = np.unique(np.array(gt_img[gt_img!=255]))
    pred_colors = np.unique(np.array(prediction[prediction!=255]))

    c_matrix = np.zeros((len(gt_colors), len(pred_colors)))

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            c_matrix[i, j] += np.count_nonzero((gt_img == gt_colors[i]) & (prediction == pred_colors[j])) / np.count_nonzero(gt_img == gt_colors[i])
    
    _correct = 0
    for i in range(c_matrix.shape[0]):
        if np.count_nonzero(c_matrix[i, :] > T) == 1:
            pred = np.argmax(c_matrix[i, :])
            if np.count_nonzero(c_matrix[:, pred] > 1-T) == 1:
                _correct+=1

    # _correct = np.count_nonzero(np.count_nonzero(c_matrix > T, axis=1) == 1)
    _partial = np.count_nonzero(np.count_nonzero((c_matrix > 1-T), axis=1) != 0) - _correct
    _missed = np.count_nonzero(np.count_nonzero(c_matrix < 1-T, axis=1) == c_matrix.shape[1])
    _false_positives = np.count_nonzero(np.count_nonzero(c_matrix < 1-T, axis=0) == c_matrix.shape[0])
    _over_segmented = np.count_nonzero(np.count_nonzero(c_matrix > 1-T, axis=1) > 1)
    _under_segmented = np.count_nonzero(np.count_nonzero(c_matrix > 1-T, axis=0) > 1)

    return np.array([
        _correct,
        _partial,
        _missed,
        _false_positives,
        _over_segmented,
        _under_segmented
        ])

def eval(images_dir, ocr_dir, gt_dir, results_dir, is_row=True):
    filenames = [name.split('/')[-1].rsplit('.', 1)[0] for name in glob.glob(os.path.join(gt_dir, '*.xml'))]

    metrics = np.zeros(6)
    total_gt = 0
    total_pred = 0
    for filename in filenames:
        print(filename)
        print(os.path.join(images_dir, filename + '.png'))
        if os.path.exists(os.path.join(images_dir, filename + '.png')):
            img = cv2.imread(os.path.join(images_dir, filename + '.png'), cv2.IMREAD_GRAYSCALE)
        else:
            continue
        
        if os.path.exists(os.path.join(ocr_dir, filename + '.pkl')):
            with open(os.path.join(ocr_dir, filename + '.pkl'), "rb") as f:
                ocr = pickle.load(f)
        else:
            continue

        ocr_mask = np.zeros_like(img)
        for word in ocr:
            txt = word[1].translate(str.maketrans('', '', string.punctuation))
            if len(txt.strip()) > 0:
                cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 255, -1)

        tree = ElementTree.parse(os.path.join(gt_dir, filename + '.xml'))
        root = tree.getroot()
        for i, obj in enumerate(root.findall(".//Table")):
            table_name = filename + '_' + str(i)

            columns = []
            rows = []
            rect = [int(obj.attrib["x0"]),
                    int(obj.attrib["y0"]),
                    int(obj.attrib["x1"]),
                    int(obj.attrib["y1"])]

            img_crop = img[rect[1]:rect[3], rect[0]:rect[2]]
            ocr_mask_crop = ocr_mask[rect[1]:rect[3], rect[0]:rect[2]]
            ocr_mask_crop2 = ocr_mask_crop.copy()

            for col in obj.findall(".//Column"):
                columns.append(int(col.attrib['x0']) - rect[0])
            for row in obj.findall(".//Row"):
                rows.append(int(row.attrib['y0']) - rect[1])
            for cell in obj.findall(".//Cell"):
                if cell.attrib['endCol'] != cell.attrib['startCol'] or cell.attrib['endRow'] != cell.attrib['startRow']:
                    x0, y0, x1, y1 = cell.attrib['x0'], cell.attrib['y0'], cell.attrib['x1'] , cell.attrib['y1']

                    cv2.rectangle(ocr_mask_crop2, (int(x0)-rect[0], int(y0)-rect[1]), (int(x1)-rect[0], int(y1)-rect[1]), 0, -1)

            img_crop_masked = img_crop.copy()
            img_crop_masked[ocr_mask_crop == 0] = 255


            if os.path.exists(os.path.join(results_dir, table_name + '.png')):
                pred_image = cv2.imread(os.path.join(results_dir, table_name + '.png'), cv2.IMREAD_GRAYSCALE)
            else:
                continue


            seperators = get_column_seperators(pred_image, is_row=is_row)
            seperators = rescale_seperators(seperators, pred_image.shape[1], img_crop_masked.shape[0] if is_row else img_crop_masked.shape[1])

            gt_color_coded = color_code_image(rows if is_row else columns, ocr_mask_crop2, is_row)
            pred_color_coded = color_code_image(seperators, ocr_mask_crop2, is_row)
            _metrics = evaluate_color_encodings(gt_color_coded, pred_color_coded)
            metrics += _metrics
            total_gt += len(rows if is_row else columns) + 1
            total_pred += len(seperators) + 1

            # if sum(_metrics[:3]) != len(rows if is_row else columns) + 1:
            #     print(_metrics)
            # cv2.imshow("gt", gt_color_coded)
            # cv2.imshow("pred_color_coded", pred_color_coded)
            # cv2.waitKey(0)
    print("Correct: ", round(100 * metrics[0] / sum(metrics[:3]), 2), "%")
    print("Partial: ", round(100 * metrics[1] / sum(metrics[:3]), 2), "%")
    print("Missed: ", round(100 * metrics[2] / sum(metrics[:3]), 2), "%")
    print("False Positives: ", round(100 * metrics[3] / sum(metrics[:3]), 2), "%")
    print("Over Segmented: ", round(100 * metrics[4] / sum(metrics[:3]), 2), "%")
    print("Under Segmented: ", round(100 * metrics[5] / sum(metrics[:3]), 2), "%")
    print(metrics)
    print(total_gt)
    print(total_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results", type=str,
                        help="Directory to results of model", default="output/row_out")

    parser.add_argument("-i", "--images", type=str,
                        help="Directory to original images", default="data/original_images")

    parser.add_argument("-o", "--ocr_dir", type=str,
                        help="Directory to images OCR", default="data/ocr")

    parser.add_argument("-gt", "--ground_truth", type=str,
                        help="Directory to ground truths", default="data/gt")

    parser.add_argument("-m", "--mode", type=str, choices=['row', 'col'], required=True,
                        help="Select the mode in which to run the model")

    args = parser.parse_args()

    eval(args.images, args.ocr_dir, args.ground_truth, args.results, True if args.mode=='row' else False)