import os
import shutil
import glob
import argparse
import pickle
import numpy as np
from xml.dom import minidom
from xml.etree import ElementTree
from tqdm import tqdm
import cv2
import string

def process_files(image_dir, xml_dir, ocr_dir, out_dir):
    files = [file.split('/')[-1].rsplit('.', 1)[0] for file in glob.glob(os.path.join(xml_dir,'*.xml'))]
    files.sort()

    col_merge_counter = 0
    row_merge_counter = 0

    for ii, file in enumerate(files):

        image_file = os.path.join(image_dir, file + '.png')
        xml_file = os.path.join(xml_dir, file + '.xml')
        ocr_file = os.path.join(ocr_dir, file + '.pkl')

        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        with open(ocr_file, "rb") as f:
            ocr = pickle.load(f)

        ocr_mask = np.zeros_like(img)
        for word in ocr:
            txt = word[1].translate(str.maketrans('', '', string.punctuation))
            if len(txt.strip()) > 0:
                cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 255, -1)

        if os.path.exists(image_file) and os.path.exists(xml_file) and os.path.exists(ocr_file):
            print(ii, len(files), "Processing: ", file)
            tree = ElementTree.parse(xml_file)
            root = tree.getroot()
            for i, obj in enumerate(root.findall(".//Table")):
                table_name = file + '_' + str(i)

                columns = []
                rows = []
                rect = [int(obj.attrib["x0"]),
                        int(obj.attrib["y0"]),
                        int(obj.attrib["x1"]),
                        int(obj.attrib["y1"])]

                img_crop = img[rect[1]:rect[3], rect[0]:rect[2]]
                ocr_mask_crop = ocr_mask[rect[1]:rect[3], rect[0]:rect[2]]
                ocr_mask_crop2 = ocr_mask_crop.copy()

                col_spans = []
                row_spans = []
                for col in obj.findall(".//Column"):
                    columns.append(int(col.attrib['x0']) - rect[0])
                for row in obj.findall(".//Row"):
                    rows.append(int(row.attrib['y0']) - rect[1])
                flag = False
                for cell in obj.findall(".//Cell"):
                    if cell.attrib['endCol'] != cell.attrib['startCol'] or cell.attrib['endRow'] != cell.attrib['startRow']:
                        x0, y0, x1, y1 = map(int, [cell.attrib['x0'], cell.attrib['y0'], cell.attrib['x1'], cell.attrib['y1']])
                        x0 -= rect[0] - 10
                        y0 -= rect[1] - 10
                        x1 -= rect[0] + 10
                        y1 -= rect[1] + 10
                        
                        cell_mask = ocr_mask[y0:y1, x0:x1]
                        row_mask = ocr_mask[y0:y1, :]
                        col_mask = ocr_mask[:, x0:x1]

                        indices = np.where(cell_mask !=0)
                        row_indices = np.where(row_mask !=0)
                        col_indices = np.where(col_mask !=0)

                        if len(indices[0]) != 0:
                            x_min = np.amin(indices[1]) + x0
                            x_max = np.amax(indices[1]) + x0
                            y_min = np.amin(indices[0]) + y0
                            y_max = np.amax(indices[0]) + y0

                            flag = True
                            if cell.attrib['endCol'] != cell.attrib['startCol']:
                                col_spans.append((
                                    np.amin(col_indices[1]) + x0, 
                                    np.amin(indices[0]) + y0, 
                                    np.amax(col_indices[1]) + x0, 
                                    np.amax(indices[0]) + y0
                                ))
                                col_merge_counter += 1

                            if cell.attrib['endRow'] != cell.attrib['startRow']:
                                row_spans.append((
                                    np.amin(indices[1]) + x0, 
                                    np.amin(row_indices[0]) + y0, 
                                    np.amax(indices[1]) + x0, 
                                    np.amax(row_indices[0]) + y0, 
                                ))
                                row_merge_counter += 1

                        cv2.rectangle(ocr_mask_crop2, (x0, y0), (x1, y1), 0, -1)

                # if flag:
                #     continue



                # bboxes_table = []
                # for box in ocr:
                #     coords = box[2:]
                #     intrsct = [
                #                 max(coords[0], rect[0]), 
                #                 max(coords[1], rect[1]), 
                #                 min(coords[2], rect[2]), 
                #                 min(coords[3], rect[3])
                #                 ]
                #     w = intrsct[2] - intrsct[0]
                #     h = intrsct[3] - intrsct[1] 

                #     w2 = coords[2] - coords[0]
                #     h2 = coords[3] - coords[1]
                #     if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                #         box = list(box)
                #         if len(box[1]) > self.max_length_of_word:
                #             box[1] = box[1][:self.max_length_of_word]
                #         bboxes_table.append(box)
                # ocr = [box for box in ocr if box not in bboxes_table]

                img_crop_masked = img_crop.copy()
                img_crop_masked[ocr_mask_crop == 0] = 255

                col_gt_mask = np.zeros_like(img_crop)
                row_gt_mask = np.zeros_like(img_crop)

                non_zero_rows = np.append(np.where(np.count_nonzero(ocr_mask_crop2, axis=1)!=0)[0], [0, img_crop.shape[0]])
                non_zero_cols = np.append(np.where(np.count_nonzero(ocr_mask_crop2, axis=0)!=0)[0], [0, img_crop.shape[1]])

                for col in columns:
                    if col == 0 or col == img_crop.shape[1]:
                        continue
                    diff = non_zero_cols - col
                    left = min(-diff[diff < 0]) + 1
                    right = min(diff[diff > 0])
                    col_gt_mask[:, col - left: col + right] = 255

                for row in rows:
                    if row == 0 or row == img_crop.shape[0]:
                        continue
                    diff = non_zero_rows - row
                    above = min(-diff[diff < 0]) + 1
                    below = min(diff[diff > 0])
                    row_gt_mask[row - above: row + below, :] = 255

                cv2.imwrite(os.path.join(out_dir, 'org_images', table_name + '.png'), img_crop)
                # cv2.imwrite(os.path.join(out_dir, 'masked_images', table_name + '.png'), img_crop_masked)
                # with open(os.path.join(out_dir, 'gt', table_name + '.pkl'), 'wb') as f:
                #     pickle.dump({'row_gt': row_gt_mask[:, 0], 'col_gt': col_gt_mask[0, :]}, f)

                # cv2.imwrite(os.path.join("out_imgs", table_name + '_row.png'), row_gt_mask)
                # cv2.imwrite(os.path.join("out_imgs", table_name + '_col.png'), col_gt_mask)

                with open(os.path.join(out_dir, "labels", table_name + '_row.txt'), 'w') as f:
                    for i in row_gt_mask[:, 0]:
                        f.write(str(i)+"\n")

                with open(os.path.join(out_dir, "labels", table_name + '_col.txt'), 'w') as f:
                    for i in col_gt_mask[0, :]:
                        f.write(str(i)+"\n")

                with open(os.path.join(out_dir, "merges", table_name + '.pkl'), 'wb') as f:
                    pickle.dump({"row": row_spans, "col": col_spans}, f)

                    # pickle.dump({'row_gt': row_gt_mask[:, 0], 'col_gt': col_gt_mask[0, :]}, f)

                # cv2.imwrite(os.path.join(out_dir, 'gt_row', table_name + '.png'), np.repeat(row_gt_mask[:,0][:, np.newaxis], img_crop.shape[1], axis=1))
                # cv2.imwrite(os.path.join(out_dir, 'gt_col', table_name + '.png'), np.repeat(col_gt_mask[0,:][np.newaxis, :], img_crop.shape[0], axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--image_dir", type=str,
                        help="Directory for images", default="data/augmented/images")

    parser.add_argument("-xml", "--xml_dir", type=str,
                        help="Directory for xmls", default="data/augmented/gt")

    parser.add_argument("-ocr", "--ocr_dir", type=str,
                        help="Directory for ocr files", default="data/augmented/ocr")

    parser.add_argument("-o", "--out_dir", type=str,
                        help="Output directory for generated data", default="data/processed")
                        
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'org_images'), exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir,'masked_images'), exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir,'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'labels'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'merges'), exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir,'gt_col'), exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir,'gt_row'), exist_ok=True)

    process_files(args.image_dir, args.xml_dir, args.ocr_dir, args.out_dir)
