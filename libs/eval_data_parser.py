import os
import glob
import pickle
import string
import functools

import cv2
import PIL
import numpy as np
import pytesseract
from xml.etree import ElementTree as ET


class GenerateTFRecord:
    def __init__(self, imagespath, ocrpath, xmlpath, outxmlpath):
        self.unlvocrpath = ocrpath
        self.unlvimagespath = imagespath
        self.unlvtablepath = xmlpath
        self.outtablepath = outxmlpath
        self.visualizeimgs = False

        self.num_of_max_vertices = 900
        self.max_length_of_word = 30

        self.num_data_dims = 5
        self.max_height = 1024
        self.max_width = 1024

        self.counter = 0
        self.tmp_unlv_tables = None
        self.xmlfilepaths = glob.glob(os.path.join(self.unlvtablepath, "*.xml"))

    def str_to_int(self, str):
        intsarr = np.array([ord(chr) for chr in str])
        padded_arr = np.zeros(shape=(self.max_length_of_word), dtype=np.int64)
        padded_arr[: len(intsarr)] = intsarr
        return padded_arr

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def pad_with_zeros(self, arr, shape):
        dummy = np.zeros(shape, dtype=np.int64)
        dummy[: arr.shape[0], : arr.shape[1]] = arr
        return dummy

    def generate_tf_record(
        self,
        im,
        gt_matrices,
        pred_matrices,
        arr,
        tablecategory,
        imgindex,
        output_file_name,
    ):
        """This function generates tfrecord files using given information"""
        gt_matrices = [
            self.pad_with_zeros(
                matrix, (self.num_of_max_vertices, self.num_of_max_vertices)
            ).astype(np.int64)
            for matrix in gt_matrices
        ]
        pred_matrices = [
            self.pad_with_zeros(
                matrix, (self.num_of_max_vertices, self.num_of_max_vertices)
            ).astype(np.int64)
            for matrix in pred_matrices
        ]

        im = im.astype(np.int64)
        img_height, img_width = im.shape

        words_arr = arr[:, 1].tolist()
        no_of_words = len(words_arr)

        lengths_arr = self.convert_to_int(arr[:, 0])
        vertex_features = np.zeros(
            shape=(self.num_of_max_vertices, self.num_data_dims), dtype=np.int64
        )
        lengths_arr = np.array(lengths_arr).reshape(len(lengths_arr), -1)
        sample_out = np.array(np.concatenate((arr[:, 2:], lengths_arr), axis=1))
        vertex_features[:no_of_words, :] = sample_out

        vertex_text = np.zeros(
            (self.num_of_max_vertices, self.max_length_of_word), dtype=np.int64
        )
        vertex_text[:no_of_words] = np.array(list(map(self.str_to_int, words_arr)))

        result = {
            "image": im.astype(np.float32),
            "sampled_ground_truths": gt_matrices,
            "sampled_predictions": pred_matrices,
            "sampled_indices": None,
            "global_features": np.array(
                [img_height, img_width, no_of_words, tablecategory]
            ).astype(np.float32),
            "vertex_features": vertex_features.astype(np.float32),
        }

        return result

    @staticmethod
    def apply_ocr(path, image):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            w, h = image.size
            r = 2500 / w
            image = image.resize((2500, int(r * h)))

            print("OCR file not found: ", path, "\t...Applying OCR")
            ocr = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config="--oem 1"
            )

            bboxes = []
            for i in range(len(ocr["conf"])):
                if ocr["level"][i] > 4 and ocr["text"][i].strip() != "":
                    bboxes.append(
                        [
                            len(ocr["text"][i]),
                            ocr["text"][i],
                            int(ocr["left"][i] / r),
                            int(ocr["top"][i] / r),
                            int(ocr["left"][i] / r) + int(ocr["width"][i] / r),
                            int(ocr["top"][i] / r) + int(ocr["height"][i] / r),
                        ]
                    )

            bboxes = sorted(
                bboxes,
                key=lambda box: (box[4] - box[2]) * (box[5] - box[3]),
                reverse=True,
            )
            threshold = np.average(
                [
                    (box[4] - box[2]) * (box[5] - box[3])
                    for box in bboxes[len(bboxes) // 20 : -len(bboxes) // 4]
                ]
            )
            bboxes = [
                box
                for box in bboxes
                if (box[4] - box[2]) * (box[5] - box[3]) < threshold * 30
            ]

            with open(path, "wb") as f:
                pickle.dump(bboxes, f)

            return bboxes

    def create_same_matrix(self, arr, ids):
        """Given a list of lists with each list consisting of all ids considered same, this function
         generates a matrix """
        matrix = np.zeros(shape=(ids, ids))
        for subarr in arr:
            for element in subarr:
                matrix[element, subarr] = 1
        return matrix

    def data_generator(self):
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

        for counter, filename in enumerate(self.xmlfilepaths):
            print("[", counter, "/", len(self.xmlfilepaths), "] Processing:", filename)
            filename = ".".join(filename.split("/")[-1].split(".")[:-1])
            if not os.path.exists(os.path.join(self.unlvtablepath, filename + ".xml")):
                print("WARNING: Ground truth not found for image ", filename)
                continue
            tree = ET.parse(os.path.join(self.unlvtablepath, filename + ".xml"))
            root = tree.getroot()
            xml_tables = root.findall(".//Table")
            if os.path.exists(os.path.join(self.unlvimagespath, filename + ".png")):
                im = PIL.Image.open(
                    os.path.join(self.unlvimagespath, filename + ".png")
                ).convert("RGB")
            else:
                continue

            bboxes = GenerateTFRecord.apply_ocr(
                os.path.join(self.unlvocrpath, filename + ".pkl"), im.copy()
            )

            for i, obj in enumerate(xml_tables):
                x0 = int(obj.attrib["x0"])
                y0 = int(obj.attrib["y0"])
                x1 = int(obj.attrib["x1"])
                y1 = int(obj.attrib["y1"])
                im2 = im.crop((x0, y0, x1, y1))

                bboxes_table = []
                for box in bboxes:
                    coords = box[2:]
                    intrsct = [
                        max(coords[0], x0),
                        max(coords[1], y0),
                        min(coords[2], x1),
                        min(coords[3], y1),
                    ]
                    w = intrsct[2] - intrsct[0]
                    h = intrsct[3] - intrsct[1]

                    w2 = coords[2] - coords[0]
                    h2 = coords[3] - coords[1]
                    if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                        box = list(box)
                        text = box[1]
                        text = text.translate(
                            str.maketrans("", "", string.punctuation)
                        ).strip()

                        if len(text) == 0:
                            continue

                        if len(box[1]) > self.max_length_of_word:
                            box[1] = box[1][: self.max_length_of_word]
                        bboxes_table.append(box)
                bboxes = [box for box in bboxes if box not in bboxes_table]

                bboxes_table.sort(key=functools.cmp_to_key(compare))

                if len(bboxes_table) > self.num_of_max_vertices:
                    print(
                        "\n\nWARNING: Number of vertices (",
                        len(bboxes_table),
                        ")is greater than limit (",
                        self.num_of_max_vertices,
                        ").\n\n",
                    )
                    bboxes_table = bboxes_table[: self.num_of_max_vertices]

                same_cell_boxes = [[] for _ in range(len(obj.findall(".//Cell")))]
                same_row_boxes = [[] for _ in range(len(obj.findall(".//Row")) + 1)]
                same_col_boxes = [[] for _ in range(len(obj.findall(".//Column")) + 1)]

                for idx, cell in enumerate(obj.findall(".//Cell")):
                    if cell.attrib["dontCare"] == "true":
                        continue

                    _x0 = int(cell.attrib["x0"])
                    _y0 = int(cell.attrib["y0"])
                    _x1 = int(cell.attrib["x1"])
                    _y1 = int(cell.attrib["y1"])
                    for idx2, box in enumerate(bboxes_table):
                        coords = box[2:]

                        intrsct = [
                            max(coords[0], _x0),
                            max(coords[1], _y0),
                            min(coords[2], _x1),
                            min(coords[3], _y1),
                        ]
                        w = intrsct[2] - intrsct[0]
                        h = intrsct[3] - intrsct[1]

                        w2 = coords[2] - coords[0]
                        h2 = coords[3] - coords[1]
                        if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                            same_cell_boxes[idx].append(idx2)

                    for j in range(
                        int(cell.attrib["startCol"]), int(cell.attrib["endCol"]) + 1
                    ):
                        same_col_boxes[j] += same_cell_boxes[idx]
                    for j in range(
                        int(cell.attrib["startRow"]), int(cell.attrib["endRow"]) + 1
                    ):
                        same_row_boxes[j] += same_cell_boxes[idx]

                gt_matrices = [
                    self.create_same_matrix(same_cell_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_row_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_col_boxes, len(bboxes_table)),
                ]

                table_name = os.path.join(
                    self.outtablepath, filename + "_" + str(i) + ".xml"
                )
                if not os.path.exists(table_name):
                    print('\nERROR: "', table_name, '" not found.')
                    continue
                root_pred = ET.parse(os.path.join(table_name)).getroot()
                table_pred = root_pred.findall(".//Table")[0]

                same_cell_boxes = [
                    [] for _ in range(len(table_pred.findall(".//Cell")))
                ]
                same_row_boxes = [
                    [] for _ in range(len(table_pred.findall(".//Row")) + 1)
                ]
                same_col_boxes = [
                    [] for _ in range(len(table_pred.findall(".//Column")) + 1)
                ]

                for idx, cell in enumerate(table_pred.findall(".//Cell")):
                    if cell.attrib["dontCare"] == "true":
                        continue

                    _x0 = int(cell.attrib["x0"]) + x0
                    _y0 = int(cell.attrib["y0"]) + y0
                    _x1 = int(cell.attrib["x1"]) + x0
                    _y1 = int(cell.attrib["y1"]) + y0
                    for idx2, box in enumerate(bboxes_table):
                        coords = box[2:]

                        intrsct = [
                            max(coords[0], _x0),
                            max(coords[1], _y0),
                            min(coords[2], _x1),
                            min(coords[3], _y1),
                        ]
                        w = intrsct[2] - intrsct[0]
                        h = intrsct[3] - intrsct[1]

                        w2 = coords[2] - coords[0]
                        h2 = coords[3] - coords[1]
                        if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                            same_cell_boxes[idx].append(idx2)

                    for j in range(
                        int(cell.attrib["startCol"]), int(cell.attrib["endCol"]) + 1
                    ):
                        same_col_boxes[j] += same_cell_boxes[idx]
                    for j in range(
                        int(cell.attrib["startRow"]), int(cell.attrib["endRow"]) + 1
                    ):
                        same_row_boxes[j] += same_cell_boxes[idx]

                pred_matrices = [
                    self.create_same_matrix(same_cell_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_row_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_col_boxes, len(bboxes_table)),
                ]

                w_org, h_org = im2.size
                h, w = self.max_height, self.max_width

                if im2.size[0] < 20 or im2.size[1] < 20:
                    continue

                im2 = im2.resize(
                    (im2.size[0] * 2500 // im.size[0], im2.size[1] * 2500 // im.size[0])
                )

                if im2.size[0] > w:
                    im2 = im2.resize((w, im2.size[1] * w // im2.size[0]))
                if im2.size[1] > h:
                    im2 = im2.resize((im2.size[0] * h // im2.size[1], h))

                w_new, h_new = im2.size

                new_im = im2
                # new_im.paste(im2)

                r = w_org / h_org

                for j in range(len(bboxes_table)):
                    bboxes_table[j][2] -= x0
                    bboxes_table[j][4] -= x0
                    bboxes_table[j][2] = bboxes_table[j][2] * w_new // w_org
                    bboxes_table[j][4] = bboxes_table[j][4] * w_new // w_org

                    bboxes_table[j][3] -= y0
                    bboxes_table[j][5] -= y0
                    bboxes_table[j][3] = bboxes_table[j][3] * h_new // h_org
                    bboxes_table[j][5] = bboxes_table[j][5] * h_new // h_org

                if len(bboxes_table) == 0:
                    print(
                        "WARNING: No word boxes found inside table #",
                        i,
                        " in image ",
                        filename,
                    )
                    continue

                img = np.asarray(new_im, np.int64)[:, :, 0]

                gt_matrices = [
                    np.array(matrix, dtype=np.int64) for matrix in gt_matrices
                ]
                pred_matrices = [
                    np.array(matrix, dtype=np.int64) for matrix in pred_matrices
                ]

                yield self.generate_tf_record(
                    img,
                    gt_matrices,
                    pred_matrices,
                    np.array(bboxes_table),
                    0,
                    counter,
                    "_",
                )

