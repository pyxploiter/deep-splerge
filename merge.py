import os
import glob
import pickle
import argparse
from xml.dom import minidom
from xml.etree import ElementTree as ET

from termcolor import cprint

import re
import cv2
import numpy as np

import libs.merge_utility as ut
import libs.GTElement as gc


def col_merging(_table, ocr_data):
    for word in ocr_data:
        txt = re.sub('[^a-zA-Z0-9]', '', str(word[1]))
        if len(txt) == 0:
            continue
        for col in _table.gtCols:
            if (word[2] < col.x1) and (word[4] > col.x2):
                merging_col = gc.Row(word[2], word[3], word[4])
                _table.gtSpans.append(merging_col)
    _table.evaluateCells()


def row_merging(_table, ocr_data):
    for word in ocr_data:
        txt = re.sub('[^a-zA-Z0-9]', '', str(word[1]))
        if len(txt) == 0:
            continue
        for row in _table.gtRows:
            if (word[3] < row.y1) and (word[5] > row.y2):
                merging_row = gc.Column(word[2], word[3], word[5])
                _table.gtSpans.append(merging_row)
    _table.evaluateCells()


def remove_redundant_seperators(_table, ocr_data, check_threshold=0.4):
    _table.populate_ocr(ocr_data)

    remove_rows = [True for i in range(len(_table.gtRows))]
    remove_cols = [True for i in range(len(_table.gtCols))]
    row_threshold_count = [0 for i in range(len(_table.gtRows))]
    col_threshold_count = [0 for i in range(len(_table.gtCols))]

    for i in range(1, len(_table.gtCells)):
        for j in range(len(_table.gtCells[i])):
            if (
                _table.gtCells[i][j].dontCare == False
                and len(_table.gtCells[i][j].words) > 0
            ):
                if len(_table.gtCells[0]) < 4:
                    current_thresh = 1.0
                else:
                    row_threshold_count[i - 1] += 1
                    current_thresh = row_threshold_count[i -
                                                         1] / len(_table.gtCells[i])
                if current_thresh >= check_threshold:
                    remove_rows[i - 1] = False
                    break

    for j in range(1, len(_table.gtCells[0])):
        for i in range(len(_table.gtCells)):
            if (
                _table.gtCells[i][j].dontCare == False
                and len(_table.gtCells[i][j].words) > 0
            ):
                if len(_table.gtCells) < 5:
                    current_thresh = 1.0
                else:
                    col_threshold_count[j - 1] += 1
                    current_thresh = col_threshold_count[j -
                                                         1] / len(_table.gtCells)
                if current_thresh >= check_threshold:
                    remove_cols[j - 1] = False
                    break

    removed_rows = [
        _table.gtRows[i].y1 for i in range(len(_table.gtRows)) if remove_rows[i]
    ]
    removed_cols = [
        _table.gtCols[i].x1 for i in range(len(_table.gtCols)) if remove_cols[i]
    ]

    _table.gtRows = [
        _table.gtRows[i] for i in range(len(_table.gtRows)) if not remove_rows[i]
    ]
    _table.gtCols = [
        _table.gtCols[i] for i in range(len(_table.gtCols)) if not remove_cols[i]
    ]
    _table.evaluateCells()
    return removed_rows, removed_cols


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


def get_grid_structure(xml_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    tables = root.findall(".//Table")
    
    assert len(tables) == 1

    w = int(tables[0].attrib["x1"])
    h = int(tables[0].attrib["y1"])

    columns = list(map(lambda x: int(x.attrib["x0"]), tables[0].findall(".//Column")))
    rows = list(map(lambda x: int(x.attrib["y0"]), tables[0].findall(".//Row")))

    return rows, columns, h, w


def execute_pipeline(xml_filename, ocr_data, org_image):
    _list_rows, _list_col, h, w = get_grid_structure(xml_filename)
    
    if org_image is not None:
        org_image[_list_rows, :] = (0, 255, 0)
        org_image[:, _list_col] = (0, 255, 0)

    _table = get_table(_list_rows, _list_col, h, w)

    col_merging(_table, ocr_data)
    row_merging(_table, ocr_data)

    removed_rows, removed_cols = remove_redundant_seperators(
        _table, ocr_data, 0.2)

    removed_rows_iter_2, removed_cols_iter_2 = remove_redundant_seperators(
        _table, ocr_data, 0.4)

    removed_rows.extend(removed_rows_iter_2)
    removed_cols.extend(removed_cols_iter_2)

    if org_image is not None:
        org_image[removed_rows, :] = (0, 0, 255)
        org_image[:, removed_cols] = (0, 0, 255)

    _table.merge_header_v2(ocr_data)
    return _table


def visualize_merging(_table, ocr_data, img, image_name):
    for item in _table.gtSpans:
        cv2.line(img, (item.x1, item.y1),
                 (item.x2 - 1, item.y2 - 1), (0, 0, 255), 2)
    for item in ocr_data:
        cv2.rectangle(img, (item[2], item[3]),
                      (item[4], item[5]), (245, 66, 176), 1)
    cv2.imwrite(image_name, img)

def data_pipeline(xml_in_path, out_path, images_path, ocr_path):
    filenames = map(os.path.basename, glob.glob(os.path.join(xml_in_path, "*.xml")))

    for item in filenames:
        try:
            filename = item.rsplit(".", 1)[0]

            xml_filename = os.path.join(xml_in_path, filename + ".xml")
            if images_path:
                org_img = cv2.imread(os.path.join(images_path, filename + ".png"))
            else:
                org_img = None
            
            with (open(os.path.join(ocr_path, filename + ".pkl"), "rb")) as f:
                ocr_data = pickle.load(f)
            ocr_data = ut.clean_ocr_data(ocr_data)
            ocr_data = ut.make_sentences(ocr_data)

            final_table = execute_pipeline(xml_filename, ocr_data, org_img)

            out_root = ET.Element("GroundTruth")
            out_root.attrib["InputFile"] = filename + ".png"
            out_tables = ET.SubElement(out_root, "Tables")
            table_xml = final_table.get_xml_object()
            out_tables.append(table_xml)
            out_data = minidom.parseString(
                ET.tostring(out_root)).toprettyxml(indent="    ")

            with open(os.path.join(out_path, "xmls", filename + ".xml"), "w") as _file:
                _file.write("\n".join(out_data.split("\n")))

            if org_img is not None:
                visualize_merging(
                    final_table,
                    ocr_data,
                    org_img,
                    os.path.join(out_path, "visualization", filename + ".png"),
                )

            cprint("Processed: ", "green", attrs=["bold"], end="")
            print(filename)
        except Exception as e:
            cprint("Error: ", "red", attrs=["bold"], end="")
            print(filename)
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_xml_dir",
        help="Path to folder containing XML files predicted by infer.py",
        required=True,
    )    
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to folder for writing output XML files and visualization (optional) of merge heuristics.",
        required=True,
    )
    parser.add_argument(
        "-ocr",
        "--ocr_dir",
        help="Path to folder containing table-level OCR files generated by prepare_data.py",
        required=True,
    )
    parser.add_argument(
        "-img",
        "--images_dir",
        help="Path to table-level images generated by prepare_data.py (Optional. If not provided merge visualization will not be written).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "xmls"), exist_ok=True)
    if args.images_dir:
        os.makedirs(os.path.join(args.output_dir, "visualization"), exist_ok=True)

    data_pipeline(args.input_xml_dir, args.output_dir, args.images_dir, args.ocr_dir)
