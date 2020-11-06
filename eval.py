import os
import csv
import pickle
import argparse

import cv2
import gzip
import numpy as np
from tqdm import tqdm
import networkx as nx

from libs.Rect import Rect
from libs.eval_data_parser import GenerateTFRecord


class Block(Rect):
    def __init__(self, x0=0, y0=0, x1=1, y1=1, w_ids=None, label=0, cells=None):
        super().__init__(x0, y0, x1, y1)
        self.label = -1
        self.w_ids = w_ids
        self.cells = cells

    def copy(self):
        return Block(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            w_ids=self.w_ids.copy(),
            label=self.label,
            cells=self.cells,
        )


class InferenceOutputEvaluator:
    def __init__(self, image_path, ocr_path, gt_path, pred_path, output_path):
        self.data_generator = GenerateTFRecord(
            image_path, ocr_path, gt_path, pred_path,
        ).data_generator()

        self._output_path = output_path
        self._output_images2 = os.path.join(output_path, "visualization")
        self._all_files = []
        self.metrics = {
            "cell": {"correct": 0, "missed": 0, "incorrect": 0},
            "row": {"correct": 0, "missed": 0, "incorrect": 0},
            "col": {"correct": 0, "missed": 0, "incorrect": 0},
        }

        if not os.path.exists(self._output_path):
            os.mkdir(self._output_path)
        if not os.path.exists(self._output_images2):
            os.mkdir(self._output_images2)

    def _draw_skeletal(self, img, blocks, is_row=False):
        """Used for Drawing lines along a column or row
            ARGUMENTS:
                img: contains the image that will be drawn upon
                blocks: contains the list of rows/columns
                bool is_row: True if skeletal of rows is to be drawn, False if column.
            RETURN:
                an operating function, returns nothing, performs the operation in place.
        """

        if is_row:
            color = (0, 200, 0)
        else:
            color = (255, 0, 0)

        for i in range(len(blocks)):
            if is_row:
                blocks[i].cells.sort(key=lambda x: x.x1)
            else:
                blocks[i].cells.sort(key=lambda x: x.y1)
            cells = blocks[i].cells
            label = blocks[i].label
            for j in range(1, len(cells)):
                c1 = cells[j - 1]
                c2 = cells[j]

                x1, y1 = (c1.x1 + c1.x2) // 2, (c1.y1 + c1.y2) // 2
                x2, y2 = (c2.x1 + c2.x2) // 2, (c2.y1 + c2.y2) // 2

                if is_row:
                    cv2.circle(img, (x1, y1), 1, color, 15)
                    cv2.circle(img, (x2, y2), 1, color, 15)
                else:
                    cv2.circle(img, (x1, y1), 1, color, 9)
                    cv2.circle(img, (x2, y2), 1, color, 9)

                if not label:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

    def _convert_to_blocks(self, adj):
        """creates a networkx graph and finds the maximum cliques.
            ARGUMENTS:
                adj: takes an adjacency matrix of relations between words
            RETURN:
                blocks: a list of lists containing the row, column relationships between words
        """
        G = nx.Graph(adj)
        blocks = nx.find_cliques(G)
        to_return = []
        for block in blocks:
            rect = [10000, 10000, 0, 0]
            for w_id in block:
                rect[0] = min(rect[0], self.vertex_features[w_id][0])
                rect[1] = min(rect[1], self.vertex_features[w_id][1])
                rect[2] = max(rect[2], self.vertex_features[w_id][2])
                rect[3] = max(rect[3], self.vertex_features[w_id][3])
            to_return.append(Block(rect[0], rect[1], rect[2], rect[3], w_ids=block))

        return to_return

    def evaluate(self):
        self.idx = 0
        for sample in self.data_generator:
            self.process_sample(sample)
            self.idx += 1

        with open(os.path.join(self._output_path, "evaluation.csv"), "w") as f:
            csv_writer = csv.writer(f, delimiter=",")

            csv_writer.writerow([""] + [key for key in self.metrics])
            csv_writer.writerow(
                ["True Positives"]
                + [self.metrics[key]["correct"] for key in self.metrics]
            )
            csv_writer.writerow(
                ["False Negatives"]
                + [self.metrics[key]["missed"] for key in self.metrics]
            )
            csv_writer.writerow(
                ["False Positives"]
                + [self.metrics[key]["incorrect"] for key in self.metrics]
            )

            csv_writer.writerow(
                ["Precision"]
                + [
                    str(
                        round(
                            self.metrics[key]["correct"]
                            * 100
                            / max(
                                1,
                                self.metrics[key]["correct"]
                                + self.metrics[key]["incorrect"],
                            ),
                            2,
                        )
                    )
                    + "%"
                    for key in self.metrics
                ]
            )
            csv_writer.writerow(
                ["Recall"]
                + [
                    str(
                        round(
                            self.metrics[key]["correct"]
                            * 100
                            / max(
                                1,
                                self.metrics[key]["correct"]
                                + self.metrics[key]["missed"],
                            ),
                            2,
                        )
                    )
                    + "%"
                    for key in self.metrics
                ]
            )

    def evaluate_blocks(self, blocks_pred_all, blocks_gt_all):
        for adj_name in ["cell", "row", "col"]:
            blocks_gt = blocks_gt_all[adj_name]
            blocks_pred = blocks_pred_all[adj_name]

            overlap_matrix = np.zeros((len(blocks_gt), len(blocks_pred)))

            for i, block in enumerate(blocks_gt):
                block.w_ids = set(block.w_ids)
                for j, block_pred in enumerate(blocks_pred):
                    block_pred.w_ids = set(block_pred.w_ids)
                    overlap_matrix[i, j] = len(
                        block.w_ids.intersection(block_pred.w_ids)
                    ) / max(1, len(block.w_ids.union(block_pred.w_ids)))

            correct = 0

            for i, block in enumerate(blocks_pred):

                rect = [10000, 10000, 0, 0]
                for w_id in block.w_ids:
                    x1 = self.vertex_features[w_id][0]
                    y1 = self.vertex_features[w_id][1]
                    x2 = self.vertex_features[w_id][2]
                    y2 = self.vertex_features[w_id][3]
                    rect[0] = min(rect[0], x1)
                    rect[1] = min(rect[1], y1)
                    rect[2] = max(rect[2], x2)
                    rect[3] = max(rect[3], y2)

                if np.count_nonzero(overlap_matrix[:, i] == 1) > 0:
                    self.metrics[adj_name]["correct"] += 1
                    correct += 1
                    blocks_pred[i].label = 1
                else:
                    self.metrics[adj_name]["incorrect"] += 1
                    blocks_pred[i].label = 0

            self.metrics[adj_name]["missed"] += len(blocks_gt) - correct

    def recalculate_cells(self, blocks):
        cells = []
        columns = [col.copy() for col in blocks["col"]]
        rows = [row.copy() for row in blocks["row"]]

        for i in range(len(columns)):
            if columns[i].cells is None:
                columns[i].cells = []
        for j in range(len(rows)):
            if rows[j].cells is None:
                rows[j].cells = []
        for i, col in enumerate(blocks["col"]):
            for j, row in enumerate(blocks["row"]):
                cell = col.w_ids.intersection(row.w_ids)
                if len(cell) == 0:
                    continue
                x1, y1, x2, y2 = 10000, 10000, 0, 0
                for wid in cell:
                    x1 = min(x1, int(self.vertex_features[wid][0]))
                    y1 = min(y1, int(self.vertex_features[wid][1]))
                    x2 = max(x2, int(self.vertex_features[wid][2]))
                    y2 = max(y2, int(self.vertex_features[wid][3]))

                cell = Block(x1, y1, x2, y2, w_ids=cell)
                if cell not in cells:
                    cells.append(cell)
                columns[i].cells.append(cell)
                rows[j].cells.append(cell)

        return cells, columns, rows

    def process_sample(self, sample):
        image = sample["image"]
        sampled_ground_truths = sample["sampled_ground_truths"]
        sampled_predictions = sample["sampled_predictions"]

        h, w, n_words, _ = sample["global_features"]
        h, w, n_words = int(h), int(w), int(n_words)

        self.vertex_features = sample["vertex_features"]

        sampled_ground_truths = [
            arr[:n_words, :n_words] for arr in sampled_ground_truths
        ]
        sampled_predictions = [arr[:n_words, :n_words] for arr in sampled_predictions]

        cv2.imwrite(os.path.join(self._output_images2, str(self.idx) + ".png"), image)

        blocks_pred = {"row": None, "col": None, "cell": None}
        blocks_gt = {"row": None, "col": None, "cell": None}

        for adj_name, predictions, gt in zip(
            ["cell", "row", "col"], sampled_predictions, sampled_ground_truths
        ):
            predictions = np.triu(predictions)
            gt = np.triu(gt)

            predictions = predictions + predictions.transpose()
            gt = gt + gt.transpose()
            np.fill_diagonal(predictions, 1)
            np.fill_diagonal(gt, 1)

            blocks_pred[adj_name] = self._convert_to_blocks(predictions)
            blocks_gt[adj_name] = self._convert_to_blocks(gt)

        self.evaluate_blocks(blocks_pred, blocks_gt)

        cells, columns, rows = self.recalculate_cells(blocks_pred)

        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self._draw_skeletal(img, rows, is_row=True)
        self._draw_skeletal(img, columns, is_row=False)
        cv2.imwrite(
            os.path.join(self._output_images2, str(self.idx) + "-skeletal.png"), img
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--images_dir",
        type=str,
        required=True,
        help="path to directory containing document-level images.",
    )

    parser.add_argument(
        "-xml",
        "--xml_dir",
        type=str,
        required=True,
        help="path to directory containing document-level ground-truth XML files.",
    )

    parser.add_argument(
        "-o",
        "--ocr_dir",
        type=str,
        required=True,
        help="path to directory containing document-level ocr.",
    )

    parser.add_argument(
        "-p",
        "--pred_dir",
        type=str,
        required=True,
        help="path to directory containing table-level prediction XML files.",
    )

    parser.add_argument(
        "-e",
        "--eval_out",
        type=str,
        required=True,
        help="path of directory in which to write the evaluation results.",
    )

    args = parser.parse_args()

    InferenceOutputEvaluator(
        args.images_dir, args.ocr_dir, args.xml_dir, args.pred_dir, args.eval_out
    ).evaluate()

