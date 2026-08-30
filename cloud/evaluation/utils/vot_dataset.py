# evaluation/utils/vot_dataset.py

"""
VOT2022 Dataset Loader
"""

import os
import cv2
import torch

from torch.utils.data import Dataset


# =========================================================
# RefVotDataset
# =========================================================
class RefVotDataset:

    def __init__(self, vot_dataset_root):

        vot_dataset_root = os.path.expanduser(vot_dataset_root)

        self.vot_dataset_root = vot_dataset_root

        sequences_dir = os.path.join(
            vot_dataset_root,
            "sequences"
        )

        self.sequences = sorted(
            os.listdir(sequences_dir)
        )

        self.sequence_index = 0

    def __iter__(self):

        self.sequence_index = 0
        return self

    def __len__(self):

        return len(self.sequences)

    def __next__(self):

        if self.sequence_index >= len(self.sequences):
            raise StopIteration

        sequence = self.sequences[self.sequence_index]

        self.sequence_index += 1

        sequence_root_dir = os.path.join(
            self.vot_dataset_root,
            "sequences",
            sequence
        )

        # ===================================
        # language.txt
        # ===================================

        language_file = os.path.join(
            sequence_root_dir,
            "language.txt"
        )

        if os.path.exists(language_file):

            with open(language_file, "r") as f:

                sequence_query = f.readline().strip()

        else:

            sequence_query = sequence

        return sequence_query, VotSequenceDataset(
            sequence_root_dir,
            transform=None
        )


# =========================================================
# VotSequenceDataset
# =========================================================
class VotSequenceDataset(Dataset):

    def __init__(
        self,
        sequence_root_dir,
        transform=None
    ):

        self.sequence_root_dir = sequence_root_dir

        self.transform = transform

        self.gt_file = os.path.join(
            sequence_root_dir,
            "groundtruth.txt"
        )

        self.image_dir = os.path.join(
            sequence_root_dir,
            "color"
        )

        self.sequence_name = os.path.basename(
            sequence_root_dir
        )

        self._data = self.setup_dataset()

        self.idx = 0

    # =====================================================
    # setup dataset
    # =====================================================
    def setup_dataset(self):

        with open(self.gt_file, "r") as f:

            lines = f.readlines()

        gt_bboxes = [
            list(map(float, line.strip().split(",")))
            for line in lines
        ]

        gt_bboxes = [
            ltwh_to_xyxy(box)
            for box in gt_bboxes
        ]

        gt_bboxes = [
            torch.tensor(box)
            for box in gt_bboxes
        ]

        file_list = sorted(
            os.listdir(self.image_dir)
        )

        images = [
            os.path.join(self.image_dir, img_name)
            for img_name in file_list
        ]

        data = []

        for image, gt_bbox in zip(images, gt_bboxes):

            data.append({
                "image": image,
                "gt_bbox": gt_bbox
            })

        return data

    # =====================================================
    # len
    # =====================================================
    def __len__(self):

        return len(self._data)

    # =====================================================
    # iter
    # =====================================================
    def __iter__(self):

        self.idx = 0
        return self

    # =====================================================
    # next
    # =====================================================
    def __next__(self):

        if self.idx >= len(self._data):
            raise StopIteration

        sample = self._data[self.idx]

        self.idx += 1

        image_path = sample["image"]

        gt_bbox = sample["gt_bbox"]

        image = cv2.imread(image_path)

        success = image is not None

        return image, success, gt_bbox

    # =====================================================
    # helper functions
    # =====================================================
    def get_sequence_name(self):

        return self.sequence_name

    def get_resolution(self):

        image_path = self._data[0]["image"]

        image = cv2.imread(image_path)

        return image.shape[0], image.shape[1]

    def get_sequence_path(self):

        return self.image_dir


# =========================================================
# bbox conversion
# =========================================================
def ltwh_to_xyxy(ltwh):

    x, y, w, h = ltwh

    return [
        x,
        y,
        x + w,
        y + h
    ]