import os
import math
import cv2
import numpy as np
from math import ceil


class Dataset:
    def __init__(self, batch_size, folder='dataset'):
        self.batch_size = batch_size

        train_files, validation_files, test_files = self.train_valid_test_split(
                                                            os.listdir(os.path.join(folder, 'inputs')))

        self.train_inputs, self.train_targets = self.file_paths_to_images(folder, train_files)
        self.validation_inputs, self.validation_targets = self.file_paths_to_images(folder, validation_files)
        self.test_inputs, self.test_targets = self.file_paths_to_images(folder, test_files)

        self.pointer = 0

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.8, 0.1, 0.1)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def file_paths_to_images(self, folder, files_list):
        inputs = []
        targets = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs', file)
            input_image = np.array(cv2.imread(input_image))
            inputs.append(input_image)

            target_image = os.path.join(folder, 'targets', file)
            target_image = cv2.imread(target_image)
            target_image = np.multiply(target_image, 1.0 / 255)
            targets.append(target_image)

        return inputs, targets

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)