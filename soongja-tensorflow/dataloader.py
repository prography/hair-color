import os
import cv2
import math
import numpy as np


class Dataset:
    def __init__(self, height, width, batch_size, folder):
        self.height = height
        self.width = width
        self.batch_size = batch_size

        # file names(inputs and targets have the same filenames, but are in different directories)
        train_names, val_names, test_names = self.train_valid_test_split(
            os.listdir(os.path.join(folder, 'images')), ratio=(0.98, 0.01, 0.01))

        self.train_inputs, self.train_targets = self.file_paths_to_images(folder, train_names)
        self.val_inputs, self.val_targets = self.file_paths_to_images(folder, val_names)
        self.test_inputs, self.test_targets = self.file_paths_to_images(folder, test_names)

        self.pointer = 0

    def file_paths_to_images(self, folder, file_names):
        inputs = []
        targets = []

        for name in file_names:
            input_file = os.path.join(folder, 'images', name)
            target_file = os.path.join(folder, 'masks', name)

            _input = cv2.imread(input_file)[:, :, ::-1]
            _input = np.array(cv2.resize(_input, (self.height, self.width)))
            # _input = np.multiply(_input, 1.0 / 255)
            inputs.append(_input)

            target = cv2.imread(target_file, 0)
            target = cv2.resize(target, (self.height, self.width))
            target = np.expand_dims(target, axis=3)
            # target = np.multiply(target, 1.0 / 255)
            targets.append(target)

        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, .15, .15)

        N = len(X)
        return (
            X[:int(math.ceil(N * ratio[0]))],
            X[int(math.ceil(N * ratio[0])): int(math.ceil(N * ratio[0] + N * ratio[1]))],
            X[int(math.ceil(N * ratio[0] + N * ratio[1])):]
        )

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
        # print('Pointer:', self.pointer)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

    def val_set(self):
        # permutation = np.random.permutation(len(self.val_inputs))
        # self.val_inputs = [self.val_inputs[i] for i in permutation]
        # self.val_targets = [self.val_targets[i] for i in permutation]
        return np.array(self.val_inputs, dtype=np.float32), np.array(self.val_targets, dtype=np.float32)

    def test_set(self):
        return np.array(self.test_inputs, dtype=np.float32), np.array(self.test_targets, dtype=np.float32)