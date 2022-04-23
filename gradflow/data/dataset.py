from typing import Tuple

import numpy as np

class Dataset:
    def __init__(self, features: np.ndarray, labels: np.ndarray, batch_size=16) -> None:
        self._features = features
        self._labels = labels
        self._batch_size = batch_size
        self._cur_index = 0

    
    def __iter__(self):
        self._cur_index = 0
        return self


    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cur_index >= len(self.samples):
            raise StopIteration
        
        sample_batch, label_batch = self[self._cur_index]
        self._cur_index += self._batch_size
        
        return sample_batch, label_batch

    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        sample_batch = self._features[self._cur_index: self._cur_index + self._batch_size]
        label_batch = self._labels[self._cur_index : self._cur_index + self._batch_size]

        return sample_batch, label_batch

    
    def __len__(self) -> int:
        return int(len(self._features) / self._batch_size)