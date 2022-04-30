from typing import Tuple

import numpy as np

from gradflow.grad_engine import Variable

class Dataset:
    def __init__(self, features: np.ndarray, labels: np.ndarray, batch_size=16) -> None:
        self._features = features
        self._labels = labels
        self._batch_size = batch_size
        self._cur_index = 0

    
    def __iter__(self):
        self._cur_index = 0
        return self


    def __next__(self) -> Tuple[Variable, Variable]:
        if self._cur_index >= len(self):
            raise StopIteration
        
        sample_batch, label_batch = self[self._cur_index]
        self._cur_index += self._batch_size
        
        return sample_batch, label_batch

    
    def __getitem__(self, idx) -> Tuple[Variable, Variable]:
        if idx >= len(self):
            raise IndexError

        sample_batch = Variable(self._features[self._cur_index: self._cur_index + self._batch_size])
        label_batch = Variable(self._labels[self._cur_index : self._cur_index + self._batch_size])

        return sample_batch, label_batch

    
    def __len__(self) -> int:
        return int(len(self._features) / self._batch_size)