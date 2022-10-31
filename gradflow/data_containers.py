from __future__ import annotations
from abc import ABC, abstractmethod
from ast import Import
from typing import Tuple
import importlib

import numpy as np

import gradflow

_has_pycuda = importlib.util.find_spec("pycuda") is not None
_pycuda_is_imported = False


class DataContainerBase(ABC):
	@abstractmethod
	def __add__(self, other):
		pass

	@abstractmethod
	def __sub__(self, other):
		pass

	@abstractmethod
	def __matmul__(self, other):
		pass

	@abstractmethod
	def __pow__(self, exp):
		pass

	@abstractmethod
	def __eq__(self, __o: object) -> bool:
		pass

	@abstractmethod
	def __le__(self, __o: object) -> bool:
		pass

	@abstractmethod
	def __ge__(self, __o: object) -> bool:
		pass

	@property
	@abstractmethod
	def shape(self) -> tuple[int]:
		pass

	@abstractmethod
	def item(self):
		pass

	@abstractmethod
	def __mul__(self, other: float):
		pass

	@abstractmethod
	def max(self) -> DataContainerBase:
		pass

	@abstractmethod
	def exp(self) -> DataContainerBase:
		pass

	@abstractmethod
	def T(self) -> DataContainerBase:
		pass


class NumpyDataContainer(DataContainerBase):
	def __init__(self, data: np.ndarray | NumpyDataContainer) -> None:
		super().__init__()

		if isinstance(data, NumpyDataContainer):
			data = data.data
		self.data = data

	def __add__(self, other: NumpyDataContainer):
		return NumpyDataContainer(self.data + other.data)

	def __sub__(self, other: NumpyDataContainer):
		return NumpyDataContainer(self.data - other.data)

	def __matmul__(self, other: NumpyDataContainer):
		return NumpyDataContainer(self.data @ other.data)

	def __pow__(self, exp: NumpyDataContainer):
		return NumpyDataContainer(self.data ** exp)

	def __eq__(self, other: NumpyDataContainer) -> bool:
		return self.data == other.data

	@property
	def shape(self) -> tuple[int]:
		return self.data.shape

	def __le__(self, other: float):
		return self.data <= other

	def __ge__(self, other: float):
		return self.data >= other

	def item(self) -> float:
		return self.data.item()

	def __mul__(self, other: float):
		return self.data * other

	def max(self) -> NumpyDataContainer:
		return NumpyDataContainer(np.array(np.max(self.data)))

	def exp(self) -> NumpyDataContainer:
		return NumpyDataContainer(np.exp(self.data))

	def T(self) -> NumpyDataContainer:
		return NumpyDataContainer(np.transpose(self.data))


class GPUDataContainer(DataContainerBase):
	def __init__(self, data: np.array | GPUDataContainer) -> None:
		super().__init__()
		global _pycuda_is_imported
		global gpuarray

		if not _has_pycuda:
			raise ImportError("In order to use the GPU data backend you must install PyCuda.")
		if not _pycuda_is_imported:
			import pycuda.gpuarray as gpuarray
			import pycuda.autoinit
			_pycuda_is_imported = True


		if isinstance(data, GPUDataContainer):
			data = data.data
		elif isinstance(data, np.ndarray):
			data = gpuarray.to_gpu(data)
		self.data = data

	def __add__(self, other: GPUDataContainer):
		return GPUDataContainer(self.data + other.data)

	def __sub__(self, other: GPUDataContainer):
		return GPUDataContainer(self.data - other.data)

	def __matmul__(self, other: GPUDataContainer):
		return GPUDataContainer(self.data @ other.data)

	def __pow__(self, exp: GPUDataContainer):
		return GPUDataContainer(self.data ** exp)

	def __eq__(self, other: GPUDataContainer) -> bool:
		return self.data == other.data

	@property
	def shape(self) -> tuple[int]:
		return self.data.shape

	def __le__(self, other: float):
		return self.data <= other

	def __ge__(self, other: float):
		return self.data >= other

	def item(self) -> float:
		return self.data.get()

	def __mul__(self, other: float):
		return GPUDataContainer(self.data * other)

	def max(self) -> GPUDataContainer:
		return GPUDataContainer(np.array(np.max(self.data)))

	def exp(self) -> GPUDataContainer:
		return GPUDataContainer(np.exp(self.data))

	def T(self) -> GPUDataContainer:
		return GPUDataContainer(np.transpose(self.data))


if __name__ == "__main__":
	pass
