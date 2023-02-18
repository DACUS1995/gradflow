import logging
import threading
from typing import Any


logger = logging.getLogger()


def init_logger():
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	handler = logging.StreamHandler()
	handler.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	
	logger.addHandler(handler)


_thread_local_storage = threading.local()
_thread_local_storage.grad_enabled: bool = True


def grad_enabled() -> bool:
	return _thread_local_storage.grad_enabled


def set_grad_enabled(enabled: bool = True):
	_thread_local_storage.grad_enabled = enabled


class no_grad:
	def __init__(self) -> None:
		self.grad_enabled_prev = False

	def __enter__(self) -> None:
		self.grad_enabled_prev = grad_enabled()
		set_grad_enabled(False)

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
		if exc_type is None:
			set_grad_enabled(self.grad_enabled_prev)
		else:
			logger.warning(f"exc_type={exc_type} | exc_value={exc_value} \n traceback={traceback}")
	