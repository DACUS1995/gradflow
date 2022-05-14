import logging

def init_logger():
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	handler = logging.StreamHandler()
	handler.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	
	logger.addHandler(handler)