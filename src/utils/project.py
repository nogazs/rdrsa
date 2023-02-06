from utils.files import *
import sys
from tdict import Tdict
import numpy as np
import logging
from IPython import get_ipython
import os


def running_in_notebook():
    try:
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            return False
    except:
        return False
    else:  # pragma: no cover
        return True


def get_logger(logger_name, level='debug'):

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    if level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warn':
        logger.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger
