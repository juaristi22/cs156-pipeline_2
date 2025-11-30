import torch
# from icecream import install  # Commented out - optional debugging tool

torch.set_num_threads(1)
# install()  # Commented out - optional debugging tool

from . import env  # noqa
from .data import *  # noqa
from .deep import *  # noqa
from .env import *  # noqa
from .metrics import *  # noqa
from .util import *  # noqa