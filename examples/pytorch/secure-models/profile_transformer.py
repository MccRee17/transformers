import torch
import torch.nn as nn

import crypten
import crypten.communicator as comm
from crypten.config import cfg

from utils import encrypt_tensor, encrypt_model


# setup crypten
crypten.init()
cfg.communicator.verbose = True

# setup model and data




commCost = crypten.communicator.get().get_communication_stats()
