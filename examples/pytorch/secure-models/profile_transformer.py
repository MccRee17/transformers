import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from models import Bert, BertEmbeddings

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 2
       self.num_hidden_layers = 6
       self.hidden_size = 512
       self.intermediate_size = 2048
       self.sequence_length = 512
       self.max_position_embeddings = 512
       self.hidden_act = "relu"
       self.softmax_act = "softmax"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 8
       self.vocab_size = 100
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "10.117.1.22"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
print(commInit)
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().cuda()

m = Bert(config)
model = encrypt_model(m, Bert, config, input_ids).eval()

# encrpy inputs
input_ids = encrypt_tensor(input_ids)

commInit = crypten.communicator.get().get_communication_stats()
print(commInit)
time_s = time.time()
# run a forward pass
with crypten.no_grad():
    model(input_ids)

commFinish = crypten.communicator.get().get_communication_stats()
print(commFinish)
time_e = time.time()
print(time_e - time_s)
