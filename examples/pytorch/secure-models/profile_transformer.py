import sys

import torch
import torch.nn as nn

from transformers import AutoConfig, BertForSequenceClassification

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model


# Inference arguments
batch_size = 32
sequence_length = 512
model_name = "prajjwal1/bert-small"
act = "relu"
softmax_act = "softmax"

# inference configuration and model
config = AutoConfig.from_pretrained(
        model_name,
        num_labels=3,
        finetuning_task="mnli",
        cache_dir=None,
        revision="main",
        use_auth_token=False,
    )

config.hidden_act = act
config.softmax_act = softmax_act
config.crypten = True
print(f"using model config: {config}")

model = BertForSequenceClassification(config)

# setup fake data for timing purpose
inputs = dict
input["input_ids"] = torch.randint(low=0, high=config.vocab_size, size=(batch_size, sequence_length)).cuda()
input["token_type_ids"] = torch.zeros(batch_size, sequence_length).cuda()
input["attention_mask"] = torch.zeros(batch_size, sequence_length).cuda()
input["labels"] = torch.zeros(batch_size).cuda()


# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "10.117.1.35"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

# setup crypten
crypten.init()
cfg.communicator.verbose = True

commInit = crypten.communicator.get().get_communication_stats()
print(commInit)

# run a forward pass
model(**inputs)

commFinish = crypten.communicator.get().get_communication_stats()
print(commFinish)
