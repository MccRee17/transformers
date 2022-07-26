import os
import subprocess
import json
import shutil
import itertools
import argparse
import torch

exp_name = "exp2"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--network_type', type=str)
parser.add_argument('--metric_name', type=str)
args = parser.parse_args()

task_name = args.task_name
model_path = args.model_path 
model_type = args.model_type
network_type = args.network_type
metric_name = args.metric_name

base_dir = f"./tmp/{exp_name}/{task_name}/{model_type}"
if not os.path.exists("./tmp"):
    os.mkdir("./tmp")
if not os.path.exists(f"./tmp/{exp_name}"):
    os.mkdir(f"./tmp/{exp_name}")
if not os.path.exists(f"./tmp/{exp_name}/{task_name}"):
    os.mkdir(f"./tmp/{exp_name}/{task_name}")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
tinybert_path = "/home/ubuntu/transformers/examples/pytorch/Pretrained-Language-Model/TinyBERT"
log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("new run \n")

num_devices = torch.cuda.device_count()

def HPO_S0():
    lr_list = [5e-6,1e-6]
    bs_list = [384, 256]
    best = None
    best_metric = 0

    for lr in lr_list:
        for bs in bs_list:
            output_dir = os.path.join(base_dir, "S0" ,str(lr), str(bs))
            result_path = os.path.join(output_dir, "eval_results.json")
            cmd = f"python run_glue_scratch.py --model_name_or_path bert-base-uncased \
                  --task_name {task_name} --warmup_ratio 0.2\
                  --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {str(bs//num_devices)} --learning_rate {str(lr)} \
                  --num_train_epochs 10 --act quad --softmax_act 2relu --output_dir {output_dir} --overwrite_output_dir"

            subprocess.run(cmd, shell=True)
            result = json.load(open(result_path))
            metric = float(result[metric_name])
            if metric > best_metric:
                best = (lr, bs)
                best_metric = metric
            with open(log_path, "a") as f:
                f.write(f"pretrain S0 with lr {str(lr)} bs {str(bs)}, acc: {metric} \n")

    best_lr, best_bs = best
    with open(log_path, "a") as f:
        f.write(f"best S0 with lr {best_lr} bs {best_bs}, acc: {best_metric} \n")


def HPO_S1():
    lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    bs_list = [256, 128, 64, 32, 16]
    epoch_list = [50, 80, 3, 10, 30]
    #lr_list = [1e-4]
    #bs_list = [8]
    best = None
    best_metric = 0

    for lr in lr_list:
        for bs in bs_list:
            for epoch in epoch_list:
                output_dir = os.path.join(base_dir, "HPO_S1" ,str(lr), str(bs), str(epoch))
                result_path = os.path.join(output_dir, "eval_results.json")
                cmd = f"python run_glue.py --model_name_or_path {model_path} \
                      --task_name {task_name} \
                      --do_train --do_eval --max_seq_length 128 --warmup_ratio 0.2 --per_device_train_batch_size {str(bs//num_devices)} --learning_rate {str(lr)} \
                      --num_train_epochs {epoch} --act quad --softmax_act 2relu --output_dir {output_dir} --overwrite_output_dir"

                subprocess.run(cmd, shell=True)
                result = json.load(open(result_path))
                metric = float(result[metric_name])
                if metric > best_metric:
                    best = (lr, bs, epoch)
                    best_metric = metric
                with open(log_path, "a") as f:
                    f.write(f"fine-tuned S1 with lr {str(lr)} bs {str(bs)} epoch {epoch}, acc: {metric} \n")

    best_lr, best_bs, best_epoch = best
    with open(log_path, "a") as f:
        f.write(f"best S1 with lr {best_lr} bs {best_bs} epoch {best_epoch}, acc: {best_metric} \n")

assert network_type in ["S0", "S1"]
if network_type == "S0":
    HPO_S0()
else:
    HPO_S1()

# hold GPU
a = torch.randn(50,50).cuda()
while True:
    a ** 2
