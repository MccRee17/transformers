import os
import subprocess
import json
import shutil
import itertools
import argparse
import torch

exp_name = "HPO_S0"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--metric_name', type=str)
args = parser.parse_args()

task_name = args.task_name
metric_name = args.metric_name

base_dir = f"./tmp/{exp_name}/{task_name}/"
if not os.path.exists("./tmp"):
    os.mkdir("./tmp")
if not os.path.exists(f"./tmp/{exp_name}"):
    os.mkdir(f"./tmp/{exp_name}")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
tinybert_path = "/home/ubuntu/transformers/examples/pytorch/Pretrained-Language-Model/TinyBERT"
log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("new run")

num_devices = torch.cuda.device_count()

# task specific epoch list
task_default_epoch = {"MNLI": {3, 10, 20}, "RTE": {2}, "CoLA": {5, 20, 50}, "SST2": {30, 50}}

def HPO_S0():
    lr_list = [1e-5]#, 1e-4, 1e-3]
    global_bs_list = [256]
    gradient_accumulation_steps = 4
    pretrain_ratio = 0.9 # Bert-base use 90% of the time as seq=128 training
    total_epoch_list = task_default_epoch[task_name] #[5, 10, 20]
    best, best_metric = None, 0

    for lr in lr_list:
        for global_bs in global_bs_list:
            for total_epoch in total_epoch_list:
                bs = global_bs // gradient_accumulation_steps
                epoch = int(total_epoch * pretrain_ratio)
                output_dir = os.path.join(base_dir, str(epoch) + "_" + str(lr) + "_" + str(bs))
                result_path = os.path.join(output_dir, "eval_results.json")
                cmd = f"python run_glue_scratch.py --model_name_or_path bert-base-uncased \
                      --task_name {task_name} --warmup_ratio 0.2\
                      --do_train --do_eval --max_seq_length 128 --gradient_accumulation_steps 4 --fp16 --per_device_train_batch_size {str(bs//num_devices)} --learning_rate {str(lr)} \
                      --num_train_epochs {epoch} --act quad --softmax_act 2relu --output_dir {output_dir}"

                subprocess.run(cmd, shell=True)
                result = json.load(open(result_path))
                metric = float(result[metric_name])
                if metric > best_metric:
                    best = (lr, bs, total_epoch)
                    best_metric = metric
                with open(log_path, "a") as f:
                    f.write(f"(HPO) S0 with lr {str(lr)} bs {str(bs)} in {epoch} epoches, acc: {metric} \n")

    best_lr, best_bs, best_epoch = best
    with open(log_path, "a") as f:
        f.write(f"(HPO) Best S0 with lr {str(best_lr)} bs {str(best_bs)} in {best_epoch} epoches, acc: {best_metric} \n")
    
    # Training using the best config and seq=512, use 90% as seq=128 training
    output_dir = os.path.join(base_dir, str(best_epoch) + "_" + str(best_lr) + "_" + str(best_bs) + "_seq512")
    result_path = os.path.join(output_dir, "eval_results.json")
    trained_epoch = int(best_epoch * pretrain_ratio)
    to_train_epoch = best_epoch - trained_epoch
    to_train_lr = best_lr * (1/2) # Follow Nvidia lr schedule
    trained_dir = os.path.join(base_dir, str(trained_epoch) + "_" + str(best_lr) + "_" + str(best_bs))
    cmd = f"python run_glue.py --model_name_or_path  {trained_dir} \
            --task_name {task_name} --warmup_ratio 0.2 \
            --do_train --do_eval --max_seq_length 512 --gradient_accumulation_steps 4 --fp16 --per_device_train_batch_size {str(best_bs//num_devices)} --learning_rate {to_train_lr} \
            --num_train_epochs {to_train_epoch} --act quad --softmax_act 2relu --output_dir {output_dir}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    result = json.load(open(result_path))
    metric = float(result[metric_name])
    with open(log_path, "a") as f:
        f.write(f"Finish S0 with lr {str(best_lr)} bs {str(best_bs)} in {best_epoch} epoches (seq=512), acc: {metric} \n")

HPO_S0()

# hold GPU
a = torch.randn(50,50).cuda()
while True:
    a ** 2
