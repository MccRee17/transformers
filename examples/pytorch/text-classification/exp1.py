import os
import subprocess
import json
import shutil
import itertools
import argparse

exp_name = "exp1"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type='str')
parser.add_argument('--gpu', type="str")

args = parser.parse_args()
task_name = args.task_name
gpu = parser.gpu

metric_name = "eval_accuracy"
base_dir = f"./tmp/{exp_name}/{task_name}/"
if not os.path.exists("./tmp"):
    os.mkdir("./tmp")
if not os.path.exists(f"./tmp/{exp_name}"):
    os.mkdir(f"./tmp/{exp_name}")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
tinybert_path = "/home/ubuntu/transformers/examples/pytorch/Pretrained-Language-Model/TinyBERT"
log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "w") as f:
    pass

def HPO_teacher():
    lr_list = [2e-5, 3e-5, 4e-5, 5e-5]
    teacher_acc = []

    for lr in lr_list:
        output_dir = os.path.join(base_dir, str(lr))
        result_path = os.path.join(output_dir, "eval_results.json")
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python run_glue.py --model_name_or_path bert-base-uncased --task_name {task_name} \
              --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate {str(lr)} \
              --num_train_epochs 3 --act gelu --softmax_act softmax --output_dir {output_dir} --overwrite_output_dir"
        subprocess.run(cmd, shell=True)
        result = json.load(open(result_path))
        acc = float(result[metric_name])
        teacher_acc.append(acc)
        with open(log_path, "a") as f:
            f.write(f"fine-tuned Bert base with lr {str(lr)}, acc: {acc} \n")

    max_acc = max(teacher_acc)
    best_lr = lr_list[teacher_acc.index(max_acc)]
    with open(log_path, "a") as f:
        f.write(f"best teacher with lr {best_lr}, acc: {max_acc} \n")

    teacher_path = os.path.join(base_dir, str(best_lr))
    dst_path = os.path.join(tinybert_path, f"{exp_name}_{task_name}_t")
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    shutil.copytree(teacher_path, dst_path)

def HPO_S1():
    lr_list = [2e-5, 5e-5, 1e-4]
    bs_list = [16, 64, 128]
    best = None
    best_metric = 0

    for lr in lr_list:
        for bs in bs_list:
            output_dir = os.path.join(base_dir, "S1" ,str(lr), str(bs))
            result_path = os.path.join(output_dir, "eval_results.json")
            cmd = f"CUDA_VISIBLE_DEVICES={gpu} python run_glue.py --model_name_or_path huawei-noah/TinyBERT_General_4L_312D \
                  --task_name {task_name} \
                  --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {str(bs)} --learning_rate {str(lr)} \
                  --num_train_epochs 3 --act quad --softmax_act 2relu --output_dir {output_dir} --overwrite_output_dir"

            subprocess.run(cmd, shell=True)
            result = json.load(open(result_path))
            metric = float(result[metric_name])
            if metric > best_metric:
                best = (lr, bs)
                best_metric = metric
            with open(log_path, "a") as f:
                #f.write(cmd)
                f.write(f"fine-tuned S1 with lr {str(lr)} bs {str(bs)}, acc: {metric} \n")

    best_lr, best_bs = best
    with open(log_path, "a") as f:
        f.write(f"best S1 with lr {best_lr} bs {best_bs}, acc: {best_metric} \n")

HPO_teacher()
HPO_S1()
