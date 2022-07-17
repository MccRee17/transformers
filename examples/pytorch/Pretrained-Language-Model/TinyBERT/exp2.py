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
parser.add_argument('--metric_name', type=str)
parser.add_argument('--network_type', type=str)
parser.add_argument('--tune_lr_hidden', type=float, default=None)
parser.add_argument('--tune_lr_pred', type=float, default=None)
parser.add_argument('--tune_bs', type=int, default=None)

args = parser.parse_args()
task_name = args.task_name
metric_name = args.metric_name
tune_lr_hidden = args.tune_lr_hidden
tune_lr_pred = args.tune_lr_pred
tune_bs = args.tune_bs

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

num_devices = torch.cuda.device_count()

def HPO_teacher():
    lr_list = [2e-5, 3e-5, 4e-5, 5e-5]
    bs = 32 // num_devices
    teacher_acc = []
    
    for lr in lr_list:
        output_dir = os.path.join(base_dir, str(lr))
        result_path = os.path.join(output_dir, "eval_results.json")
        cmd = f"python run_glue.py --model_name_or_path bert-base-uncased --task_name {task_name} \
              --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {bs} --learning_rate {str(lr)} \
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
            cmd = f"python run_glue.py --model_name_or_path huawei-noah/TinyBERT_General_4L_312D \
                  --task_name {task_name} \
                  --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {str(bs//num_devices)} --learning_rate {str(lr)} \
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

def HPO_S2():
    #lr_list = [3e-5, 5e-5, 1e-4]
    #bs_list = [16, 64, 128]
    lr_list = [3e-5]
    bs_list = [8]
    best = None
    best_metric = 0

    for lr in lr_list:
        for bs in bs_list:
            # distill hidden layers
            output_dir = os.path.join(base_dir, "S2" ,str(lr), str(bs))
            result_path = os.path.join(output_dir, "eval_results.json")
            data_dir = os.path.join("glue_data", task_name)
            teacher_dir = f"exp_{task_name}_t"
            student_dir = f"{exp_name}_{task_name}_S2"
            
            # overwrite config
            config = json.load(open(os.path.join(teacher_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(teacher_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)

            config = json.load(open(os.path.join(student_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(student_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)

            
            cmd = f"python task_distill.py --teacher_model {teacher_dir} \
                       --student_model {student_dir} \
                       --data_dir {data_dir} --task_name {task_name} --output_dir {output_dir} \
                       --max_seq_length 512 --train_batch_size {bs} \
                       --do_lower_case --log_path {log_path}"

            subprocess.run(cmd, shell=True)

            # distill pred layers
            config = json.load(open(os.path.join(output_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(output_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)
            
            output_dir_stage2 = os.path.join(base_dir, "S2" ,str(lr), str(bs)+"_stage2")
            result_path = os.path.join(output_dir_stage2, "eval_results.json")
            data_dir = os.path.join("glue_data", task_name)
            cmd = f"python task_distill.py --pred_distill  \
                       --teacher_model {teacher_dir} \
                       --student_model {output_dir} \
                       --data_dir {data_dir} \
                       --task_name {task_name} \
                       --output_dir {output_dir_stage2} \
                       --do_lower_case \
                       --learning_rate {lr}  \
                       --num_train_epochs  5 \
                       --eval_step 100 \
                       --max_seq_length 512 \
                       --train_batch_size {bs} --log_path {log_path}"

            subprocess.run(cmd, shell=True)
            with open(log_path, "a") as f:
                f.write(f"distilled S2 with lr {str(lr)} bs {str(bs)} \n")

def tune_S2():
    assert tune_lr_hidden is not None
    assert tune_lr_pred is not None
    assert tune_bs is not None


    for _ in [0]:
        for _ in [0]:
        # distill hidden layers
            output_dir = os.path.join(base_dir, "S2" ,str(tune_lr_hidden)+"_"+str(tune_lr_pred), str(tune_bs))
            result_path = os.path.join(output_dir, "eval_results.json")
            data_dir = os.path.join("glue_data", task_name)
            teacher_dir = f"exp_{task_name}_t"
            student_dir = f"{exp_name}_{task_name}_S2"
            
            # overwrite config
            config = json.load(open(os.path.join(teacher_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(teacher_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)

            config = json.load(open(os.path.join(student_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(student_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)

            
            cmd = f"python task_distill.py --teacher_model {teacher_dir} \
                       --student_model {student_dir} \
                       --data_dir {data_dir} --task_name {task_name} --output_dir {output_dir} \
                       --max_seq_length 512 --train_batch_size {tune_bs} --learning_rate {tune_lr_hidden}\
                       --do_lower_case --log_path {log_path}"

            subprocess.run(cmd, shell=True)

            # distill pred layers
            config = json.load(open(os.path.join(output_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(output_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)
            
            output_dir_stage2 = os.path.join(base_dir, "S2" ,str(tune_lr_hidden)+"_"+str(tune_lr_pred), str(tune_bs)+"_stage2")
            result_path = os.path.join(output_dir_stage2, "eval_results.json")
            data_dir = os.path.join("glue_data", task_name)
            cmd = f"python task_distill.py --pred_distill  \
                       --teacher_model {teacher_dir} \
                       --student_model {output_dir} \
                       --data_dir {data_dir} \
                       --task_name {task_name} \
                       --output_dir {output_dir_stage2} \
                       --do_lower_case \
                       --learning_rate {tune_lr_pred}  \
                       --num_train_epochs  5 \
                       --eval_step 100 \
                       --max_seq_length 512 \
                       --train_batch_size {tune_bs} --log_path {log_path}"

            subprocess.run(cmd, shell=True)
            with open(log_path, "a") as f:
                f.write(f"distilled S2 with lr {str(tune_lr_hidden)} {str(tune_lr_pred)} bs {str(bs)} \n")
assert args.network_type in ["teacher", "S1", "S2", "tune_S2"]
if args.network_type == "teacher":
    HPO_teacher()
elif args.network_type == "S1":
    HPO_S1()
elif args.network_type == "S2":
    HPO_S2()
else:
    tune_S2()

# hold GPU
a = torch.randn(50,50).cuda()
while True:
    a ** 2
