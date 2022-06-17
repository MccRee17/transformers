# example scripts for running various KD methods
# use cifar10, resnet110 (teacher) and resnet20 (student) as examples

# Baseline
CUDA_VISIBLE_DEVICES=3 python -u train_base.py \
                           --save_root "./results/base/" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --net_name resnet20 \
			   --note base-resnet20
