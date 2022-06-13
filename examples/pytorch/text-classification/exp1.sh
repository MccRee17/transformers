CUDA_VISIBLE_DEVICES=1,2,3,4 python run_glue.py   --model_name_or_path bert-base-uncased   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 512   --per_device_train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 3   --output_dir ./tmp/$TASK_NAME/ft_gelu --act gelu --softmax_act softmax  --overwrite_output_dir --fp16 --eval_steps 200 --evaluation_strategy steps > finetune.txt
