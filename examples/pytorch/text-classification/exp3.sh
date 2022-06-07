CUDA_VISIBLE_DEVICES=5,6 python run_glue.py   --model_name_or_path prajjwal1/bert-small   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 128   --learning_rate 1e-4   --num_train_epochs 50   --output_dir ./tmp/$TASK_NAME/quad --act gelu --softmax_act softmax  --fp16 --overwrite_output_dir  --eval_steps 200 --evaluation_strategy steps > gelu_softmax.txt
