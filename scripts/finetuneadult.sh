# python qlora.py \
#     --model_name_or_path /zoo/llama2/llama2-7b-hf \
#     --output_dir ./output/vanillaadult \
#     --logging_steps 40 \
#     --save_strategy steps \
#     --data_seed 42 \
#     --save_steps 10 \
#     --save_total_limit 15 \
#     --evaluation_strategy no \
#     --eval_dataset_size 1024 \
#     --max_eval_samples 1000 \
#     --per_device_eval_batch_size 1 \
#     --max_new_tokens 60 \
#     --dataloader_num_workers 3 \
#     --group_by_length \
#     --logging_strategy steps \
#     --remove_unused_columns False \
#     --do_train \
#     --do_eval \
#     --do_mmlu_eval \
#     --lora_r 64 \
#     --lora_alpha 16 \
#     --lora_modules all \
#     --double_quant \
#     --quant_type nf4 \
#     --bf16 \
#     --bits 4 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type constant \
#     --gradient_checkpointing \
#     --dataset ./data/vanilla-adult.dat \
#     --dataset_format 'input-output' \
#     --source_max_len 16 \
#     --target_max_len 512 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --max_steps 150 \
#     --eval_steps 187 \
#     --learning_rate 0.0002 \
#     --adam_beta2 0.999 \
#     --max_grad_norm 0.3 \
#     --lora_dropout 0.1 \
#     --weight_decay 0.0 \
#     --seed 0
python vanillasample.py \
    --model_name_or_path /zoo/llama2/llama2-7b-hf \
    --output_dir ./output/vanillaadult \
    --logging_steps 40 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 10 \
    --save_total_limit 12 \
    --evaluation_strategy no \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 45 \
    --do_sample True \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset ./data/vanilla-adult.dat \
    --dataset_format 'input-output' \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 150 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0
