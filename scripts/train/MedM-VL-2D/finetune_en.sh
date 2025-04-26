export PYTHONPATH=$PYTHONPATH:/home/che/

deepspeed --include localhost:0 --master_port 29501 lvlm/train.py \
    --deepspeed scripts/train/utils/zero3.json \
    --data_path data/dataset/qd_train.json \
    --conv_version qwen2 \
    --image_path / \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_encoder_image full \
    --tune_type_connector_image full \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir work_dirs/MedM-VL-2D-3B-en-finetune \
    --resume_from_checkpoint work_dirs/MedM-VL-2D-3B-en \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
