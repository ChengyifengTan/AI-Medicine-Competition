export PYTHONPATH=$PYTHONPATH:/home/che/MedM-VL
MODEL_PATH="work_dirs/Qwen2.5-3B-Instruct"
CUDA_VISIBLE_DEVICES=0 python lvlm/playground.py \
    --model_dtype bfloat16 \
    --conv_version llama3 \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 2048 \
    --num_beams 1 \
    --temperature 0