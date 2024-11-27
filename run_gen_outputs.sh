echo "Model Name: " $1

python3 generate_dataset.py \
    --dataset_name Kendamarron/jimba-instuction-1k-beta \
    --split train \
    --column_name instruction \
    --prompt_path prompts/evaluation_prompt.txt \
    --num_generations 4 \
    --sample_size 5 \
    --output_dir ./output \
    --model_name $1 \
    --batch_size 4