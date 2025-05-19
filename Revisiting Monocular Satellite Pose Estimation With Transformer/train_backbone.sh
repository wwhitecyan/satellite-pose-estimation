export CUDA_VISIBLE_DEVICES=$1

# train 1
# ---------------------- backbone resnet50s16 vs resnet50s8 -------------------------
batch_size=25
num_queries=40
num_enc=3

if [ $1 -eq 0 ]; then
    # resnet50s16
    input_size=448
    output_dir=./work_dirs_analyze/backbone/resnet50s16
    python main.py --num_workers 2 --batch_size $batch_size\
        --num_queries $num_queries\
        --backbone resnet50\
        --input_size $input_size\
        --resume ./detr-r50-e632da11.pth\
        --train_index_file train_1.txt\
        --val_index_file val_1.txt\
        --output_dir $output_dir\
        --enc_layers $num_enc --dec_layers $num_enc
elif [ $1 -eq 1 ]; then
    # resnet50s8
    input_size=224
    output_dir=./work_dirs_analyze/backbone/resnet50s8
    python main.py --num_workers 2 --batch_size $batch_size\
        --num_queries $num_queries\
        --backbone resnet50s8\
        --input_size $input_size\
        --resume ./detr-r50-e632da11.pth\
        --train_index_file train_1.txt\
        --val_index_file val_1.txt\
        --output_dir $output_dir\
        --enc_layers $num_enc --dec_layers $num_enc
else
    echo wrong
fi
