export CUDA_VISIBLE_DEVICES=$1

# train 1
# ---------------------- number of enc_layers -------------------------
# num of enc and dec
batch_size=15
num_queries=40
input_size=224

num_enc=$(($1+4))
output_dir=./work_dirs_analyze/train_ed${num_enc}_resnet50s8_l2_1
python main.py --num_workers 2 --batch_size $batch_size\
    --num_queries $num_queries\
    --backbone resnet50s8\
    --input_size $input_size\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir $output_dir\
    --enc_layers $num_enc --dec_layers $num_enc

num_enc=$(($1+1))
output_dir=./work_dirs_analyze/train_ed${num_enc}_resnet50s8_l2_1
python main.py --num_workers 2 --batch_size $batch_size\
    --num_queries $num_queries\
    --backbone resnet50s8\
    --input_size $input_size\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir $output_dir\
    --enc_layers $num_enc --dec_layers $num_enc

## ---------------------- number of query -------------------------
batch_size=20
input_size=224
num_enc=4

num_queries=$(($1*5+40))
output_dir=./work_dirs_analyze/train_ed4_resnet50s8_${num_queries}_l2_1
python main.py --num_workers 2 --batch_size $batch_size\
    --num_queries $num_queries\
    --backbone resnet50s8\
    --input_size $input_size\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir $output_dir\
    --enc_layers $num_enc --dec_layers $num_enc

num_queries=$(($1*5+25))
output_dir=./work_dirs_analyze/train_ed4_resnet50s8_${num_queries}_l2_1
python main.py --num_workers 2 --batch_size $batch_size\
    --num_queries $num_queries\
    --backbone resnet50s8\
    --input_size $input_size\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir $output_dir\
    --enc_layers $num_enc --dec_layers $num_enc

# ---------------------- input_size ------------------------------
batch_size=10
num_queries=40
num_enc=4

input_size=$(($1*16+240))
output_dir=./work_dirs_analyze/train_ed4_resnet50s8_query_40_input_${input_size}_l2_1
python main.py --num_workers 2 --batch_size $batch_size\
    --num_queries $num_queries\
    --backbone resnet50s8\
    --input_size $input_size\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir $output_dir\
    --enc_layers $num_enc --dec_layers $num_enc

input_size=$(($1*16+192))
output_dir=./work_dirs_analyze/train_ed4_resnet50s8_query_40_input_${input_size}_l2_1
python main.py --num_workers 2 --batch_size $batch_size\
    --num_queries $num_queries\
    --backbone resnet50s8\
    --input_size $input_size\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir $output_dir\
    --enc_layers $num_enc --dec_layers $num_enc

