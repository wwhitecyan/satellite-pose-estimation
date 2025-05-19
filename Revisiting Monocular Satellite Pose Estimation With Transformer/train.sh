export CUDA_VISIBLE_DEVICES=4,5,6,7
# ---------------------- multi gpu train for submission -------------------------
batch_size=10
num_queries=20
input_size=256
num_enc=3

for idx in 1 2 3 4 5 6
do
#idx=6
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py\
        --lr $lr --lr_backbone $lr_backbone --bn sync_bn\
        --enc_layers $num_enc --dec_layers $num_enc\
        --num_workers 2\
        --backbone resnet50s8\
        --resume ./detr-r50-e632da11.pth\
        --batch_size $batch_size\
        --num_queries $num_queries\
        --input_size $input_size\
        --train_index_file train_$idx.txt\
        --val_index_file val_$idx.txt\
        --output_dir ./work_dirs_sync_bn/split_$idx
    ps -e | grep python | awk '{print $1}' | xargs kill
done
