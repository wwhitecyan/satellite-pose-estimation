export CUDA_VISIBLE_DEVICES=$1
num_queries=$(($1*5+20))
python main.py --num_workers 2 --batch_size 20\
    --lr 0.0001 --lr_backbone 0.00001 --num_queries $num_queries\
    --backbone resnet50s8\
    --position_embedding sine --input_size 224\
    --dim_feedforward 2048 --set_cost_pts 5 \
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir ./work_dirs/train_ed4_resnet50s8_query_${num_queries}_l2_1\
    --enc_layers 4 --dec_layers 4

num_queries=$(($1*5+35))
python main.py --num_workers 2 --batch_size 20\
    --lr 0.0001 --lr_backbone 0.00001 --num_queries $num_queries\
    --backbone resnet50s8\
    --position_embedding sine --input_size 224\
    --dim_feedforward 2048 --set_cost_pts 5 \
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_1.txt\
    --val_index_file val_1.txt\
    --output_dir ./work_dirs/train_ed4_resnet50s8_query_${num_queries}_l2_1\
    --enc_layers 4 --dec_layers 4
