#python main.py --output_dir train1 --num_workers 4 --lr 0.0001 --num_queries 100
#python main.py --output_dir train_sine --num_workers 4 --batch_size 36 \
#    --lr 0.0001 --num_queries 100 --position_embedding sine --input_size 512\
#    --resume ./detr-r50-e632da11.pth
#python main.py --output_dir ./work_dirs/train_layer3 --num_workers 2\
#    --batch_size 70 --lr 0.001 --lr_backbone 0.0001 --num_queries 30\
#    --position_embedding sine --input_size 256\
#    --dim_feedforward 2048 --set_cost_pts 5
#python main.py --output_dir ./work_dirs/train_layer3_wo_pretrain --num_workers 2\
#    --batch_size 100 --lr 0.0001 --lr_backbone 0.00001 --num_queries 30\
#    --position_embedding sine --input_size 256\
#    --dim_feedforward 2048 --set_cost_pts 5 \
#    --enc_layers 3 --dec_layers 3

#python main.py --output_dir ./work_dirs/train_resnet18s --num_workers 2\
#    --batch_size 100 --lr 0.0001 --lr_backbone 0.00001 --num_queries 30\
#    --backbone resnet18s\
#    --position_embedding sine --input_size 224\
#    --dim_feedforward 2048 --set_cost_pts 5 \
#    --enc_layers 4 --dec_layers 4

export CUDA_VISIBLE_DEVICES=$1
python main.py --num_workers 2 --batch_size 20\
    --lr 0.0001 --lr_backbone 0.00001 --num_queries 40\
    --backbone resnet50s8\
    --position_embedding sine --input_size 224\
    --dim_feedforward 2048 --set_cost_pts 5 \
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_$(($1+1)).txt\
    --val_index_file val_$(($1+1)).txt\
    --output_dir ./work_dirs/train_ed4_query40_resnet50s8_l2_$(($1+1)) \
    --enc_layers 4 --dec_layers 4

python main.py --num_workers 2 --batch_size 20\
    --lr 0.0001 --lr_backbone 0.00001 --num_queries 40\
    --backbone resnet50s8\
    --position_embedding sine --input_size 224\
    --dim_feedforward 2048 --set_cost_pts 5\
    --resume ./detr-r50-e632da11.pth\
    --train_index_file train_$(($1+4)).txt\
    --val_index_file val_$(($1+4)).txt\
    --output_dir ./work_dirs/train_ed4_query40_resnet50s8_l2_$(($1+4)) \
    --enc_layers 4 --dec_layers 4
