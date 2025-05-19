python main.py --output_dir train1 --num_workers 4 --lr 0.0001 --resume train1/checkpoint0099.pth \
    --val_ann_file wz_real.json --val_index_file real.txt --val_img_dir images/real --gt_file real.json \
    --eval --num_queries 100
