from .speed import SpeedTrain, make_transforms


def build_dataset(args, train=True):
    if train:
        ann_file = args.train_ann_file
        index_file = args.train_index_file
        img_dir = args.train_img_dir
    else:
        ann_file = args.val_ann_file
        index_file = args.val_index_file
        img_dir = args.val_img_dir

    input_size = args.input_size
    transforms = make_transforms(
        train=train,
        img_size=input_size)
    return SpeedTrain(
        ann_file, index_file, img_dir,
        input_size, train, transforms
    )
