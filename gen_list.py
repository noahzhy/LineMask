# generate train and val list via given txt list
#
import os, glob, random, shutil


def gen_list(txt_dir, tar_dir="data"):
    # factor
    train_factor = 0.8
    val_factor = 0.2
    # find all txt via txt_dir
    txt_paths = glob.glob(os.path.join(txt_dir, '*.png'))
    # shuffle
    random.shuffle(txt_paths)
    # split
    train_num = int(len(txt_paths) * train_factor)
    val_num = int(len(txt_paths) * val_factor)
    # train
    train_paths = txt_paths[0:train_num]
    # val
    val_paths = txt_paths[train_num:]
    # save
    with open(os.path.join(tar_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_paths))

    with open(os.path.join(tar_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_paths))


# main
if __name__ == "__main__":
    # txt dir
    txt_dir = "data"
    # tar dir
    tar_dir = "configs"
    # gen list
    gen_list(txt_dir, tar_dir)
