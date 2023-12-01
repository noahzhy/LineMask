# copy file via given list of files to destination directory
import os, glob, shutil


def copy_files(file_list, dst_dir):
    for file in file_list:
        print(file)
        shutil.copy(file, dst_dir)
        # get txt file
        txt_file = file.replace('png', 'txt')
        shutil.copy(txt_file, dst_dir)



# main
if __name__ == '__main__':
    txt_path = r"configs/val_line.txt"
    dst_dir = r"Z:/old_data"

    lines = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    copy_files(lines, dst_dir)
    
