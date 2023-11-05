import os, sys, glob, random
from shutil import copyfile

from PIL import Image, ImageDraw
import tqdm

# func txt2yolo

def txt2yolo(txt_path, img_path, tar_dir="data"):
    # load txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # x, y, w, h, color

    # get image size
    img = Image.open(img_path)
    _w, _h = img.size
    # new txt file
    new_lines = []
    for line in lines:
        line = line.strip().split(',')
        # x, y, w, h, color -> cx, cy, w, h, color
        line = [float(x) for x in line[0:4]]
        # x, y, w, h -> cx, cy, w, h
        cx = line[0] / _w + line[2] / _w / 2
        cy = line[1] / _h + line[3] / _h / 2
        w = line[2] / _w
        h = line[3] / _h

        if h == 0 or w == 0:
            continue

        line = ' '.join([str(x) for x in [0, cx, cy, w, h]])
        new_lines.append(line)

    # save new txt file
    with open(os.path.join(tar_dir, os.path.basename(txt_path)), 'w') as f:
        f.write('\n'.join(new_lines))


# show bbox on image
def show_bbox(img_path, txt_path):
    # load txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # x, y, w, h, color
    lines = [line.strip().split(' ') for line in lines]
    # draw
    img = Image.open(img_path)
    print(img.size)
    # size
    _w, _h = img.size
    draw = ImageDraw.Draw(img)
    for line in lines:
        # split by ' '
        # to int 
        line = [float(x) for x in line]
        c, cx, cy, w, h = line
        # to x y w h
        x = cx - w / 2
        y = cy - h / 2
        x = x * _w
        y = y * _h
        w = w * _w
        h = h * _h
        # to int
        x, y, w, h = int(x), int(y), int(w), int(h)
        draw.rectangle((x, y, x+w, y+h), outline=(255, 0, 0), width=1)
    img.show()




# main
if __name__ == "__main__":
    txt_list = glob.glob("h_data/*.txt")

    for txt_path in tqdm.tqdm(txt_list):
        img_path = txt_path.replace(".txt", ".png")

        # copy png file to tar_dir
        copyfile(img_path, os.path.join("data", os.path.basename(img_path)))
        txt2yolo(txt_path, img_path)
        # replace dir name
        dir_name = os.path.dirname(img_path)
        # replace dir name
        txt_path = txt_path.replace(dir_name, "data")
        # show_bbox(img_path, txt_path)

        # break


    # # get txt list
    # txt_list = glob.glob("data/*.txt")
    # # # random pick one to show
    # txt_path = random.choice(txt_list)
    # img_path = txt_path.replace(".txt", ".png")
    # show_bbox(img_path, txt_path)