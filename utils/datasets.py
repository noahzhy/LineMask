import os
import random

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from utils.toStyle import add_noise, texture



def old_style(img, bbox, mask):
    img = add_noise(texture(img, sigma=4, turbulence=4), sigma=5)
    height, width, _ = img.shape
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx = bx/width
        by = by/height
        bw = bw/width
        bh = bh/height

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)
    return img, output, mask


def do_nothing(image, boxes, mask):
    height, width, _ = image.shape
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx = bx/width
        by = by/height
        bw = bw/width
        bh = bh/height

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)
    return image, output, mask


def random_crop(image, boxes, mask):
    height, width, _ = image.shape
    # random crop imgage
    cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape

    roi_mask = mask[cy:cy + ch, cx:cx + cw]
    
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx - cx)/roi_w, (by - cy)/roi_h
        bw, bh = bw/roi_w, bh/roi_h

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)

    return roi, output, roi_mask


def random_narrow(image, boxes, mask):
    height, width, _ = image.shape
    # random narrow
    cw, ch = random.randint(width, int(width * 1.25)), random.randint(height, int(height * 1.25))
    cx, cy = random.randint(0, cw - width), random.randint(0, ch - height)

    background = np.ones((ch, cw, 3), np.uint8) * 128
    background[cy:cy + height, cx:cx + width] = image

    bg_mask = np.zeros((ch, cw, mask.shape[2]), dtype=np.int64)
    bg_mask[cy:cy + height, cx:cx + width] = mask

    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx + cx)/cw, (by + cy)/ch
        bw, bh = bw/cw, bh/ch

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)

    return background, output, bg_mask


def collate_fn(batch):
    img, label, mask = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0), torch.stack(mask)


# function to gen line mask via given [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
def gen_line_mask(lines, T=10, img_width=512, img_height=512, line_width=5):
    masks = np.zeros((img_height, img_width, T), dtype=np.int64)
    for idx, line in enumerate(lines):
        x = line[:][0::2] * img_width
        y = line[:][1::2] * img_height
        # draw line
        img = Image.fromarray(masks[:, :, idx], mode='L')
        ImageDraw.Draw(img).line(list(zip(x, y)), fill=255, width=line_width)
        masks[:, :, idx] = img

    return masks


class TensorDataset():
    def __init__(self, path, img_width=512, img_height=512, line_max=10, aug=False):
        assert os.path.exists(path), "%s file is not exist" % path

        self.aug = aug
        self.path = path
        self.data_list = []
        self.img_width = img_width
        self.img_height = img_height
        self.line_max = line_max
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']

        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label_path = img_path.split(".")[0] + ".txt"

        img = cv2.imread(img_path)
        # # to hsv color space
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([0] + l)

            label = np.array(label[:6], dtype=np.float32)
            height, width, _ = img.shape
            mask = gen_line_mask(label[:, 6:], T=self.line_max, img_width=width, img_height=height)
        else:
            raise Exception("%s is not exist" % label_path)


        if self.aug:
            if random.randint(1, 10) % 2 == 0:
                img, label, mask = random_narrow(img, label, mask)
            else:
                img, label, mask = random_crop(img, label, mask)
        else:
            img, label, mask = do_nothing(img, label, mask)

        # print(img.shape, label.shape, mask.shape, img_path)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        img = add_noise(texture(img, sigma=2, turbulence=4), sigma=3) / 255.0
        img = img.transpose(2,0,1)

        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2,0,1)

        return torch.from_numpy(img), torch.from_numpy(label), torch.from_numpy(mask//255).long()

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    data = TensorDataset("configs/train_line.txt", aug=True)
    img, label, mask = data.__getitem__(0)
    print(img.shape, label.shape, mask.shape)

    def random_lines(kpts_num=10, line_num=3):
        lines = []
        for i in range(line_num):
            x = np.arange(kpts_num)
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = np.random.randn(kpts_num)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            # zip them as [x1, y1, x2, y2, ...]
            line = np.array(list(zip(x, y))).flatten()
            lines.append(line)
        return lines

    # test gen line mask
    # lines = np.array(random_lines())
    # mask = gen_line_mask(lines)
    # mask_len = np.sum(mask, axis=2)
    fig = plt.figure()
    # tight_layout
    fig.tight_layout()

    for i in range(5):
        if i==0:
            plt.subplot(1, 6, i+1)
            # show original image on first subplot
            # convert tensor to numpy array
            img = img.numpy().transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis('off')
            continue

        # show it on plt in one figure
        plt.subplot(1, 6, i+1)
        plt.imshow(mask[i-1, :, :], cmap='gray')
        plt.axis('off')

    # draw bbox on image
    for box in label:
        index, category = box[0], box[1]
        bx, by = box[2] * img.shape[1], box[3] * img.shape[0]
        bw, bh = box[4] * img.shape[1], box[5] * img.shape[0]
        # draw bbox
        cv2.rectangle(img, (int(bx-bw/2), int(by-bh/2)), (int(bx+bw/2), int(by+bh/2)), (0, 255, 0), 2)

    plt.subplot(1, 6, 6)
    plt.imshow(img)
    plt.axis('off')

    plt.show()
