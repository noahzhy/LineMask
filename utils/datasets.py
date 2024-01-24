import os
import random

import cv2
import torch
import numba as nb
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from utils.toStyle import add_noise, texture
from toStyle import add_noise, texture


def random_crop(image, bg_mask, mask):
    height, width, _ = image.shape
    # random crop imgage
    cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

    img = image[cy:cy + ch, cx:cx + cw]
    bg = bg_mask[cy:cy + ch, cx:cx + cw]
    ins = mask[cy:cy + ch, cx:cx + cw]
    return img, bg, ins


# @nb.jit(nopython=True)
def random_narrow(image, bg_mask, mask):
    height, width, _ = image.shape
    # random narrow
    cw, ch = random.randint(width, int(width * 1.25)), random.randint(height, int(height * 1.25))
    cx, cy = random.randint(0, cw - width), random.randint(0, ch - height)

    background = np.ones((ch, cw, 3), np.float32) * 128
    background[cy:cy + height, cx:cx + width] = image

    bg_mask = np.zeros((ch, cw), dtype=np.int64)
    bg_mask[cy:cy + height, cx:cx + width] = bg_mask

    _mask = np.zeros((ch, cw, mask.shape[2]), dtype=np.int64)
    _mask[cy:cy + height, cx:cx + width] = mask

    return background, bg_mask, _mask


def collate_fn(batch):
    img, sem_mask, ins_mask, cnt = zip(*batch)
    return torch.stack(img), torch.stack(sem_mask), torch.stack(ins_mask), torch.stack(cnt)


# function to gen line mask via given [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
def gen_line_mask(lines, T=10, img_width=512, img_height=512, line_width=3):
    masks = np.zeros((img_height, img_width, T), dtype=np.int64)
    for idx, line in enumerate(lines):
        x = line[:][0::2] * img_width
        y = line[:][1::2] * img_height
        # draw line
        img = Image.fromarray(masks[:, :, idx], mode='L')
        ImageDraw.Draw(img).line(list(zip(x, y)), fill=255, width=line_width)
        masks[:, :, idx] = img

    return masks


class ChartDataset():
    def __init__(self, path, img_width=768, img_height=768, line_max=10, aug=False):
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
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([0] + l)

            label = np.array(label[:6], dtype=np.float32)
            height, width, _ = img.shape
            num_objs = len(label)
            mask = gen_line_mask(label[:, 6:], T=self.line_max, img_width=width, img_height=height)
            # to one channel
            bg_mask = np.sum(mask[:, :, :num_objs], axis=2)

        else:
            raise Exception("%s is not exist" % label_path)

        if self.aug:
            if random.randint(1, 10) % 2 == 0:
                img, bg_mask, mask = random_narrow(img, bg_mask, mask)
            else:
                img, bg_mask, mask = random_crop(img, bg_mask, mask)

        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        img = add_noise(texture(img, sigma=3, turbulence=4), sigma=3)
        img = img.transpose(2,0,1)

        bg_mask = cv2.resize(bg_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        # bg_mask = bg_mask.transpose(2,0,1)

        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2,0,1)

        # img and mask to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0
        bg_mask = torch.from_numpy(bg_mask//255).long()
        bg_mask = mask2onehot(bg_mask, 2)
        mask = torch.from_numpy(mask).float() / 255.0

        return img, bg_mask, mask, torch.tensor(num_objs).long()

    def __len__(self):
        return len(self.data_list)


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = torch.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=torch.int64)
    for i in range(num_classes):
        _mask[i, :, :] = mask == i

    return _mask


if __name__ == "__main__":
    data = ChartDataset("configs/train.txt", aug=False)
    _len = len(data)
    _pick_idx = random.randint(0, _len)
    data = data.__getitem__(_pick_idx)
    img, bg_mask, mask, num_objs = data

    print("img shape:", img.shape)
    print("bg_mask shape:", bg_mask.shape)
    print("mask shape:", mask.shape)
    print("num_objs:", num_objs)

    fig = plt.figure(figsize=(30, 10))
    fig.tight_layout()

    for i in range(5):
        if i==0:
            plt.subplot(1, 6, i+1)
            # show original image on first subplot
            img = img.numpy().transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis('off')
            continue

        # show it on plt in one figure
        plt.subplot(1, 6, i+1)
        plt.imshow(mask[i-1, :, :], cmap='gray')
        plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(img)
    plt.savefig("data.png")
