import os
import cv2
import time
import argparse

import torch
from utils.tool import *
from module.unet import UNet_CBAM_Deeper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="configs/charts.yaml", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')
    parser.add_argument('--img', type=str, default='data/002.png', help='The path of test image')
    parser.add_argument('--cpu', action="store_true", default=True, help='Run on cpu')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "please check yaml file path"
    assert os.path.exists(opt.img), "please check image file path"

    if opt.weight is None:
        # find latest weight in checkpoint dir
        weight_list = os.listdir("checkpoints")
        weight_list.sort(key=lambda fn: os.path.getmtime(os.path.join("checkpoints", fn)))
        opt.weight = os.path.join("checkpoints", weight_list[-1])
        print("use latest weight:%s"%opt.weight)

    device = torch.device("cpu") if opt.cpu else torch.device("cuda")
    cfg = LoadYaml(opt.yaml)
    print(cfg)

    # load model
    print("load weight from:%s"%opt.weight)
    model = UNet_CBAM_Deeper(usegpu=False).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    #sets the module in eval node
    model.eval()

    # 数据预处理
    img = cv2.imread(opt.img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg.input_width, cfg.input_height), interpolation = cv2.INTER_LINEAR) 
    img = img.reshape(1, cfg.input_height, cfg.input_width, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    img_org = img.clone()
    # save as png file
    img_org = img_org.squeeze(0)
    img_org = img_org.cpu().numpy()
    img_org = img_org.transpose(1, 2, 0)
    img_org = img_org * 255
    img_org = img_org.astype('uint8')
    cv2.imwrite("input.png", img_org)

    # get prediction
    preds = model(img)
    # post process
    output = handle_preds(preds, device)

    # cv2.imwrite("result.png", ori_img)
