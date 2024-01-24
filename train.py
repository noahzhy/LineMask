import os
import math
import argparse

import torch
from tqdm import tqdm
from torch import optim
from torchsummary import summary
from torch.nn import functional as F

from utils.tool import *
from module.loss import *
from utils.datasets import *
from module.unet import UNet_CBAM_Deeper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cpu mode
# device = torch.device("cpu")

class Model:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', type=str, default="configs/charts.yaml", help='.yaml config')
        parser.add_argument('--weight', type=str, default=None, help='.weight config')
        parser.add_argument('--optimizer', type=str, default="adam", help='optimizer')

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "yaml file not exist"

        self.cfg = LoadYaml(opt.yaml)
        self.model = UNet_CBAM_Deeper().to(device)

        if opt.weight is not None:
            print("load weight from:%s"%opt.weight)
            self.model.load_state_dict(torch.load(opt.weight, map_location=device))

        if opt.optimizer == "sgd":
            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.cfg.learn_rate,
                momentum=0.949,
                weight_decay=0.0005)
        elif opt.optimizer == "adam":
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                lr=self.cfg.learn_rate,
                weight_decay=0.0005)

        print("use {} optimizer".format(self.optimizer.__class__.__name__))

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.milestones,
            gamma=0.1)

        self.criterion = DiscriminativeLoss(usegpu=True)
        self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        self.mse_loss = torch.nn.MSELoss().to(device)
        self.dice_loss = DiceLoss()
        # self.metric = SegmentationMetric(self.cfg.category_num)

        train_dataset = ChartDataset(self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, 10, False)
        val_dataset = ChartDataset(self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, 10, False)

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=32,
            drop_last=False,
            persistent_workers=True)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=32,
            drop_last=True,
            persistent_workers=True)

    def train(self):
        batch_num = 0
        loss = 0
        print('Starting training for %g epochs...' % self.cfg.end_epoch)
        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for images, sem_mask, ins_mask, num_objs in pbar:
                images = images.to(device)
                sem_mask = sem_mask.to(device)
                ins_mask = ins_mask.to(device)
                num_objs = num_objs.to(device)

                pred_sem_mask, pred_ins_mask, pred_num_objs = self.model(images)
                loss = self.criterion(pred_ins_mask, ins_mask, num_objs)
                loss += self.dice_loss(pred_sem_mask, sem_mask)
                num_objs = num_objs.unsqueeze(1).float()
                loss += self.mse_loss(pred_num_objs, num_objs/10)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for g in self.optimizer.param_groups:
                    if self.cfg.warmup:
                        warmup_num =  5 * len(self.train_dataloader)
                        if batch_num <= warmup_num:
                            scale = math.pow(batch_num/warmup_num, 4)
                            g['lr'] = self.cfg.learn_rate * scale

                    lr = g["lr"]

                info = "Epoch:%d LR:%f Loss:%.4f" % (epoch, lr, loss)
                pbar.set_description(info)
                batch_num += 1

            if epoch % 10 == 0 and epoch > 0:
                print("computer mAP...")
                # pbar = tqdm(self.val_dataloader)
                # for image, sem, ins, cnt in pbar:
                #     image = image.to(device)
                #     sem = sem.to(device)
                #     ins = ins.to(device)
                #     cnt = cnt.to(device)

                #     with torch.no_grad():
                #         # model to eval mode
                #         model = self.model.eval()
                #         pred_sem, pred_ins, pred_cnt = model(image)
                #         print(pred_sem.shape, pred_ins.shape, pred_cnt.shape)

                # # compute mAP
                # dice_score = dice_score / len(self.val_dataloader)

                file_name = "{:d}_{:.4f}.pth".format(epoch, loss)
                torch.save(self.model.state_dict(), "checkpoints/" + file_name)

            self.scheduler.step()


if __name__ == "__main__":
    model = Model()
    model.train()
