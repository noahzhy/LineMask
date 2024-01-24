import os
import math
import argparse

import torch
from tqdm import tqdm
from torch import optim
from torchsummary import summary
from torch.nn import functional as F

from utils.tool import *
from utils.datasets import *
from utils.metric import *
from module.mobilenetv3_seg import MobileNetV3Seg
from module.unet import UNet
from module.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# cpu mode
device = torch.device("cpu")

class Model:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', type=str, default="configs/charts.yaml", help='.yaml config')
        parser.add_argument('--weight', type=str, default=None, help='.weight config')
        parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
        parser.add_argument('--ohem', action="store_true", default=True, help='use ohem loss')
        parser.add_argument('--aux-weight', type=float, default=0.4, help='auxiliary loss weight')

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "yaml file not exist"

        self.cfg = LoadYaml(opt.yaml)
        # self.model = MobileNetV3Seg(self.cfg.category_num, backbone='mobilenetv3_large').to(device)
        self.model = UNet(3, self.cfg.category_num).to(device)

        if opt.weight is not None:
            print("load weight from:%s"%opt.weight)
            self.model.load_state_dict(torch.load(opt.weight, map_location=device))

        # self.optimizer = optim.SGD(params=self.model.parameters(),
        #                            lr=self.cfg.learn_rate,
        #                            momentum=0.949,
        #                            weight_decay=0.0005,
        # #                            )
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.cfg.learn_rate,
                                    weight_decay=0.0005,
                                    )
        print("use {} optimizer".format(self.optimizer.__class__.__name__))

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.cfg.milestones,
                                                        gamma=0.1)

        if opt.ohem:
            min_kept = int(self.cfg.batch_size // opt.num_gpus * self.cfg.input_width ** 2 // 16)
            self.criterion = MixSoftmaxCrossEntropyOHEMLoss(opt.aux_weight, min_kept=min_kept, ignore_index=-1).to(device)
        else:
            self.criterion = MixSoftmaxCrossEntropyLoss(opt.aux_weight, ignore_index=-1).to(device)

        self.metric = SegmentationMetric(self.cfg.category_num)

        train_dataset = ChartDataset(self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, 10, False)
        val_dataset = ChartDataset(self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, 10, False)

        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.cfg.batch_size,
                                                          shuffle=False,
                                                          collate_fn=collate_fn,
                                                          num_workers=32,
                                                          drop_last=False,
                                                          persistent_workers=True
                                                          )

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.cfg.batch_size,
                                                            shuffle=True,
                                                            collate_fn=collate_fn,
                                                            num_workers=32,
                                                            drop_last=True,
                                                            persistent_workers=True
                                                            )

    def train(self):
        batch_num = 0
        print('Starting training for %g epochs...' % self.cfg.end_epoch)
        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

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
                for i, (image, target) in enumerate(self.val_dataloader):
                    image = image.to(device)
                    mask_true = target.to(device)
                    # as int64
                    mask_true = mask_true.long()

                    with torch.no_grad():
                        mask_pred = self.model(image)
                        # convert to one-hot format
                        mask_true = F.one_hot(mask_true, self.cfg.category_num).permute(0, 3, 1, 2).float()
                        mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.cfg.category_num).permute(0, 3, 1, 2).float()
                        # compute the Dice score, ignoring background
                        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # compute mAP
                dice_score = dice_score / len(self.val_dataloader)

                file_name = "{:d}_{:.4f}.pth".format(epoch, dice_score)
                torch.save(self.model.state_dict(), "checkpoints/" + file_name)

            self.scheduler.step()


if __name__ == "__main__":
    model = Model()
    model.train()
