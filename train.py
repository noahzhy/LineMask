import os
import math
import torch
import argparse
from tqdm import tqdm
from torch import optim
from torchsummary import summary

from utils.tool import *
from utils.datasets import *

from utils.metric import SegmentationMetric
from module.mobilenetv3_seg import MobileNetV3Seg
from module.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', type=str, default="configs/charts.yaml", help='.yaml config')
        parser.add_argument('--weight', type=str, default=None, help='.weight config')
        parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
        parser.add_argument('--ohem', action="store_true", default=False, help='use ohem loss')
        parser.add_argument('--aux-weight', type=float, default=0.4, help='auxiliary loss weight')

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "yaml file not exist"

        self.cfg = LoadYaml(opt.yaml)
        self.model = MobileNetV3Seg(self.cfg.category_num, backbone='mobilenetv3_small').to(device)

        if opt.weight is not None:
            print("load weight from:%s"%opt.weight)
            self.model.load_state_dict(torch.load(opt.weight, map_location=device))

        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))

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

        train_dataset = TensorDataset(self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, 10, True)
        val_dataset = TensorDataset(self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, 10, False)

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

                info = "Epoch:%d LR:%f IOU:%.4f Obj:%.4f Cls:%.4f Seg:%.4f Total:%.4f" % (
                        epoch, lr, iou, obj, cls, seg, total)
                pbar.set_description(info)
                batch_num += 1

            if epoch % 10 == 0 and epoch > 0:
                print("computer mAP...")
                # reset metric
                self.metric.reset()
                self.model.eval()
                for i, (image, target) in enumerate(self.val_loader):
                    image = image.to(self.device)
                    target = target.to(self.device)

                    with torch.no_grad():
                        outputs = model(image)
                    self.metric.update(outputs[0], target)
                    pixAcc, mIoU = self.metric.get()
                    logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

                new_pred = (pixAcc + mIoU) / 2
                mAP = round(new_pred, 4)
                torch.save(self.model.state_dict(), "checkpoints/weight_AP05_{:.4f}_{}.pth".format(mAP, epoch))

            self.scheduler.step()


if __name__ == "__main__":
    model = Model()
    model.train()
