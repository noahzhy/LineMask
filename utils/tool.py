import yaml
import torch
import torchvision
import numpy as np
from PIL import Image

# 解析yaml配置文件
class LoadYaml:
    def __init__(self, path):
        with open(path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]

        self.warmup = data["TRAIN"]["WARMUP"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]
        
        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]
        
        self.category_num = data["MODEL"]["NC"]
        
        print("Load yaml sucess...")

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def handle_preds(preds, device):
    counter = preds[-1]
    counter = counter.cpu().detach().numpy().astype('float32')*10
    counter = round(counter[0][0])
    print("counter:", counter)
    pred_seg = preds[-2]
    print(pred_seg.shape)
    lines = np.zeros_like(pred_seg[:, 0, :, :].detach().cpu().numpy())
    # min and max
    # sum to one channel
    for i in range(counter):
        # # check min and max value
        # print(pred_seg[:, i, :, :].min(), pred_seg[:, i, :, :].max())
        # if bigger than 0.5, set to 1, else set to 0
        lines += np.where(pred_seg[:, i, :, :].cpu() > 0.5, 1, 0)

    # save first two channel as png file
    for i in range(3):
        tmp = np.where(pred_seg[:, i, :, :].cpu() > 0.5, 1, 0)
        tmp = tmp.astype('uint8') * 255
        # (1, 352, 352) => (352, 352)
        tmp = tmp.squeeze(0)
        tmp = Image.fromarray(tmp)
        tmp.save('pred_seg_%d.png'%i)

    # sum to one channel
    pred_seg = lines.sum(axis=0)
    pred_seg = pred_seg.astype('uint8') * 25
    pred_seg = Image.fromarray(pred_seg)
    pred_seg.save('pred_seg.png')
