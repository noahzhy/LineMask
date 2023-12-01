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


def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    pred_det, pred_seg = preds
    # # get one channel
    # lines = np.zeros_like(pred_seg[:, 0, :, :].detach().cpu().numpy())
    # # min and max
    # # sum to one channel
    # for i in range(pred_seg.shape[1]):
    #     # # check min and max value
    #     # print(pred_seg[:, i, :, :].min(), pred_seg[:, i, :, :].max())
    #     # if bigger than 0.5, set to 1, else set to 0
    #     lines += np.where(pred_seg[:, i, :, :].cpu() > 0.5, 1, 0)

    # # save first two channel as png file
    # for i in range(3):
    #     tmp = np.where(pred_seg[:, i, :, :].cpu() > 0.5, 1, 0)
    #     tmp = tmp.astype('uint8') * 255
    #     # (1, 352, 352) => (352, 352)
    #     tmp = tmp.squeeze(0)
    #     tmp = Image.fromarray(tmp)
    #     tmp.save('pred_seg_%d.png'%i)

    # # sum to one channel
    # pred_seg = lines.sum(axis=0)
    # pred_seg = pred_seg.astype('uint8') * 25
    # pred_seg = Image.fromarray(pred_seg)
    # pred_seg.save('pred_seg.png')

    total_bboxes, output_bboxes  = [], []

    N, C, H, W = pred_det.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = pred_det.permute(0, 2, 3, 1)
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    preg = pred[:, :, :, 1:5]
    pcls = pred[:, :, :, 5:]
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(dim=-1)

    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid() 
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H*W, 6)
    total_bboxes.append(bboxes)
        
    batch_bboxes = torch.cat(total_bboxes, 1)

    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])

        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))

    return output_bboxes
