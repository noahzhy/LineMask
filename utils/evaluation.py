import torch
import numpy as np
from tqdm import tqdm
from utils.tool import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoDetectionEvaluator():
    def __init__(self, names, device):
        self.device = device
        self.classes = []
        with open(names, 'r') as f:
            for line in f.readlines():
                self.classes.append(line.strip())
    
    def coco_evaluate(self, gts, preds):
        # Create Ground Truth
        coco_gt = COCO()
        coco_gt.dataset = {}
        coco_gt.dataset["images"] = []
        coco_gt.dataset["annotations"] = []
        k = 0
        for i, gt in enumerate(gts):
            for j in range(gt.shape[0]):
                k += 1
                coco_gt.dataset["images"].append({"id": i})
                coco_gt.dataset["annotations"].append({"image_id": i, "category_id": gt[j, 0],
                                                    "bbox": np.hstack([gt[j, 1:3], gt[j, 3:5] - gt[j, 1:3]]),
                                                    "area": np.prod(gt[j, 3:5] - gt[j, 1:3]),
                                                    "id": k, "iscrowd": 0})
                
        coco_gt.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_gt.createIndex()

        # Create preadict 
        coco_pred = COCO()
        coco_pred.dataset = {}
        coco_pred.dataset["images"] = []
        coco_pred.dataset["annotations"] = []
        k = 0
        for i, pred in enumerate(preds):
            for j in range(pred.shape[0]):
                k += 1
                coco_pred.dataset["images"].append({"id": i})
                coco_pred.dataset["annotations"].append({"image_id": i, "category_id": np.int32(pred[j, 0]),
                                                        "score": pred[j, 1], "bbox": np.hstack([pred[j, 2:4], pred[j, 4:6] - pred[j, 2:4]]),
                                                        "area": np.prod(pred[j, 4:6] - pred[j, 2:4]),
                                                        "id": k})

        coco_pred.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_pred.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP05 = coco_eval.stats[1]
        return mAP05

    # calculate miou and acc
    def cal_miou_acc(self, pred, mask):
        pred = pred.argmax(dim=1)
        mask = mask.argmax(dim=1)
        miou = []
        acc = []
        for i in range(pred.shape[0]):
            p = pred[i]
            m = mask[i]
            iou = (p & m).sum() / (p | m).sum()
            miou.append(iou)
            acc.append((p == m).sum() / p.numel())
        return torch.Tensor(miou).mean(), torch.Tensor(acc).mean()

    def compute_map(self, val_dataloader, model):
        gts, pts = [], []
        pbar = tqdm(val_dataloader)
        for i, (imgs, targets, mask) in enumerate(pbar):
            imgs = imgs.to(self.device).float() / 255.0

            with torch.no_grad():
                preds = model(imgs)
                output = handle_preds(preds, self.device, 0.001)

            N, _, H, W = imgs.shape
            for p in output:
                pbboxes = []
                for b in p:
                    b = b.cpu().numpy()
                    score = b[4]
                    category = b[5]
                    x1, y1, x2, y2 = b[:4] * [W, H, W, H]
                    pbboxes.append([category, score, x1, y1, x2, y2])
                pts.append(np.array(pbboxes))

            for n in range(N):
                tbboxes = []
                for t in targets:
                    if t[0] == n:
                        t = t.cpu().numpy()
                        category = t[1]
                        bcx, bcy, bw, bh = t[2:6] * [W, H, W, H]
                        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
                        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
                        tbboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(tbboxes))
        
        mIOU, mACC = self.cal_miou_acc(preds[1], mask)
        print("mIOU: %f, mACC: %f" % (mIOU, mACC))
        mAP05 = self.coco_evaluate(gts, pts)
        return mAP05
