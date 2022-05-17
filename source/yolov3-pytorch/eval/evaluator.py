import torch
import numpy as np
import PIL, time
from train.loss import *
import csv
from terminaltables import AsciiTable

class Evaluator:
    def __init__(self, model, eval_data, eval_loader, device, hparam):
        self.model = model
        self.class_str = eval_data.class_str
        self.eval_loader = eval_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.gt_total = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False)
        self.tp = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False) #tp
        self.fn = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False) #fn
        self.fp = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False) #fp
        self.preds = None

    def run(self):
        predict_all = []
        gt_labels = []
        for i, batch in enumerate(self.eval_loader):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, targets, _ = batch
            
            input_img = input_img.to(self.device, non_blocking=True)
            
            gt_labels += targets[...,1].tolist()

            targets[...,2:6] = cxcy2minmax(targets[...,2:6])
            input_wh = [input_img.shape[3], input_img.shape[2]]
            targets[...,2] *= input_wh[0]
            targets[...,4] *= input_wh[0]
            targets[...,3] *= input_wh[1]
            targets[...,5] *= input_wh[1]
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output, conf_thres=0.1, iou_thres=0.5)
                
            predict_all += get_batch_statistics(best_box_list, targets, iou_threshold=0.5)
                
            if len(predict_all) == 0:
                print("no detection in eval data")
                return None
            if i % 100 == 0:
                print("-------eval {}th iter -----".format(i))
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*predict_all))]

        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, gt_labels)
        
        #print eval result
        if metrics_output is not None:
            precision, recall, ap, f1, ap_class = metrics_output
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, self.class_str[c], "%.5f" % ap[i]]]
            print(AsciiTable(ap_table).table)
        print("---- mAP {AP.mean():.5f} ----")
        
    
    def evaluate(self, preds, targets):
        #class mapping
        class8 = [0,1,2,3,4,5,6,7] #Car, Van, Truck, Ped, Ped_sitting, Cyclist, Tram, Misc
        class3 = [0,0,0,1,1,2,-1,-1] #Vehicle, Ped, Cyclist
        class2 = [0,0,0,1,1,1,-1,-1] #Vehicle, Ped
        
        #move the preds and targets from device to cpu
        preds = preds.detach().cpu()
        targets['bbox'] = targets['bbox'].detach().cpu()
        targets['cls'] = targets['cls'].detach().cpu()
        #remove ignore class GT data
        targets_cls_valid = None
        targets_bbox_valid = None

        #make mask tensor 
        pred_mask = torch.ones(preds.shape[0], requires_grad=False)
        gt_mask = torch.zeros(targets_cls_valid.shape[0], requires_grad=False) if targets_cls_valid is not None else torch.tensor([])

        target_num = targets_bbox_valid.shape[0] if targets_bbox_valid is not None else 0
        for i in range(target_num):
            tbox = targets_bbox_valid
            tcls = [class3[tcls] for tcls in targets_cls_valid]

            #change the target box format cxcywh to minmax
            cxcy2minmax(tbox[i])
            tbox[i,0] = tbox[i,0] * self.model.in_width
            tbox[i,2] = tbox[i,2] * self.model.in_width
            tbox[i,1] = tbox[i,1] * self.model.in_height
            tbox[i,3] = tbox[i,3] * self.model.in_height
    
            for j, (pbox, pobj_score, pcls_score, pcls_idx) in enumerate(zip(preds[:,:4], preds[:,4:5], preds[:,5:6], preds[:,6:])):
                #print(pbox.shape, pobj_score.shape, pcls_score.shape, pcls_idx.shape, tbox.shape, tcls.shape)

                pcls_idx = class3[int(pcls_idx.item())]
                if tcls[i] != pcls_idx or pred_mask[j] == 0 or tcls[i] == -1 or pcls_idx == -1:
                    continue
                
                iou_value = iou(tbox[i:i+1], pbox.unsqueeze(0), mode=1)
                
                #print("box {} {} / iou : {}".format(tbox[i:i+1], pbox, iou_value))

                if iou_value > 0.5:
                    gt_mask[i] = 1
                    pred_mask[j] = 0
        
        gt_matched = (gt_mask == 1).nonzero(as_tuple=True)
        gt_missed = (gt_mask == 0).nonzero(as_tuple=True)
        pred_false = (pred_mask == 1).nonzero(as_tuple=True)
        pred_true = (pred_mask == 0).nonzero(as_tuple=True)

        if gt_matched[0].nelement() != 0:
            for p in range(gt_matched[0].shape[0]):
                self.tp[targets_cls_valid[gt_matched[0][p]]] += 1
        if gt_missed[0].nelement() != 0:
            for p in range(gt_missed[0].shape[0]):
                self.fn[targets_cls_valid[gt_missed[0][p]]] += 1
        if pred_false[0].nelement() != 0:
            for p in range(pred_false[0].shape[0]):
                self.fp[int(preds[pred_false[0][p],6])] += 1
        if pred_true[0].nelement() != 0:
            for p in range(pred_true[0].shape[0]):
                if self.preds is None:
                    self.preds = preds[pred_true[0][p]].reshape(1,-1)
                else:
                    self.preds = torch.cat((self.preds, preds[pred_true[0][p]].reshape(1,-1)), dim = 0)
                    
        
    def evaluate_result(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        precision = precision.detach().numpy().tolist()
        recall = recall.detach().numpy().tolist()
        tp = self.tp.detach().numpy().tolist()
        fp = self.fp.detach().numpy().tolist()
        fn = self.fn.detach().numpy().tolist()
        
        self.class_list.remove('DontCare')
        data = {'name' : ['precision', 'recall', 'TP', 'FP', 'FN']}
        
        for i, cls in enumerate(self.class_list):
            data[cls] = [precision[i], recall[i], tp[i], fp[i], fn[i]]
        
        df = pd.DataFrame(data)
        print(df)
        
        df.to_csv('./evaluation.csv')

        
