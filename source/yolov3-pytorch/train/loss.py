import enum
from wsgiref.simple_server import demo_app
import torch
from torch._C import device
import torch.nn as nn
from util.tools import *
import sys
import math


class YoloLoss(nn.Module):
    
    def __init__(self, device, num_class, ignore_cls):
        super(YoloLoss, self).__init__()
        self.device = device
        self.mseloss = nn.MSELoss(reduction='sum').to(device)
        self.bceloss = nn.BCELoss(reduction='sum').to(device)
        self.bcellogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device)).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.num_class = num_class
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 0.5
        self.lambda_iou = 0.05
        self.ignore_cls = ignore_cls
        
    def compute_loss(self, pred, targets = None, yolo_layers = None, tmp_img = None):
        loss = torch.zeros(1,device=self.device)
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        lcls3 = torch.zeros(1, device=self.device)
        tcls, tbox, tindices, tanchors = self.get_targets_v2(targets, yolo_layers, preds = pred, tmp_img=tmp_img)

        #for yolo_layers
        for pidx, pout in enumerate(pred):
            batch_id, anc_id, gy, gx = tindices[pidx]
            
            tobj = torch.zeros_like(pout[...,0], device=self.device)#target object
            
            num_targets = batch_id.shape[0]
            if num_targets:
                ps = pout[batch_id, anc_id, gy, gx]
                
                pxy = torch.sigmoid(ps[...,0:2])
                pwh = torch.exp(ps[...,2:4]) * tanchors[pidx]
                pbox = torch.cat((pxy,pwh),1)
                iou = bbox_iou(pbox.T, tbox[pidx], x1y1x2y2=False, CIoU=True)
                
                lbox += (1 - iou).mean()
                
                tobj[batch_id, anc_id, gy, gx] = iou.detach().clamp(0).type(tobj.dtype)
                
                if ps.size(1) - 5 > 1:
                    #all class loss
                    t = torch.zeros_like(ps[...,5:], device=self.device)
                    t[range(num_targets), tcls[pidx]] = 1
                    lcls += self.bcellogloss(ps[:,5:], t)

                    # #3 class loss
                    # t3 = torch.zeros_like(ps[...,5:8], device=self.device)
                    # for tc in range(tcls[pidx].shape[0]):
                    #     if tcls[pidx][tc].item() < 3:
                    #         t3[tc, 0] = 1
                    #     elif tcls[pidx][tc].item() >= 3 and tcls[pidx][tc].item() < 6:
                    #         t3[tc, 1] = 1
                    #     else:
                    #         t3[tc, 2] = 1
                    # p_cls3 = torch.cat((torch.sum(ps[...,5:8], 1).view(-1,1) / 3, torch.sum(ps[...,8:11], 1).view(-1,1) / 3, torch.sum(ps[...,11:], 1).view(-1,1) / 2),dim=1)
                    # lcls3 += self.bcellogloss(p_cls3, t3)
            lobj += self.bcellogloss(pout[...,4], tobj)
            # sys.exit(1)
        # TODO 5 : add loss method
        lcls *= 0.5
        # lcls3 *= 0.05
        lobj *= 1.0
        lbox *= 0.05
        
        loss = lcls + lobj + lbox
        loss_list = [loss.item(), lobj.item(), lcls.item(), lbox.item()]
        return loss, loss_list

    
    def get_targets(self, targets, anchors, nw, nh, lw, lh, stride, ignore_thresh, tmp_img = None, preds = None):
        bs = len(targets)
        
        mask = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        noobj_mask = torch.ones(bs, len(anchors), lh, lw, requires_grad=False)
        tx = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        ty = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        tw = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        th = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        tconf = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        tcls = torch.zeros(bs, len(anchors), lh, lw, self.num_class, requires_grad=False)
        pred_ious = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False).to(self.device)
        
        scale_w = nw / lw
        scale_h = nh / lh
        
        for b in range(bs):
            target_box = targets[b]['bbox']
            target_cls = targets[b]['cls']
            target_occ = targets[b]['occ']
            target_trunc = targets[b]['trunc']

            #if target object dont exist
            if target_box is None and target_cls is None:
                continue
            
            for t in range(target_box.shape[0]):
                #ignore Dontcare objects and occluded objects
                if int(target_cls[t]) == self.ignore_cls or target_occ[t] > 1 or target_trunc[t] > 0.5:
                    continue
                #get box position relative to grid(anchor)
                gx = target_box[t,0] * lw
                gy = target_box[t,1] * lh
                gw = target_box[t,2] * nw
                gh = target_box[t,3] * nh
                #get index of grid
                gi = int(gx)
                gj = int(gy)

                # for anc in range(3):
                #     show_pred_box = torch.zeros(4).to(self.device)
                #     show_pred_box[0] = (preds[b, anc, gj, gi, 0] + gi) * scale_w
                #     show_pred_box[1] = (preds[b, anc, gj, gi, 1] + gj) * scale_h
                #     show_pred_box[2] = (torch.exp(preds[b, anc, gj, gi, 2]) * anchors[anc][0])
                #     show_pred_box[3] = (torch.exp(preds[b, anc, gj, gi, 3]) * anchors[anc][1])
                #     drawBox(tmp_img.cpu().detach().numpy()[0],
                #             [show_pred_box], cls = target_cls)
                    #torch.tensor([gx * scale_w, gy * scale_h, gw, gh]),
                    #         torch.tensor([gi * scale_w,gj * scale_h, anchors[anc][0], anchors[anc][1]]),

                #make gt_box shape
                gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)
                #make box shape of each anchor 
                anchor_shape = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)),
                                                                 np.array(anchors)),1))
                
                #get iou between gt and anchor_shape
                anc_iou = iou(gt_box, anchor_shape, mode = 0)

                # print("tbox :",target_box[t,0:4], gx, gy, gw, gh, "gbox : ", gt_box, "anc :", anchor_shape)
                # print("anchor iou :", anc_iou)
                # anc_iou_b = bbox_iou(gt_box, anchor_shape, x1y1x2y2 = False)

                #mask to zero to ignore the prediction if larger than ignore_threshold
                noobj_mask[b, anc_iou > ignore_thresh, gj, gi] = 0
                #get best anchor box 
                best_box = np.argmax(anc_iou)
                positive_box = anc_iou > ignore_thresh
                if anc_iou[best_box] >= 0.0:
                    pred_box = torch.zeros(4).to(self.device)
                    pred_box[0] = (preds[b, best_box, gj, gi, 0] + gi) * scale_w
                    pred_box[1] = (preds[b, best_box, gj, gi, 1] + gj) * scale_h
                    pred_box[2] = (torch.exp(preds[b, best_box, gj, gi, 2]) * anchors[best_box][0])
                    pred_box[3] = (torch.exp(preds[b, best_box, gj, gi, 3]) * anchors[best_box][1])
                    gt_box = torch.tensor([gx * scale_w, gy * scale_h, gw , gh]).to(self.device)
                    
                    pred_iou = iou(gt_box.unsqueeze(0), pred_box.unsqueeze(0), mode = 0, device = self.device)
                    pred_ious[b, best_box, gj, gi] = 1 - pred_iou

                    print("pred_iou : ", pred_iou," / pred_conf: " ,preds[b, best_box, gj, gi, 4], " / conf>0.8: ", torch.sum(preds[:, :, :, :, 4] > 0.8).item())
                    #mask
                    mask[b, best_box, gj, gi] = 1
                    
                    #if IOU between pred and gt is under 0.1, negative box
                    #if pred_iou < 0.1:
                    #mask[b, best_box, gj, gi] = 0
                    #pred_ious[b, best_box, gj, gi] = 0
                    
                    #coordinate and width,height
                    tx[b, best_box, gj, gi] = gx - gi
                    ty[b, best_box, gj, gi] = gy - gj
                    tw[b, best_box, gj, gi] = math.log(gw/anchors[best_box][0] + 1e-16) #np.log(gw/(anchors[best_box][0]/stride[0]) + 1e-16)
                    th[b, best_box, gj, gi] = math.log(gh/anchors[best_box][1] + 1e-16) #np.log(gh/(anchors[best_box][1]/stride[1]) + 1e-16)
                    # print("Target : " ,tx[b, best_box, gj, gi], ty[b, best_box, gj, gi], tw[b, best_box, gj, gi], th[b, best_box, gj, gi])

                    #objectness
                    tconf[b, best_box, gj, gi] = pred_iou.detach().clamp(0).type(tconf.dtype)
                    print(pred_iou.detach().clamp(0).type(tconf.dtype))

                    #class confidence
                    tcls[b, best_box, gj, gi, int(target_cls[t])] = 1
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, pred_ious
    
    def get_targets_v2(self, targets, yolo_layer, preds = None, tmp_img = None):
        
        num_anc = 3
        num_t = targets.shape[0]
        tcls, tboxes, indices, anch = [], [], [], []
        
        if torch.equal(targets,torch.zeros(6).to(targets.device)):
            return tcls, tboxes, indices, anch
        
        gain = torch.ones(7, device=self.device)
        
        ai = torch.arange(num_anc, device=targets.device).float().view(num_anc, 1).repeat(1, num_t)
        targets = torch.cat((targets.repeat(num_anc, 1, 1), ai[:, :, None]), 2).to(self.device)
        
        for yi, yl in enumerate(yolo_layer):
            anchors = yl.anchor / yl.stride
            gain[2:6] = torch.tensor(preds[yi].shape)[[3, 2, 3, 2]]  # xyxy gain
            
            t = targets * gain
            if num_t:
                # Calculate ration between anchor and target box for both width and height
                r = t[:, :, 4:6] / anchors[:, None]
                # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
                j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
                # Only use targets that have the correct ratios for their anchors
                # That means we only keep ones that have a matching anchor and we loose the anchor dimension
                # The anchor id is still saved in the 7th value of each target
                t = t[j]
            else:
                t = targets[0]
                
            b, c = t[:, :2].long().T
            
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            
            gij = gxy.long()
            
            gi, gj = gij.T
            
            #anchor index
            a = t[:, 6].long()
            
            #add index list
            indices.append((b, a, gj.clamp_(0,gain[3]-1), gi.clamp_(0,gain[2]-1)))
            
            #add target box
            tboxes.append(torch.cat((gxy-gij, gwh),1))
            
            #add anchor
            anch.append(anchors[a])
            
            #add class of each target
            tcls.append(c)
        return tcls, tboxes, indices, anch
