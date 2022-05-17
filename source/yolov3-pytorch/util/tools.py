import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math
import random
import tqdm
import torchvision
import sys, os
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    print("unique_classes:", unique_classes)
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = box_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)
                
                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)

def minmax2cxcy(box):
    if len(box) != 4:
        return torch.FloatTensor([0,0,0,0])
    else:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        if cx - w/2 < 0 or cx + w/2 > 1:
            w -= 0.001
        if cy - h/2 < 0 or cy + h/2 > 1:
            h -= 0.001
        box[0] = cx
        box[1] = cy
        box[2] = w
        box[3] = h

def show_img(img_data, text):
    _img_data = img_data[0] *255
    
    _img_data = np.transpose(_img_data,[1,2,0])
    _img_data = np.array(_img_data, dtype=np.uint8)
    #print("shape:",_img_data.shape)
    img_data = Image.fromarray(_img_data)

    # directory="C:/Users/dogu/Desktop/temp/ImgAug_f/"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # img_data.save("{}{}.png".format(directory,text))
    # img_data.show()


def cxcy2minmax(box):
    y = box.new(box.shape)
    xmin = box[...,0] - box[...,2] / 2
    ymin = box[...,1] - box[...,3] / 2
    xmax = box[...,0] + box[...,2] / 2
    ymax = box[...,1] + box[...,3] / 2
    y[...,0] = xmin
    y[...,1] = ymin
    y[...,2] = xmax
    y[...,3] = ymax
    return y

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def resizeBox(box, original_wh, resize_wh):
    if len(box) != 4:
        return torch.FloatTensor([0,0,0,0])
    else:
        ratio_w, ratio_h = resize_wh[0] / original_wh[0], resize_wh[1] / original_wh[1]
        box[0] = box[0] * ratio_w
        box[1] = box[1] * ratio_h
        box[2] = box[2] * ratio_w
        box[3] = box[3] * ratio_h
        # xmin, xmax = box[0] * ratio_w, box[2] * ratio_w
        # ymin, ymax = box[1] * ratio_h, box[3] * ratio_h
        # return torch.FloatTensor([xmin, ymin, xmax, ymax])

def box_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / (h2+eps)) - torch.atan(w1 / (h1+eps)), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v + eps)
                ciou = iou - (rho2 / c2 + v * alpha)
                return ciou  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def iou(a, b, mode = 0, device = None, eps=1e-9):
    #mode 0 : cxcywh. mode 1 : minmax
    if mode == 0:
        a_x1, a_y1 = a[:,0]-a[:,2]/2, a[:,1]-a[:,3]/2
        a_x2, a_y2 = a[:,0]+a[:,2]/2, a[:,1]+a[:,3]/2
        b_x1, b_y1 = b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2
        b_x2, b_y2 = b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    else:
        a_x1, a_y1, a_x2, a_y2 = a[:,0], a[:,1], a[:,2], a[:,3]
        b_x1, b_y1, b_x2, b_y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    xmin = torch.max(a_x1, b_x1)
    xmax = torch.min(a_x2, b_x2)
    ymin = torch.max(a_y1, b_y1)
    ymax = torch.min(a_y2, b_y2)
    #get intersection area 
    inter = (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0)
    #get each box area
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1 + eps)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1 + eps)
    union = a_area + b_area - inter + eps
    
    if device is not None:
        iou = torch.zeros(b.shape[0], device=device)
    else:
        iou = torch.zeros(b.shape[0])
    iou = inter / union

    return iou

def sigmoid(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [1 / (1 + np.exp(-v)) for v in a]
    a = np.reshape(a, a_shape)
    return a

def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / exp_a.sum()

def drawBox(_img, boxes = None, cls = None, mode = 0, color = (0,255,0)):
    _img = _img * 255
    #img dim is [C,H,W]
    if _img.shape[0] == 3:
        _img_data = np.array(np.transpose(_img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(_img_data)
    elif _img.ndim == 2:
        _img_data = np.array(_img, dtype=np.uint8)
        img_data = Image.fromarray(_img_data, 'L')
    draw = ImageDraw.Draw(img_data)
    fontsize = 15
    font = ImageFont.truetype("./arial.ttf", fontsize)
    if boxes is not None:
        for i, box in enumerate(boxes):
            # if (box[4] + box[5]) / 2 < 0.5:
            #     continue
            # if cls[i] == 8:
            #     color = (255,0,0)
            # if i == 2:
            #     color = (0,255,255)
            if mode == 0:
                draw.rectangle((box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2), outline=(0,255,0), width=1)
            else:
                draw.rectangle((box[0],box[1],box[2],box[3]), outline=(0,255,0), width=1)
            draw.text((box[0],box[1]), str(int(cls[i])), fill ="red", font=font)
    plt.imshow(img_data)
    plt.show()

def drawBoxlist(_img, boxes : list = [], mode : int = 0, name : str = ""):
    _img = _img * 255
    #img dim is [C,H,W]
    if _img.shape[0] == 3:
        _img_data = np.array(np.transpose(_img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(_img_data)
    elif _img.ndim == 2:
        _img_data = np.array(_img, dtype=np.uint8)
        img_data = Image.fromarray(_img_data, 'L')
    draw = ImageDraw.Draw(img_data)
    fontsize = 15
    font = ImageFont.truetype("./arial.ttf", fontsize)
    for box in boxes:       
        if mode == 0:
            draw.rectangle((box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2), outline=(0,255,0), width=1)
            draw.text((box[0],box[1]), str(int(box[5]))+","+str(int(box[4]*100)) , fill ="red", font=font)
        else:
            draw.rectangle((box[0],box[1],box[2],box[3]), outline=(0,255,0), width=1)
            draw.text((box[0],box[1]), str(int(box[5]))+","+str(int(box[4]*100)), fill ="red", font=font)
    #img_data.show("draw")
    img_data.save(name+".png")

def check_outrange(box, img_size):
    box = box.detach().cpu().numpy()
    xmin = box[0] - box[2]/2
    ymin = box[1] - box[3]/2
    xmax = box[0] + box[2]/2
    ymax = box[1] + box[3]/2
    if xmin < 0 or ymin < 0 or xmax > img_size[0] or ymax > img_size[1]:
        return 0
    else:
        return 1

def parse_hyperparam_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            type_name = line[1:-1].rstrip()
            if type_name != "net":
                continue
            module_defs.append({})        
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name != "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs
    
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        
    module_defs = []
    type_name = None
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            type_name = line[1:-1].rstrip()
            if type_name == "net":
                continue
            module_defs.append({})        
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def get_hyperparam(cfg):
    for c in cfg:
        if c['type'] == 'net':
            batch = int(c['batch'])
            subdivision = int(c['subdivisions'])
            momentum = float(c['momentum'])
            decay = float(c['decay'])
            saturation = float(c['saturation'])
            hue = float(c['hue'])
            exposure = float(c['exposure'])
            lr = float(c['learning_rate'])
            burn_in = int(c['burn_in'])
            max_batch = int(c['max_batches'])
            lr_policy = c['policy']
            steps = [int(x) for x in c['steps'].split(',')]
            scales = [float(x) for x in c['scales'].split(',')]
            in_width = int(c['width'])
            in_height = int(c['height'])
            in_channels = int(c['channels'])
            _class = int(c['class'])
            ignore_cls = int(c['ignore_cls'])

            return {'batch':batch,
                    'subdivision':subdivision,
                    'momentum':momentum,
                    'decay':decay,
                    'saturation':saturation,
                    'hue':hue,
                    'exposure':exposure,
                    'lr':lr,
                    'burn_in':burn_in,
                    'max_batch':max_batch,
                    'lr_policy':lr_policy,
                    'steps':steps,
                    'scales':scales,
                    'in_width':in_width,
                    'in_height':in_height,
                    'in_channels':in_channels,
                    'class':_class,
                    'ignore_cls':ignore_cls}
        else:
            continue
        
def non_max_sup(input, num_classes, conf_th = 0.5, nms_th = 0.5, objectness = True):
    
    box = input.new(input.shape)
    box[:,:,0] = input[:,:,0] - input[:,:,2] / 2
    box[:,:,1] = input[:,:,1] - input[:,:,3] / 2
    box[:,:,2] = input[:,:,0] + input[:,:,2] / 2
    box[:,:,3] = input[:,:,1] + input[:,:,3] / 2
    box[:,:,4:] = input[:,:,4:]
    input[:,:,:4] = box[:,:,:4]
    
    #output = [None for _ in range(len(input))]
    output = None
    for i, pred in enumerate(box):
        #get the highst score & class of all pred
        class_conf_all, _ = torch.max(pred[:,5:5+num_classes], 1, keepdim=True)
        if objectness:
            pred_score = pred[:,4]
        else:
            pred_score = class_conf_all * 0.3 + pred[:,4] * 0.7

        conf_mask = (pred_score >= conf_th).squeeze()
        pred = pred[conf_mask]
        if not pred.size(0):
            continue

        #get the highst score & class of masked pred
        class_conf, class_pred = torch.max(pred[:,5:5+num_classes], 1, keepdim=True)
        
        #Convert predictions type [x,y,w,h,obj,class_conf,class_pred]
        detections = torch.cat((pred[:,:5], class_conf.float(), class_pred.float()),1)
        
        device = detections.device
        
        unique_labels = detections[:,-1].cpu().unique()
        if input.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            #get the detection with certain class
            detections_c = detections[detections[:,-1] == c]
            #sort the detections by maximum obj score
            _, conf_sort_index = torch.sort(detections_c[:,4], descending=True)
            detections_c = detections_c[conf_sort_index]
            #perforom non-maximum suppression
            max_detections = []
            while detections_c.size(0):
                #get detection with highest confidence
                max_detections.append(detections_c[0].unsqueeze(0))

                if len(detections_c) == 1:
                    break
                
                #get IOUs for all boxes with lower conf
                ious = iou(max_detections[-1], detections_c[1:], device = device)
                #remove detections iou >= nms threshold
                detections_c = detections_c[1:][ious < nms_th]

            max_detections = torch.cat(max_detections).data
            #update outputs
            output = max_detections if output is None else torch.cat((output, max_detections))
    return output

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = cxcy2minmax(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

    return output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
