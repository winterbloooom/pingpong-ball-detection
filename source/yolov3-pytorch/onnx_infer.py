import onnx
import onnxruntime
import argparse
import sys
import torch
import numpy as np
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from dataloader.data_transforms import * 
from util.tools import *

def parse_args():
    parser = argparse.ArgumentParser(description="onnx_inference")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of device ids.")
    parser.add_argument('--model', type=str, help="model path",
                        default=None, dest='model')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

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

    output = [np.zeros(6)] * prediction.shape[0]

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

def main():
    print("onnx_inference")
    print("onnxruntime :" , onnxruntime.get_device())
    
    model = onnx.load(args.model)
    
    img = cv2.imread("C:/data//kitti_dataset//testing//Images//000315.png", cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (608,608), cv2.INTER_LINEAR)
    #cv2.imshow("show input", img)
    #cv2.waitKey(0)
    img = np.transpose(np.array(img, dtype=np.float32) / 255, (2, 0, 1))
    np_img = np.expand_dims(img, axis=0)
    print(np_img.dtype)
    img = torch.FloatTensor(np.expand_dims(img, axis=0)).to(torch.device("cuda:0"))
    
    
    print("input dim : ", img.shape)
    
    print(onnx.checker.check_model(model))
    x_test = torch.randn(1, 3, 608, 608, requires_grad=True).to(torch.device("cuda:0"))
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648,
        'trt_fp16_enable': True,
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
    ]
    ort_session = onnxruntime.InferenceSession(args.model,providers=providers) #, 'CPUExecutionProvider' ['TensorrtExecutionProvider', 'CUDAExecutionProvider']

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: np_img} #to_numpy(img)
    
    ort_outs = ort_session.run(None, ort_inputs)
    print("out dim: ", ort_outs[0].shape)

if __name__ == "__main__":
    args = parse_args()
    main()