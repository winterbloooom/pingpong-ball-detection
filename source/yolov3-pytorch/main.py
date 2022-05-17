import os,sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from eval.evaluator import Evaluator
#import warnings
#warnings.filterwarnings("error")
from torch.utils.data.dataloader import DataLoader
import torchsummary as summary
from model.yolov3 import DarkNet53
from dataloader.yolodata import *
from train.trainer import Trainer
from demo.demo import Demo
from dataloader.data_transforms import *
from tensorboardX import SummaryWriter
import pynvml

import onnx,onnxruntime

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

def get_memory_total_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.total // 1024 ** 2

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3-PYTORCH")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of device ids.")
    parser.add_argument('--mode', dest='mode', help="train / eval / demo / onnx",
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg', help="model config path",
                        default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help = "the path of checkpoint",
                        default=None, type=str)
    parser.add_argument('--pretrained', dest='pretrained', help = "the path of pre-trained model (.weights)",
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    #skip invalid frames
    if len(batch) == 0:
        return

    imgs, targets, anno_path = list(zip(*batch))

    imgs = torch.stack([img for img in imgs])
    
    if targets[0] is None or anno_path[0] is None:
        return imgs, None, None

    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets,0)

    return imgs, targets, anno_path

def train(cfg_param = None, using_gpus = None):
    #Train dataloader
    transforms = get_transformations(cfg_param, is_train = True)

    train_data = Yolodata(is_train=True,
                          transform=transforms,
                          cfg_param = cfg_param)
    
    train_loader = DataLoader(train_data, 
                              batch_size=cfg_param['batch'],
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True,
                              collate_fn=collate_fn,
                              worker_init_fn=worker_seed_set)

    #evaluation dataloader
    eval_transforms = get_transformations(cfg_param, is_train = False)
    eval_data = Yolodata(is_train = False,
                         transform = eval_transforms,
                         cfg_param = cfg_param)
    eval_dataloader = DataLoader(eval_data,
                                 batch_size = cfg_param['batch'],
                                 num_workers = 0,
                                 pin_memory = True,
                                 drop_last = False,
                                 shuffle = False,
                                 collate_fn=collate_fn,
                                 worker_init_fn=worker_seed_set)

    


    #Get OD model
    model = DarkNet53(args.cfg, cfg_param)
    
    #load pre-trained darknet weights
    if args.pretrained is not None:
        print("load pretrained model")
        model.load_darknet_weights(args.pretrained)
    else:
        model.initialize_weights()
    
    #Set the device what you use, GPU or CPU
    for i in using_gpus:
        print("GPU total memory : {} free memory : {}".format(get_memory_total_MiB(i), get_memory_free_MiB(i)))
        if get_memory_free_MiB(i) / get_memory_total_MiB(i) < 0.5:
            print("Avaliable memory is {}%, GPU is already used now, Exit process".format(get_memory_free_MiB(i) / get_memory_total_MiB(i)))
            sys.exit(1)
    if len(using_gpus) == 1:
        device = torch.device("cuda:"+str(using_gpus[0]) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(using_gpus[0])
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
    elif len(using_gpus) == 0:
        print("Disable to use GPU. Exit process")
        device = torch.device("cpu")
        model = model.to(device)
    elif len(using_gpus) > 1:
        print("using_gpus : {}".format(using_gpus))
        device = torch.device("cuda:"+str(using_gpus[0]) if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')

    torch.backends.cudnn.benchmark = True

    #If checkpoint is existed, load the previous checkpoint.
    checkpoint = None
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        for key, value in checkpoint['model_state_dict'].copy().items():
            new_key = "module." + key
            checkpoint['model_state_dict'][new_key] = checkpoint['model_state_dict'].pop(key)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    #Pre-check the model structure and size of parameters
    summary.summary(model, input_size=(3, cfg_param["in_width"], cfg_param["in_height"]), device='cuda') #or 'cpu'
    
    #Setting the torch log directory to use tensorboard
    torch_writer = SummaryWriter("./output")
    
    if len(using_gpus) > 0:
        yolo_model = model.module
    else:
        yolo_model = model
        
    yolo_model.train()
    
    #Set trainer
    trainer = Trainer(yolo_model, train_loader, eval_dataloader, cfg_param, eval_data.class_str, device, checkpoint, torch_writer = torch_writer)
    trainer.run()

def eval(cfg_param = None, using_gpus = None):
    print("evaluation")
    transforms = get_transformations(cfg_param, is_train = False)    
    eval_data = Yolodata(is_train = False, transform = transforms, cfg_param = cfg_param)
    eval_loader = DataLoader(eval_data, batch_size = 1, num_workers = 0, pin_memory = True, drop_last = False, shuffle = False, collate_fn=collate_fn)
    
    model = DarkNet53(args.cfg, cfg_param)

    if len(using_gpus) == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device == torch.device('cuda'):
        print("device is cuda")
    elif device == torch.device('cpu'):
        print('device is cpu')

    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    
    model.eval()
    
    torch.backends.cudnn.benchmark = True

    evaluator = Evaluator(model, eval_data, eval_loader, device, cfg_param)
    
    evaluator.run()
    
def demo(cfg_param = None, using_gpus = None):
    print("demo")
    transforms = get_transformations(cfg_param, is_train = False)    
    data = Yolodata(is_train = False, transform = transforms, cfg_param = cfg_param)
    demo_loader = DataLoader(data, batch_size = 1, num_workers = 0, pin_memory = True, drop_last = False, shuffle = False, collate_fn=collate_fn)
    
    model = DarkNet53(args.cfg, cfg_param)
    model.eval()

    #load pre-trained darknet weights
    if args.pretrained is not None:
        print("load pretrained model")
        model.load_darknet_weights(args.pretrained)
        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}
        target = args.pretrained.replace(".weights", ".pth")
        torch.save(chkpt, target)
    else:
        model.initialize_weights()
    
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if len(using_gpus) == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device == torch.device('cuda'):
        print("device is cuda")
    elif device == torch.device('cpu'):
        print('device is cpu')
    
    model = model.to(device)
    model.eval()
    
    if args.pretrained is not None:
        darknet_weights_name = args.pretrained.replace(".weights", "_new.weights")
    elif args.checkpoint is not None:
        darknet_weights_name = args.checkpoint.replace(".pth", ".weights")
    model.save_darknet_weights(darknet_weights_name, cutoff=-1)

    torch.backends.cudnn.benchmark = True

    demo = Demo(model, data, demo_loader, device, cfg_param)
    
    demo.run()


#convert trained yolov3 model from pytorch to ONNX format 
def torch2onnx(cfg_param = None, using_gpus = None):
    #Get OD model
    cfg_param['batch'] = 1
    model = DarkNet53(args.cfg, cfg_param)
    
    if args.pretrained is not None:
        model.load_darknet_weights(args.pretrained)
    #Set the device what you use, GPU or CPU
    for i in using_gpus:
        print("GPU total memory : {} free memory : {}".format(get_memory_total_MiB(i), get_memory_free_MiB(i)))
        if get_memory_free_MiB(i) / get_memory_total_MiB(i) < 0.5:
            print("Avaliable memory is {}%, GPU is already used now, Exit process".format(get_memory_free_MiB(i) / get_memory_total_MiB(i)))
            sys.exit(1)
    if len(using_gpus) == 1:
        device = torch.device("cuda:"+str(using_gpus[0]) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(using_gpus[0])
        #model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
    elif len(using_gpus) == 0:
        print("Disable to use GPU. Exit process")
        device = torch.device("cpu")
        model = model.to(device)
    elif len(using_gpus) > 1:
        print("using_gpus : {}".format(using_gpus))
        device = torch.device("cuda:"+str(using_gpus[0]) if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')

    torch.backends.cudnn.benchmark = True

    #If checkpoint is existed, load the previous checkpoint.
    checkpoint = None
    if args.checkpoint is not None:
        print("load checkpoint model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    darknet_weights_name = args.checkpoint.replace(".pth", ".weights")
    onnx_weights_name = args.checkpoint.replace(".pth", ".onnx")
    #save the model to darknet format
    model.save_darknet_weights(darknet_weights_name, cutoff=-1)
    
    #export from torch model to ONNX format
    x_test = torch.ones(1, 3, cfg_param["in_width"], cfg_param["in_height"], requires_grad=True, dtype=torch.float32).to(device)
    torch.onnx.export(model, x_test, onnx_weights_name, export_params=True, opset_version=9, input_names=['input'], output_names=['output'] )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_weights_name, providers=['CPUExecutionProvider'])

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x_test)}
    
    #inference onnx_model
    ort_outs = ort_session.run(None, ort_inputs)
    #inference torch_model
    torch_outs = model(x_test)
    
    #print("torch output : ", len(torch_outs), " ", torch_outs.shape)
    print("onnx out: ", len(ort_outs), ort_outs[0].shape)
    for i in range(len(torch_outs)):
        # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
        torch_np_outs = to_numpy(torch_outs[i])
        np.testing.assert_allclose(torch_np_outs, ort_outs[i], rtol=1e-03, atol=1e-05)

if __name__ == "__main__":
    args = parse_args()
    cfg_data = parse_hyperparam_config(args.cfg)
    cfg_param = get_hyperparam(cfg_data)
    
    # multi-gpu
    print("GPUS : ", args.gpus)
    using_gpus = [int(g) for g in args.gpus]

    if args.mode == "train":
        train(cfg_param, using_gpus)
    elif args.mode == "eval":
        eval(cfg_param, using_gpus)
    elif args.mode == "demo":
        demo(cfg_param, using_gpus)
    elif args.mode == "onnx":
        torch2onnx(cfg_param, using_gpus)
    else:
        print("Unknown mode error")
    print("finish")
