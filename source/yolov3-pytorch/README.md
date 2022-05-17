# YOLOv3-pytorch

single stage object detection Yolov3.

This is made with Pytorch.


<img src=https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-24_at_12.52.19_PM_awcwYBa.png width=416>

----------------------------

## Install

### Windows

#### Use Anaconda

1. Download Anaconda : https://www.anaconda.com/products/individual#windows
2. conda create --name ${environment_name} python=3.8
3. activate ${environment_name}
4. git clone https://github.com/2damin/yolov3-pytorch.git


### Linux

#### Use docker

I recommend Nvidia NGC docker image. [link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

1. docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
2. docker run --gpus all -it --rm -v local_dir:container_dir -p 8888:8888 nvcr.io/nvidia/pytorch:xx.xx-py3
   1. check "nvidia-smi"
   2. check "nvcc --version"
3. git clone https://github.com/2damin/yolov3-pytorch.git


## Dependency

pip install -r requirements.txt
```
python >= 3.6

Numpy

torch >= 1.9

torchvision >= 0.10

tensorboard

tensorboardX

torchsummary

pynvml

imgaug

onnx

onnxruntime
```

-------------------

## Run

If training,

```{r, engine='bash', count_lines}
(single gpu) python main.py --mode train --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}

(multi gpu) python main.py --mode train --cfg ./yolov3.cfg --gpus 0 1 2 3 --checkpoint ${saved_checkpoint_path}
```

If evaluate,

```{r, engine='bash', count_lines}
python main.py --mode eval --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

If test,

```{r, engine='bash', count_lines}
python main.py --mode demo --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

If converting torch to onnx,

target tensorrt version > 7
```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

target tensorrt version is 5.x

1. **ONNX_EXPORT = True** in 'model/yolov3.py'
   
   tensorrt(v5.x) is not support upsample scale factor, so you have to change upsample layer not using scale factor.

```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

### option

--mode : train/eval/demo.

--cfg : the path of model.cfg.

--gpu : if you use GPU, set 1. If you use CPU, set 0.

--checkpoint (optional) : the path of saved model checkpoint. Use it when you want to load the previous train, or you want to test(evaluate) the model.

--pretrained (optional) : the path of darknet pretrained weights. Use it when you want to fine-tuning the model.



## Visualize training graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir=./output --port 8888
```

-------------------------

# Reference

[YOLOv3 paper](https://arxiv.org/abs/1804.02767)

[KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


# error

- if libgl.so error when cv2
```
apt-get update
apt-get install libgl1-mesa-glx
```