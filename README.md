## Introduction
 A complete project for yolov3 deployment on tensorrt-inference-server

## Prerequisites
- `python3`
- `torch==1.3.1`
- `torchvision==0.4.2`
- `onnx==1.6.0`
- `onnx-tf==1.5.0`
- `onnxruntime-gpu==1.0.0` 
- `tensorflow-gpu==1.15.0`

## Docker
- `docker pull zldrobit/onnx:10.0-cudnn7-devel`
- `docker pull yingchao126/tensorrt_plugin:7.0`
- `docker pull yingchao126/tensorrtserver:20.02-py3`

## Usage

### darknet2onnx : get onnx model file in container onnx:10.0-cudnn7-devel
- **1. Download pretrained Darknet weights:**
```
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
```
```
you can also use darknet proj train your custom models

```

- **2. Convert YOLO v3 model from Darknet weights to ONNX model:** 
Change `ONNX_EXPORT` to `True` in `models.py`. Run 
```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
```
```
The output ONNX file is `weights/export.onnx`.
image:batchx416x416x3 output0:batchx10647x1x4 output1:batchx10647x3
```

### trtBatchNms : build tensorrt batchnms-plugin in container yingchao126/tensorrt_plugin:7.0
- **1. copy custom_plugin_dynamicshape project to container get the libNMSPlugin.so:**
```
cd your_path/custom_plugin_dynamicshape
mkdir build && cd build && cmake .. && make
```
- **2. copy onnx2trt project to container get the model.engine:**
```
cd your_path/onnx2trt
edit the onnx_yolov3.py file where appoint classnum 
then exec the onnx_yolov3.py script
```
### trtBatchNms : build tensorrt batchnms-plugin in container yingchao126/tensorrt_plugin:7.0



## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
  
