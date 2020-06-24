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
### models :  deployment yolov3 inference server in container yingchao126/tensorrtserver:20.02-py3
- **1. the libNMSPlugin.so to trtserver container:**
```
docker cp libNMSPlugin.so your_server_container:/opt/tensorrtserver/lib/custom/libNMSPlugin.so 
```
- **2. prepel model file and run servers :**
```
cd your_path/models/your_model/1 && cp your_path/model.engine model.plan
edit your model config.pbtxt
docker cp your_path/models your_server_container:/models
docker run -it --rm --gpus all --shm-size=4g --ulimit memlock=-1 --name trtserver --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v your_path/models/:/models -eLD_PRELOAD=/opt/tensorrtserver/lib/custom/libNMSPlugin.so yingchao126/tensorrtserver:20.02-py3 trtserver --model-repository=/models --pinned-memory-pool-byte-size=0 2>&1


## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
