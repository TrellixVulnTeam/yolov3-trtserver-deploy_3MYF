import argparse
from sys import platform
import sys

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import onnxruntime as rt
import cv2
import time

import numpy as np
import tensorflow as tf

class detector:
    def __init__(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=opt.weights)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()


def detect(save_txt=False, save_img=False):

    img_size = (416, 416)
    weights = opt.weights

    # Initialize
    device = torch_utils.select_device(device='cpu')

    interpreter = tf.lite.Interpreter(model_path=opt.weights)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    half = False
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    input_shape = input_details[0]['shape']
    print("input_shape", input_shape)
    vid = cv2.VideoCapture(0)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (1920, 1080))
        else:
            raise ValueError("No image!")
        cv2.resize(frame, img, )
        img = frame
        img = img[None, :, :, :]
        input_data = img.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print("output_data.shape", output_data.shape)
        pred = torch.Tensor(output_data)
        # pred = torch.Tensor(pred)
        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov3.tflite', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()

