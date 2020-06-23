import os

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import torchvision

TOP_K = 100


def nms(inputs):
    """

    :param inputs:
    :return:
    """
    inputs = inputs.view(-1, 85)
    boxes = inputs[..., 0:4]
    scores = inputs[..., 4]
    index = torchvision.ops.nms(boxes, scores, 0.4)
    result = torch.index_select(inputs, 0, index)
    detection_scores = result[..., 4].unsqueeze(1)
    _, top100_indict = torch.topk(detection_scores, 100, 0)
    result = torch.index_select(result, 0, top100_indict.squeeze())

    detection_classes = torch.argmax(result[..., 5:], 1, keepdim=False).unsqueeze(1)

    # detection_num = torch.tensor(detection_classes.shape[0], device='cuda').unsqueeze(0).repeat(100)
    # detection_num = detection_num.view(100, 1)

    detection_boxes = result[..., 0:4]
    detection_scores = result[..., 4].unsqueeze(1)
    result = torch.cat((detection_boxes, detection_scores, detection_classes.type(torch.float32)), 1).unsqueeze(0)
    return result


def postprocess(inputs):

    result = list(map(nms, inputs))
    top100 = torch.cat(result, 0)

    detection_boxes = top100[..., 0:4]
    detection_scores = top100[..., 4:5]
    detection_classes = top100[..., 5:6].type(torch.int64)
    # detection_num = top100[..., 6:7].type(torch.int64)

    return detection_boxes, detection_scores, detection_classes



if __name__ == "__main__":
    from trtis import onnx_backend

    inputs_def = [
        {
            "name": "yolov3_result",
            "dims": [None, 10647, 85],
            "data_type": "TYPE_FP32"
        }
    ]

    outputs_def = [
        {
            "name": "detection_boxes",
            "dims": [None, TOP_K, 4],
            "data_type": "TYPE_FP32",
        },
        {
            "name": "detection_scores",
            "dims": [None, TOP_K, 1],
            "data_type": "TYPE_FP32",
        },
        {
            "name": "detection_classes",
            "dims": [None, TOP_K, 1],
            "data_type": "TYPE_FP32",
        }
        # {
        #     "name": "num_detections",
        #     "dims": [None, TOP_K, 1],
        #     "data_type": "TYPE_FP32",
        # }
    ]

    onnx_backend.torch2onnx(
        computation_graph=postprocess,
        graph_name="detection-nms",
        model_file=None,
        inputs_def=inputs_def,
        outputs_def=outputs_def,
        instances=16,
        gpus=[0, 1, 2, 3],
        version=1,
        export_path="nms"
    )
