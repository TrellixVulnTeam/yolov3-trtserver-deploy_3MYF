#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


import ctypes
import logging


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class HostDeviceMem(object):
    r""" Simple helper data class that's a little nicer to use than a 2-tuple.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        logger.error("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))


def check_network(network):
    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    logger.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logger.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))       

def get_plugin_creator_by_name(plugin_registry, plugin_name):
    plugin_creator_list = plugin_registry.plugin_creator_list
    for c in plugin_creator_list:
        if c.name == plugin_name:
            return c
def get_all_plugin_names(plugin_registry):
    return [c.name for c in plugin_registry.plugin_creator_list]

def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
        # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]

            profiles[bs].set_shape(inp.name, min=(1, *shape), opt=(1, *shape), max=(2, *shape))

    return list(profiles.values())

def add_profiles(config, inputs, opt_profiles):
    logger.debug("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)


class ModelData(object):
    MODEL_PATH = "yolov3_batch.onnx"
    INPUT_SHAPE = (3, 416, 416)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
        # Load our CustomPlugin library
    plugin_library = os.path.join("/workspace/custom_plugin_dynamicshape/build", "libNMSPlugin.so")
    logger.info("Loading plugin library: {}".format(plugin_library))
    ctypes.cdll.LoadLibrary(plugin_library)

    logger.info("Initializing plugin registry")
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    plugin_registry = trt.get_plugin_registry()
    # Get plugin creator for our custom plugin.

    # List all registered plugins. Should see our CustomPlugin in this list.
    logger.info("Registered Plugins:")
    print("\n".join([c.name for c in plugin_registry.plugin_creator_list]))


    plugin_name = "DY_BatchedNMS"
    logger.info("Looking up IPluginCreator for {}".format(plugin_name))
    plugin_creator = get_plugin_creator_by_name(plugin_registry, plugin_name)
    if not plugin_creator:
        raise Exception("[{}] IPluginCreator not found.".format(plugin_name))
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as builder_config :
        builder_config.max_workspace_size = common.GiB(4)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        logger.info("Creating PluginFields for {} plugin".format(plugin_name))

        logger.info("Creating PluginFields for {} plugin".format(plugin_name))
        plugin_fields = []
        plugin_field = trt.PluginField("shareLocation".format(1), np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("backgroundLabelId", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("numClasses", np.array([80], dtype=np.int32), trt.PluginFieldType.INT32)
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("topK", np.array([100], dtype=np.int32), trt.PluginFieldType.INT32)
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("keepTopK", np.array([100], dtype=np.int32), trt.PluginFieldType.INT32)
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("scoreThreshold", np.array([0.5], dtype=np.float), trt.PluginFieldType.FLOAT32)
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("iouThreshold", np.array([0.3], dtype=np.float), trt.PluginFieldType.FLOAT32)        
        plugin_fields.append(plugin_field)
        plugin_field = trt.PluginField("isNormalized", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)          
        plugin_fields.append(plugin_field)
        
        logger.info("Creating PluginFieldCollection for {} plugin".format(plugin_name))
        plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
        logger.info("Creating {} plugin from PluginFieldCollection".format(plugin_name))
        customPlugin = plugin_creator.create_plugin(plugin_name, plugin_field_collection)

        logger.info("Adding {} plugin to network.".format(plugin_name))
        network.get_input(0).name = "inputs"

        boxes_1 = network.get_output(0)
        scores_1 = network.get_output(1)
        boxs_reshape_op = network.add_shuffle(input=boxes_1)      
        boxs_reshape_op.reshape_dims = [-1, 10647, 1, 4]             
        boxs = boxs_reshape_op.get_output(0)         

        scores_reshape_op = network.add_shuffle(input=scores_1)      
        scores_reshape_op.reshape_dims = [-1, 10647, 80]             
        scores = scores_reshape_op.get_output(0)        
  


        nms = network.add_plugin_v2([boxs, scores], customPlugin)

        print('nms.plugin.plugin_namespace', nms.plugin.plugin_namespace)
        nms.plugin.plugin_namespace = ""
        print('nms.plugin.plugin_namespace', nms.plugin.plugin_namespace)

        nms.get_output(0).name = "num"
        print("nms.get_output(0).shape", nms.get_output(0).shape)
        # reshape num_detections to adapt dynamic batch dim of 
        # Triton (TensorRT) Inference Server
        det = nms.get_output(0)
        det_reshape_op = network.add_shuffle(input=det)
        det_reshape_op.reshape_dims = [-1, 1]
        det_reshape = det_reshape_op.get_output(0)
        print("det_rehshape.shape", det_reshape.shape)
        det_reshape.name = "num_detections"
        network.mark_output(det_reshape)

        # network.mark_output(nms.get_output(0))
        nms.get_output(1).name = "detection_boxes"
        network.mark_output(nms.get_output(1))
        nms.get_output(2).name = "detection_scores"
        network.mark_output(nms.get_output(2))
        nms.get_output(3).name = "detection_classes"
        network.mark_output(nms.get_output(3))
        # print(boxes, boxes.name)
        # print(scores, scores.name)
        network.unmark_output(boxes_1)
        network.unmark_output(scores_1)

        check_network(network)

        batch_sizes = [1]
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
        add_profiles(builder_config, inputs, opt_profiles)

        return builder.build_engine(network, builder_config)

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return image_arr / 255.0

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def main():


    # Set the data path to the directory that contains the trained models and test images for inference.
    _, data_files = common.find_sample_data(description="Runs a Yolov3 network with a TensorRT inference engine.", subfolder="yolov3", find_files=["bus.jpg", ModelData.MODEL_PATH])
    # Get test images, models and labels.
    test_images = data_files[0:1]
    onnx_model_file = data_files[1]

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file) as engine:
        filename = "yolov3.engine"
        with open(filename, "wb") as f:
            f.write(engine.serialize())
            logger.info("Serialized engine file written to {}".format(filename))



def allocate_buffers_1(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size * -1
        print(engine.get_binding_shape(binding))

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print("++++++++++++++++++++++++++++++")
        print(size)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def allocate_buffers_2(engine: trt.ICudaEngine, batch_size: int):
    print('Allocating buffers ...')

    inputs = []
    outputs = []
    dbindings = []
    global output_dict
    output_dict = dict()

    stream = cuda.Stream()

    for i, binding in enumerate(engine):
        # size = batch_size * trt.volume(-1 * engine.get_binding_shape(binding))
        output_name = engine.get_binding_name(i)
        print("binding %d's name: %s" % (i, engine.get_binding_name(i)))
        size = batch_size * trt.volume(engine.get_binding_shape(binding)[1:])
        # print('size', size)
        print('shape', engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print('dtype', dtype)
        # Allocate host and device buffers
        # print(size)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        dbindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_dict[output_name] = host_mem

    return inputs, outputs, dbindings, stream


def main1():
    plugin_library = os.path.join("/workspace/custom_plugin_dynamicshape/build", "libNMSPlugin.so")
    logger.info("Loading plugin library: {}".format(plugin_library))
    ctypes.cdll.LoadLibrary(plugin_library)
    _, data_files = common.find_sample_data(description="Runs a Yolov3 network with a TensorRT inference engine.", subfolder="yolov3", find_files=["bus.jpg", ModelData.MODEL_PATH])
    # Get test images, models and labels.
    test_images = data_files[0:1]
    engine_path = "yolov3.engine"
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        TRT_LOGGER = trt.Logger()
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        engine = runtime.deserialize_cuda_engine(f.read())

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings, stream = allocate_buffers_2(engine, 1)

    with engine.create_execution_context() as context:
        test_image = random.choice(test_images)
        test_case = load_normalized_test_case(test_image, inputs[0].host)
        context.set_binding_shape(0, [1, 3, 416, 416])
        trt_outputs = common.do_inference_v2(context, bindings=dbindings, inputs=inputs, outputs=outputs, stream=stream)
        print("end")

    # try:
    #     with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    #         inputs, outputs, bindings, stream = allocate_buffers_2(engine, 1)
    #         with engine.create_execution_context() as context:
    #             test_image = random.choice(test_images)
    #             test_case = load_normalized_test_case(test_image, inputs[0].host)
                
    #             trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #             print("end")
    # except:
    #     print("read engine fail!!!")

if __name__ == '__main__':
    main()
