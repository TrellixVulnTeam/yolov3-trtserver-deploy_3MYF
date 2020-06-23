import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

from random import randint
from PIL import Image
import numpy as np
import tempfile

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "input"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SHAPE = (10, )
    DTYPE = trt.float32



# For more information on TRT basics, refer to the introductory parser samples.
def build_engine(deploy_file, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        # Parse the model and build the engine.
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        return builder.build_cuda_engine(network)

# Tries to load an engine from the provided engine_path, or builds and saves an engine to the engine_path.
def get_engine(engine_path):
    try:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    except:
        print("read engine fail!!!")
        return engine

# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case():
    case_num = randint(0, 9)
    [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"])
    # Flatten the image into a 1D array, and normalize.
    img = np.array(Image.open(test_case_path)).ravel() - mean
    return img, case_num

def main():

    engine_path = os.path.join(".", "nms_plugin.engine")
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        mean = retrieve_mean(mean_proto)
        # For more information on performing inference, refer to the introductory samples.
        inputs[0].host, case_num = load_normalized_test_case()
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.argmax(output)
        print("Test Case: " + str(case_num))
        print("Prediction: " + str(pred))

if __name__ == "__main__":
    main()
