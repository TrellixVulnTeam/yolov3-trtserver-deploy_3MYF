#!/usr/bin/env python3

# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ctypes
import logging

import numpy as np
import tensorrt as trt
import common

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

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

            profiles[bs].set_shape(inp.name, min=(1, *shape), opt=(1, *shape), max=(16, *shape))

    return list(profiles.values())

def add_profiles(config, inputs, opt_profiles):
    logger.debug("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)


if __name__ == '__main__':


    # Load our CustomPlugin library
    plugin_library = os.path.join("/opt/plugin/custom_plugin_dynamicshape/build", "libNMSPlugin.so")
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


    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
    }

    # Add our custom plugin to a network, and build a TensorRT engine from it.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
        builder.create_builder_config() as builder_config :


        builder_config.max_workspace_size = 3*2**30 # 1GiB
        # for flag in builder_flag_map:
        #         builder_config.set_flag(builder_flag_map[flag])

        # if 1 and not builder.platform_has_fast_fp16:
        #     logger.warning("FP16 not supported on this platform.")

        # builder.max_workspace_size = common.GiB(2)
        input0 = (-1, 10647, 1, 4)
        input1 = (-1, 10647, 80)
        data0 = network.add_input("input0", trt.DataType.FLOAT, input0)
        data1 = network.add_input("input1", trt.DataType.FLOAT, input1)


        # reshape = network_definition.add_shuffle(y)
		# reshape.set_input(1, network_definition.add_shape(X)->get_output(0))

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
        out = network.add_plugin_v2([data0, data1], customPlugin)
        network.mark_output(out.get_output(0))
        network.mark_output(out.get_output(1))
        network.mark_output(out.get_output(2))
        network.mark_output(out.get_output(3))

        batch_sizes = [1]
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
        add_profiles(builder_config, inputs, opt_profiles)

        # Serialize our engine for future use
        logger.info("Building engine...")
        with builder.build_engine(network, builder_config) as engine:
            filename = "dy_nms_plugin.engine"
            with open(filename, "wb") as f:
                f.write(engine.serialize())
                logger.info("Serialized engine file written to {}".format(filename))


        # with open("dy_nms_plugin.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        #     engine1 = runtime.deserialize_cuda_engine(f.read())
        # print('dsfsf')
            # TODO: Add inference example
            # x = np.ones(data_shape, dtype=np.float32)

