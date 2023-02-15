/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_meta_info_helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/generic_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {
/*
 * Stack converter from fluid to tensorRT.
 */
class CustomPluginCreater : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert " << op_desc.Type() << " op to custom pluign layer";

    std::string plugin_name;

    if (engine_->with_dynamic_shape()) {
      plugin_name = op_desc.Type() + "_paddle_trt_dynamic_plugin";
    } else {
      plugin_name = op_desc.Type() + "_paddle_trt_plugin";
    }

    nvinfer1::ILayer *layer = nullptr;
    std::vector<nvinfer1::ITensor *> inputs;

    auto &op_meta_info_map = OpMetaInfoMap::Instance();
    const auto &meta_info_map = op_meta_info_map.GetMap();
    auto &op_info = meta_info_map.at(op_desc.Type()).front();

    // set inputs
    auto &op_input_names = framework::OpMetaInfoHelper::GetInputs(op_info);
    for (auto &param_name : op_input_names) {
      for (auto &arg_name : op_desc.Input(param_name)) {
        inputs.push_back(engine_->GetITensor(arg_name));
      }
    }
    auto creator =
        GetPluginRegistry()->getPluginCreator(plugin_name.c_str(), "1");
    CHECK(creator);

    // set attrs
    std::vector<nvinfer1::PluginField> plugindatas;
    auto &op_attrs_names = framework::OpMetaInfoHelper::GetAttrs(op_info);
    auto &attrs = op_desc.GetAttrMap();

    std::list<int> int_attrs;
    std::list<float> float_attrs;
    std::list<double> bool_attrs;
    std::list<std::string> string_attrs;
    std::list<std::vector<int>> ints_attrs;
    std::list<std::vector<float>> floats_attrs;

    for (auto &attr_name : op_attrs_names) {
      nvinfer1::PluginField plugindata;
      plugindata.name = attr_name.c_str();
      if (op_desc.GetAttrType(attr_name) == framework::proto::AttrType::INT) {
        int_attrs.push_back(PADDLE_GET_CONST(int, attrs.at(attr_name)));
        plugindata.data = &int_attrs.back();
        plugindata.type = nvinfer1::PluginFieldType::kINT32;
        plugindata.length = 1;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::FLOAT) {
        float_attrs.push_back(PADDLE_GET_CONST(float, attrs.at(attr_name)));
        plugindata.data = &float_attrs.back();
        plugindata.type = nvinfer1::PluginFieldType::kFLOAT32;
        plugindata.length = 1;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::BOOLEAN) {
        int_attrs.push_back(PADDLE_GET_CONST(bool, attrs.at(attr_name)));
        plugindata.data = &int_attrs.back();
        plugindata.type = nvinfer1::PluginFieldType::kINT32;
        plugindata.length = 1;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::STRING) {
        string_attrs.push_back(
            PADDLE_GET_CONST(std::string, attrs.at(attr_name)));
        plugindata.data = string_attrs.back().data();
        plugindata.type = nvinfer1::PluginFieldType::kCHAR;
        plugindata.length =
            string_attrs.back().size() + 1;  // string ends with ‘\0’
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::INTS) {
        ints_attrs.push_back(
            PADDLE_GET_CONST(std::vector<int>, attrs.at(attr_name)));
        plugindata.data = ints_attrs.back().data();
        plugindata.type = nvinfer1::PluginFieldType::kINT32;
        plugindata.length = ints_attrs.back().size();
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::FLOATS) {
        floats_attrs.push_back(
            PADDLE_GET_CONST(std::vector<float>, attrs.at(attr_name)));
        plugindata.data = floats_attrs.back().data();
        plugindata.type = nvinfer1::PluginFieldType::kFLOAT32;
        plugindata.length = floats_attrs.back().size();
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::BOOLEANS) {
        auto bools_attr =
            PADDLE_GET_CONST(std::vector<bool>, attrs.at(attr_name));
        std::vector<int> convert_to_ints_attr;
        for (bool i : bools_attr) convert_to_ints_attr.push_back(i);
        ints_attrs.push_back(convert_to_ints_attr);
        plugindata.data = ints_attrs.back().data();
        plugindata.type = nvinfer1::PluginFieldType::kINT32;
        plugindata.length = ints_attrs.back().size();
      } else {
        CHECK(false) << "UNKNOWN PluginFieldType.";
      }
      plugindatas.push_back(plugindata);
    }

    nvinfer1::PluginFieldCollection plugin_fc{(int32_t)plugindatas.size(),
                                              plugindatas.data()};

    auto *plugin = creator->createPlugin(op_desc.Type().c_str(), &plugin_fc);
    CHECK(plugin);

    if (engine_->with_dynamic_shape()) {
      layer =
          engine_->AddDynamicPlugin(inputs.data(),
                                    inputs.size(),
                                    (plugin::DynamicPluginTensorRT *)plugin);
    } else {
      layer = engine_->AddPlugin(
          inputs.data(), inputs.size(), (plugin::PluginTensorRT *)plugin);
    }

    CHECK(layer);

    // set outputs
    auto &op_output_names = framework::OpMetaInfoHelper::GetOutputs(op_info);
    std::vector<std::string> output_names;
    for (auto &param_name : op_output_names) {
      for (auto &arg_name : op_desc.Output(param_name))
        output_names.push_back(arg_name);
    }

    RreplenishLayerAndOutput(layer, op_desc.Type(), output_names, test_mode);
  }
};

class GenericPluginCreater : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    CHECK(block_);
    const framework::BlockDesc block_desc(
        nullptr, const_cast<framework::proto::BlockDesc *>(block_));

    nvinfer1::ILayer *layer = nullptr;
    std::vector<nvinfer1::ITensor *> inputs;

    phi::KernelSignature phi_kernel_signature;
    if (phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_desc.Type())) {
      const phi::ArgumentMappingFn *argument_mapping_func =
          phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_desc.Type());
      PluginArgumentMappingContext argument_mapping_context(&op_desc);
      phi_kernel_signature = (*argument_mapping_func)(argument_mapping_context);
    } else {
      phi_kernel_signature =
          phi::DefaultKernelSignatureMap::Instance().Get(op_desc.Type());
    }

    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

    plugin::GenericPlugin::InputOutPutVarInfo in_out_info;

    for (auto &param_name : phi_kernel_signature.input_names) {
      for (auto &arg_name : op_desc.Input(param_name)) {
        inputs.push_back(engine_->GetITensor(arg_name));
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            platform::errors::NotFound(
                "There is no variable called %s in block.", arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_LOD_TENSOR,
            platform::errors::InvalidArgument("TensorRT engine only takes "
                                              "LoDTensor as input"));
        in_out_info.inputs_data_type.push_back(var->GetDataType());
      }
    }

    std::vector<std::string> output_names;
    for (auto &param_name : phi_kernel_signature.output_names) {
      for (auto &arg_name : op_desc.Output(param_name)) {
        output_names.push_back(arg_name);
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            platform::errors::NotFound(
                "There is no variable called %s in block.", arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_LOD_TENSOR,
            platform::errors::InvalidArgument("TensorRT engine only takes "
                                              "LoDTensor as input"));
        in_out_info.outputs_data_type.push_back(var->GetDataType());
      }
    }
    plugin::GenericPlugin *plugin =
        new plugin::GenericPlugin(op, in_out_info, with_fp16);
    layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);

    RreplenishLayerAndOutput(layer, op_desc.Type(), output_names, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(custom_plugin_creater, CustomPluginCreater);
REGISTER_TRT_OP_CONVERTER(generic_plugin_creater, GenericPluginCreater);
