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
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

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
                  bool test_mode,
                  const framework::proto::BlockDesc *block = nullptr) override {
    framework::OpDesc op_desc(op, nullptr);
    LOG(INFO) << "convert " << op_desc.Type() << " op to custom pluign layer";

    nvinfer1::ILayer *layer = nullptr;
    std::vector<nvinfer1::ITensor *> inputs;

    // TODO(weishengying) we should get op attrs from phi, not OpMetaInfoMap in
    // fluid
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
        getPluginRegistry()->getPluginCreator(op_desc.Type().c_str(), "1");
    CHECK(creator);

    // set attrs
    std::vector<nvinfer1::PluginField> plugindatas;
    auto &op_attrs_names = framework::OpMetaInfoHelper::GetAttrs(op_info);
    auto &attrs = op_desc.GetAttrMap();

    for (auto &attr_name : op_attrs_names) {
      LOG(INFO) << "attr_name: " << attr_name;

      nvinfer1::PluginField plugindata;
      plugindata.name = attr_name.c_str();

      if (op_desc.GetAttrType(attr_name) ==
          framework::proto::AttrType::BOOLEAN) {
        bool value = PADDLE_GET_CONST(bool, attrs.at(attr_name));
        plugindata.data = &value;
        // plugindata.type = ...;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::INT) {
        int value = PADDLE_GET_CONST(int, attrs.at(attr_name));
        plugindata.data = &value;
        // plugindata.type = ...;
      } else {
        CHECK(false) << "not incompleted";
      }

      plugindatas.push_back(plugindata);
    }

    nvinfer1::PluginFieldCollection pluginFC{(int32_t)plugindatas.size(),
                                             plugindatas.data()};

    auto *plugin = creator->createPlugin(op_desc.Type().c_str(), &pluginFC);

    layer = engine_->AddDynamicPlugin(
        inputs.data(), inputs.size(), (plugin::DynamicPluginTensorRT *)plugin);

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
                  bool test_mode,
                  const framework::proto::BlockDesc *block = nullptr) override {
    framework::OpDesc op_desc(op, nullptr);
    CHECK(block);
    const framework::BlockDesc block_desc(
        nullptr, const_cast<framework::proto::BlockDesc *>(block));

    nvinfer1::ILayer *layer = nullptr;
    std::vector<nvinfer1::ITensor *> inputs;

    const phi::ArgumentMappingFn *argument_mapping_func =
        phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_desc.Type());
    tensorrt::PluginArgumentMappingContext argument_mapping_context(&op_desc);
    phi::KernelSignature phi_kernel_signature =
        (*argument_mapping_func)(argument_mapping_context);

    plugin::GenericPlugin::InputOutPutVarInfo in_out_info;
    for (auto &param_name : phi_kernel_signature.input_names) {
      for (auto &arg_name : op_desc.Input(param_name)) {
        inputs.push_back(engine_->GetITensor(arg_name));
        LOG(INFO) << "arg_name " << arg_name;
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            platform::errors::NotFound("no variable called %s in block.",
                                       arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_LOD_TENSOR,
            platform::errors::InvalidArgument("TensorRT engine only takes "
                                              "LoDTensor as input"));
        LOG(INFO) << "var->GetType(): " << int(var->GetType());
        LOG(INFO) << "var->GetDataType(): " << int(var->GetDataType());
        in_out_info.inputs_data_type.push_back(var->GetDataType());
      }
    }

    plugin::GenericPlugin *plugin = new plugin::GenericPlugin(op, in_out_info);
    layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);

    std::vector<std::string> output_names;
    for (auto &param_name : phi_kernel_signature.output_names) {
      for (auto &arg_name : op_desc.Output(param_name)) {
        output_names.push_back(arg_name);
      }
    }
    RreplenishLayerAndOutput(layer, op_desc.Type(), output_names, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(custom_plugin_creater, CustomPluginCreater);
REGISTER_TRT_OP_CONVERTER(generic_plugin_creater, GenericPluginCreater);
