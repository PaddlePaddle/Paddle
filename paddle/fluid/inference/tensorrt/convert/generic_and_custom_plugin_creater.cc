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

#include "paddle/common/errors.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/custom_generic_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/generic_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/enforce.h"

namespace paddle::inference::tensorrt {

class CustomPluginCreater : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert " << op_desc.Type() << " op to custom plugin layer";

    std::string plugin_name;

    plugin_name = op_desc.Type() + "_paddle_trt_dynamic_plugin";

    nvinfer1::ILayer *layer = nullptr;
    std::vector<nvinfer1::ITensor *> inputs;

    auto &op_meta_info_map = OpMetaInfoMap::Instance();
    const auto &meta_info_map = op_meta_info_map.GetMap();
    auto &op_info = meta_info_map.at(op_desc.Type()).front();

    // set inputs
    auto &op_input_names = OpMetaInfoHelper::GetInputs(op_info);
    for (auto &param_name : op_input_names) {
      for (auto &arg_name : op_desc.Input(param_name)) {
        inputs.push_back(engine_->GetITensor(arg_name));
      }
    }
    auto creator =
        GetPluginRegistry()->getPluginCreator(plugin_name.c_str(), "1");
    PADDLE_ENFORCE_NOT_NULL(creator,
                            common::errors::PreconditionNotMet(
                                "The op's plugin should be PluginV2."));

    // set attrs
    std::vector<nvinfer1::PluginField> plugin_datas;
    auto &op_attrs_names = OpMetaInfoHelper::GetAttrs(op_info);
    auto &attrs = op_desc.GetAttrMap();

    std::list<int> int_attrs;
    std::list<float> float_attrs;
    std::list<double> bool_attrs;
    std::list<std::string> string_attrs;
    std::list<std::vector<int>> ints_attrs;
    std::list<std::vector<float>> floats_attrs;

    for (auto &attr_name_and_type : op_attrs_names) {
      auto attr_name =
          attr_name_and_type.substr(0, attr_name_and_type.find_first_of(":"));
      nvinfer1::PluginField plugin_data;

      // NOTE: to avoid string rewrite by iterator, deep copy here
      std::vector<char> plugin_attr_name(attr_name.length() + 1, 0);
      snprintf(plugin_attr_name.data(),
               attr_name.length() + 1,
               "%s",
               attr_name.c_str());
      plugin_data.name = plugin_attr_name.data();

      if (op_desc.GetAttrType(attr_name) == framework::proto::AttrType::INT) {
        int_attrs.push_back(PADDLE_GET_CONST(int, attrs.at(attr_name)));
        plugin_data.data = &int_attrs.back();
        plugin_data.type = nvinfer1::PluginFieldType::kINT32;
        plugin_data.length = 1;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::FLOAT) {
        float_attrs.push_back(PADDLE_GET_CONST(float, attrs.at(attr_name)));
        plugin_data.data = &float_attrs.back();
        plugin_data.type = nvinfer1::PluginFieldType::kFLOAT32;
        plugin_data.length = 1;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::BOOLEAN) {
        int_attrs.push_back(PADDLE_GET_CONST(bool, attrs.at(attr_name)));
        plugin_data.data = &int_attrs.back();
        plugin_data.type = nvinfer1::PluginFieldType::kINT32;
        plugin_data.length = 1;
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::STRING) {
        string_attrs.push_back(
            PADDLE_GET_CONST(std::string, attrs.at(attr_name)));
        plugin_data.data = string_attrs.back().data();
        plugin_data.type = nvinfer1::PluginFieldType::kCHAR;
        plugin_data.length =
            string_attrs.back().size() + 1;  // string ends with ‘\0’
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::INTS) {
        ints_attrs.push_back(
            PADDLE_GET_CONST(std::vector<int>, attrs.at(attr_name)));
        plugin_data.data = ints_attrs.back().data();
        plugin_data.type = nvinfer1::PluginFieldType::kINT32;
        plugin_data.length = ints_attrs.back().size();
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::FLOATS) {
        floats_attrs.push_back(
            PADDLE_GET_CONST(std::vector<float>, attrs.at(attr_name)));
        plugin_data.data = floats_attrs.back().data();
        plugin_data.type = nvinfer1::PluginFieldType::kFLOAT32;
        plugin_data.length = floats_attrs.back().size();
      } else if (op_desc.GetAttrType(attr_name) ==
                 framework::proto::AttrType::BOOLEANS) {
        auto bools_attr =
            PADDLE_GET_CONST(std::vector<bool>, attrs.at(attr_name));
        std::vector<int> convert_to_ints_attr;
        for (bool i : bools_attr) convert_to_ints_attr.push_back(i);
        ints_attrs.push_back(convert_to_ints_attr);
        plugin_data.data = ints_attrs.back().data();
        plugin_data.type = nvinfer1::PluginFieldType::kINT32;
        plugin_data.length = ints_attrs.back().size();
      } else {
        PADDLE_THROW(
            common::errors::PreconditionNotMet("UNKNOWN PluginFieldType."));
      }
      plugin_datas.push_back(plugin_data);
    }

    nvinfer1::PluginFieldCollection plugin_fc{(int32_t)plugin_datas.size(),
                                              plugin_datas.data()};

    auto *plugin = creator->createPlugin(op_desc.Type().c_str(), &plugin_fc);
    PADDLE_ENFORCE_NOT_NULL(
        plugin, common::errors::NotFound("Sorry,create plugin failed."));

    layer = engine_->AddDynamicPlugin(
        inputs.data(), inputs.size(), (plugin::DynamicPluginTensorRT *)plugin);

    PADDLE_ENFORCE_NOT_NULL(
        layer, common::errors::NotFound("Sorry, add plugin layer failed."));
    // set outputs
    auto &op_output_names = OpMetaInfoHelper::GetOutputs(op_info);
    std::vector<std::string> output_names;
    for (auto &param_name : op_output_names) {
      for (auto &arg_name : op_desc.Output(param_name))
        output_names.push_back(arg_name);
    }

    ReplenishLayerAndOutput(layer, op_desc.Type(), output_names, test_mode);
  }
};

class GenericPluginCreator : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert " << op_desc.Type() << " op to generic plugin layer";

    PADDLE_ENFORCE_NOT_NULL(
        block_,
        common::errors::NotFound("The block is null, can not find the block."));
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
    VLOG(3) << phi_kernel_signature;
    PADDLE_ENFORCE_EQ(
        phi_kernel_signature.input_names.empty() ||
            phi_kernel_signature.output_names.empty(),
        false,
        common::errors::PreconditionNotMet(
            "The %s op's kernel signature (inputs and output) should not null.",
            op_desc.Type()));

    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

    plugin::GenericPlugin::InputOutPutVarInfo in_out_info;
    using paddle::inference::tensorrt::plugin::
        ProtoTypeToGeneratePluginDataType;
    for (auto &param_name : phi_kernel_signature.input_names) {
      for (auto &arg_name : op_desc.Input(param_name)) {
        inputs.push_back(engine_->GetITensor(arg_name));
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            common::errors::NotFound("There is no variable called %s in block.",
                                     arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_DENSE_TENSOR,
            common::errors::InvalidArgument("TensorRT engine only takes "
                                            "DenseTensor as input"));
        in_out_info.inputs_data_type.push_back(
            ProtoTypeToGeneratePluginDataType(var->GetDataType()));
      }
    }

    std::vector<std::string> output_names;
    for (auto &param_name : phi_kernel_signature.output_names) {
      for (auto &arg_name : op_desc.Output(param_name)) {
        output_names.push_back(arg_name);
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            common::errors::NotFound("There is no variable called %s in block.",
                                     arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_DENSE_TENSOR,
            common::errors::InvalidArgument("TensorRT engine only takes "
                                            "DenseTensor as input"));
        in_out_info.outputs_data_type.push_back(
            ProtoTypeToGeneratePluginDataType(var->GetDataType()));
      }
    }
    plugin::GenericPlugin *plugin =
        new plugin::GenericPlugin(op, in_out_info, with_fp16);
    layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);

    ReplenishLayerAndOutput(layer, op_desc.Type(), output_names, test_mode);
  }
};

class CustomGenericPluginCreator : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert " << op_desc.Type()
            << " op to custom generic plugin layer";

    nvinfer1::ILayer *layer = nullptr;
    std::vector<nvinfer1::ITensor *> inputs;
    PADDLE_ENFORCE_NOT_NULL(
        block_,
        common::errors::NotFound("The block is null, can not find the block."));
    const framework::BlockDesc block_desc(
        nullptr, const_cast<framework::proto::BlockDesc *>(block_));

    plugin::CustomGenericPlugin::InputOutPutVarInfo in_out_info;
    using paddle::inference::tensorrt::plugin::
        ProtoTypeToGenerateCustomGenericPluginDataType;

    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

    auto &op_meta_info_map = OpMetaInfoMap::Instance();
    const auto &meta_info_map = op_meta_info_map.GetMap();
    auto &op_info = meta_info_map.at(op_desc.Type()).front();

    // set inputs
    auto &op_input_names = OpMetaInfoHelper::GetInputs(op_info);
    paddle::small_vector<const char *> input_names;
    for (auto &input_name : op_input_names) {
      input_names.emplace_back(input_name.c_str());
    }
    for (auto &param_name : input_names) {
      for (auto &arg_name : op_desc.Input(param_name)) {
        inputs.push_back(engine_->GetITensor(arg_name));
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            common::errors::NotFound("There is no variable called %s in block.",
                                     arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_DENSE_TENSOR,
            common::errors::InvalidArgument("TensorRT engine only takes "
                                            "DenseTensor as input"));
        in_out_info.inputs_data_type.push_back(
            ProtoTypeToGenerateCustomGenericPluginDataType(var->GetDataType()));
      }
    }

    // set outputs
    auto &op_output_names = OpMetaInfoHelper::GetOutputs(op_info);
    paddle::small_vector<const char *> output_names;
    for (auto &output_name : op_output_names) {
      output_names.emplace_back(output_name.c_str());
    }
    std::vector<std::string> outputs;
    for (auto &param_name : output_names) {
      for (auto &arg_name : op_desc.Output(param_name)) {
        outputs.push_back(arg_name);
        auto *var = block_desc.FindVar(arg_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            common::errors::NotFound("There is no variable called %s in block.",
                                     arg_name.c_str()));
        PADDLE_ENFORCE_EQ(
            var->GetType(),
            FluidDT::VarType_Type_DENSE_TENSOR,
            common::errors::InvalidArgument("TensorRT engine only takes "
                                            "DenseTensor as input"));
        in_out_info.outputs_data_type.push_back(
            ProtoTypeToGenerateCustomGenericPluginDataType(var->GetDataType()));
      }
    }

    auto *plugin = new plugin::CustomGenericPlugin(op, in_out_info, with_fp16);
    PADDLE_ENFORCE_NOT_NULL(
        plugin, common::errors::NotFound("Sorry,create plugin failed."));
    layer = engine_->AddDynamicPlugin(
        inputs.data(), inputs.size(), (plugin::DynamicPluginTensorRT *)plugin);
    PADDLE_ENFORCE_NOT_NULL(
        layer, common::errors::NotFound("Sorry,add plugin layer failed."));
    ReplenishLayerAndOutput(layer, op_desc.Type(), outputs, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(custom_plugin_creater,
                          CustomPluginCreater);  // typos: disable-line
REGISTER_TRT_OP_CONVERTER(generic_plugin_creator, GenericPluginCreator);
REGISTER_TRT_OP_CONVERTER(custom_generic_plugin_creator,
                          CustomGenericPluginCreator);
