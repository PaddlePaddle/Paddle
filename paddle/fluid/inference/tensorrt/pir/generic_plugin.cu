// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <map>

#include "paddle/common/macros.h"
#include "paddle/fluid/inference/tensorrt/pir/dynamic_shape_infermeta_factory.h"
#include "paddle/fluid/inference/tensorrt/pir/dynamic_shape_infermeta_registry.h"
#include "paddle/fluid/inference/tensorrt/pir/generic_plugin.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_deserialize.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_serialize.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/pir/include/core/op_info.h"

namespace paddle::inference::tensorrt::pir {

class GatherNdOpConfig : public SpecialOpConfig {
 public:
  GatherNdOpConfig() : SpecialOpConfig(true, false, false) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (is_fp16_supported &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // output
    if (pos == 2)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  }
};

class YoloBoxOpConfig : public SpecialOpConfig {
 public:
  YoloBoxOpConfig() : SpecialOpConfig(true, false, false) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (is_fp16_supported &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // output
    if (pos == 2)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  }
};

class ScatterNdAddOpConfig : public SpecialOpConfig {
 public:
  ScatterNdAddOpConfig() : SpecialOpConfig(true, false, false) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    // input X
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (is_fp16_supported &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // input Index
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // input Updates and output
    if (pos == 2 || pos == 3)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  }
};

class EmbeddingOpConfig : public SpecialOpConfig {
 public:
  EmbeddingOpConfig() : SpecialOpConfig(true, true, false) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kINT32 &&
              (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR));
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT) ||
             ((is_fp16_supported &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
                 (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // output
    if (pos == 2)
      return in_out[1].type == in_out[pos].type &&
             in_out[1].format == in_out[pos].format;
  }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) override {
    return input_types[1];
  }
};

class ArgsortOpConfig : public SpecialOpConfig {
 public:
  ArgsortOpConfig() : SpecialOpConfig(true, true, true) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    // input x
    if (pos == 0) {
      return ((in_out[pos].type == nvinfer1::DataType::kFLOAT ||
               (is_fp16_supported &&
                in_out[pos].type == nvinfer1::DataType::kHALF)) &&
              in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
    // output out
    if (pos == 1) {
      return (in_out[pos].type == in_out[0].type &&
              in_out[pos].format == in_out[0].format);
    }
    // output indices
    if (pos == 2) {
      return (in_out[pos].type == nvinfer1::DataType::kINT32 &&
              in_out[pos].format == in_out[0].format);
    }
  }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) override {
    if (index == 1) {
      return nvinfer1::DataType::kINT32;
    } else {
      return input_types[0];
    }
  }
  void outputsPostProcess(phi::DeviceContextPool& pool,  // NOLINT
                          std::vector<phi::DenseTensor>* dense_tensor_outputs,
                          void* const* outputs) override {
    for (int i = 0; i < dense_tensor_outputs->size(); i++) {
      phi::DenseTensor& output_tensor = (*dense_tensor_outputs)[i];
      phi::DataType dtype = output_tensor.dtype();
      if (dtype == phi::DataType::INT64) {
        auto& int32_tensor = output_tensor;
        auto ctx = pool.Get(output_tensor.place());
        int32_tensor = phi::funcs::TransDataType(
            reinterpret_cast<const phi::GPUContext&>(*ctx),
            output_tensor,
            phi::DataType::INT32);
        paddle::memory::Copy(output_tensor.place(),
                             outputs[i],
                             output_tensor.place(),
                             int32_tensor.data<int32_t>(),
                             int32_tensor.numel() * sizeof(int),
                             nullptr);
      }
    }
  }
};

class ScatterOpConfig : public SpecialOpConfig {
 public:
  ScatterOpConfig() : SpecialOpConfig(true, false, false) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    // input X
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (is_fp16_supported &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // Ids
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // 3:output 2:input Updates
    if (pos == 3 || pos == 2)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  }
};

class SolveOpConfig : public SpecialOpConfig {
 public:
  SolveOpConfig() : SpecialOpConfig(true, false, false) {}
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs,
                                 bool is_fp16_supported) override {
    // input X
    if (pos == 0)
      return in_out[pos].type == nvinfer1::DataType::kFLOAT &&
             in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // input Y
    if (pos == 1)
      return in_out[pos].type == nvinfer1::DataType::kFLOAT &&
             in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // output
    if (pos == 2)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  }
};

GenericPlugin::GenericPlugin(const std::string& op_name,
                             const std::string& attrs_map_info,
                             const std::vector<std::string>& inputs_type_info,
                             const std::vector<std::string>& outputs_type_info,
                             bool with_fp16) {
  op_name_ = op_name;
  attrs_map_info_ = attrs_map_info;
  inputs_type_info_ = inputs_type_info;
  outputs_type_info_ = outputs_type_info;
  ::pir::OpInfo op_info =
      ::pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  auto op_info_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  if (op_info_interface) {
    op_yaml_info_ = std::make_unique<paddle::dialect::OpYamlInfoParser>(
        op_info_interface->get_op_info_(op_name),
        paddle::dialect::IsLegacyOp(op_name));
  }
  ::pir::ProgramReader reader(1);
  auto attrs_json_data = Json::parse(attrs_map_info);
  attrs_map_ = reader.RecoverOpAttributesMap(&attrs_json_data);
  for (auto input_type_info : inputs_type_info) {
    auto type_json_data = Json::parse(input_type_info);
    inputs_type_.push_back(reader.RecoverType(&type_json_data));
  }
  for (auto output_type_info : outputs_type_info) {
    auto type_json_data = Json::parse(output_type_info);
    outputs_type_.push_back(reader.RecoverType(&type_json_data));
  }
  with_fp16_ = with_fp16;

  // Add special op config for deal with special situation
  special_op_config_["pd_op.gather_nd"] = std::make_unique<GatherNdOpConfig>();
  special_op_config_["pd_op.yolo_box"] = std::make_unique<YoloBoxOpConfig>();
  special_op_config_["pd_op.scatter_nd_add"] =
      std::make_unique<ScatterNdAddOpConfig>();
  special_op_config_["pd_op.embedding"] = std::make_unique<EmbeddingOpConfig>();
  special_op_config_["pd_op.argsort"] = std::make_unique<ArgsortOpConfig>();
  special_op_config_["pd_op.scatter"] = std::make_unique<ScatterOpConfig>();
  special_op_config_["pd_op.solve"] = std::make_unique<SolveOpConfig>();
}

GenericPlugin::GenericPlugin(void const* serial_data, size_t serial_length) {
  // deserialize with_fp16_
  paddle::platform::DeserializeValue(&serial_data, &serial_length, &with_fp16_);
  // deserialize op_name
  int op_name_size = 0;
  paddle::platform::DeserializeValue(
      &serial_data, &serial_length, &op_name_size);
  std::string op_name((char*)(serial_data), op_name_size);  // NOLINT
  op_name_ = std::move(op_name);
  reinterpret_cast<char const*&>(serial_data) += op_name_size;
  serial_length -= op_name_size;
  // deserialize attrs_map
  int attrs_map_info_size = 0;
  paddle::platform::DeserializeValue(
      &serial_data, &serial_length, &attrs_map_info_size);
  std::string attrs_map_info(reinterpret_cast<char const*&>(serial_data),
                             attrs_map_info_size);  // NOLINT
  attrs_map_info_ = std::move(attrs_map_info);
  reinterpret_cast<char const*&>(serial_data) += attrs_map_info_size;
  serial_length -= attrs_map_info_size;

  // deserialize inputs_type_info_
  int inputs_type_info_size = 0;
  paddle::platform::DeserializeValue(
      &serial_data, &serial_length, &inputs_type_info_size);
  for (int i = 0; i < inputs_type_info_size; i++) {
    int input_type_info_size = 0;
    paddle::platform::DeserializeValue(
        &serial_data, &serial_length, &input_type_info_size);
    std::string input_type_info(reinterpret_cast<char const*&>(serial_data),
                                input_type_info_size);  // NOLINT
    reinterpret_cast<char const*&>(serial_data) += input_type_info_size;
    serial_length -= input_type_info_size;
    inputs_type_info_.push_back(input_type_info);
  }
  // deserialize outputs_type_info_
  int outputs_type_info_size = 0;
  paddle::platform::DeserializeValue(
      &serial_data, &serial_length, &outputs_type_info_size);
  for (int i = 0; i < outputs_type_info_size; i++) {
    int output_type_info_size = 0;
    paddle::platform::DeserializeValue(
        &serial_data, &serial_length, &output_type_info_size);
    std::string output_type_info(reinterpret_cast<char const*&>(serial_data),
                                 output_type_info_size);  // NOLINT
    reinterpret_cast<char const*&>(serial_data) += output_type_info_size;
    serial_length -= output_type_info_size;
    outputs_type_info_.push_back(output_type_info);
  }

  ::pir::OpInfo op_info =
      ::pir::IrContext::Instance()->GetRegisteredOpInfo(op_name_);
  auto op_info_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  if (op_info_interface) {
    op_yaml_info_ = std::make_unique<paddle::dialect::OpYamlInfoParser>(
        op_info_interface->get_op_info_(op_name),
        paddle::dialect::IsLegacyOp(op_name_));
  }

  ::pir::ProgramReader reader(1);
  auto attrs_json_data = Json::parse(attrs_map_info_);
  attrs_map_ = reader.RecoverOpAttributesMap(&attrs_json_data);
  for (auto input_type_info : inputs_type_info_) {
    auto type_json_data = Json::parse(input_type_info);
    inputs_type_.push_back(reader.RecoverType(&type_json_data));
  }
  for (auto output_type_info : outputs_type_info_) {
    auto type_json_data = Json::parse(output_type_info);
    outputs_type_.push_back(reader.RecoverType(&type_json_data));
  }
}

int GenericPlugin::getNbOutputs() const TRT_NOEXCEPT {
  int num = 0;
  for (auto output_type : outputs_type_) {
    if (output_type.isa<::pir::VectorType>()) {
      num += output_type.dyn_cast<::pir::VectorType>().size();
    } else {
      num++;
    }
  }
  return num;
}

int GenericPlugin::getNbInputs() const TRT_NOEXCEPT {
  int num = 0;
  for (auto input_type : inputs_type_) {
    if (input_type.isa<::pir::VectorType>()) {
      num += input_type.dyn_cast<::pir::VectorType>().size();
    } else {
      num++;
    }
  }
  return num;
}

nvinfer1::IPluginV2DynamicExt* GenericPlugin::clone() const TRT_NOEXCEPT {
  nvinfer1::IPluginV2DynamicExt* plugin = new GenericPlugin(op_name_,
                                                            attrs_map_info_,
                                                            inputs_type_info_,
                                                            outputs_type_info_,
                                                            with_fp16_);
  plugin->initialize();
  return plugin;
}

void GenericPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  // use fp16
  paddle::platform::SerializeValue(&buffer, with_fp16_);
  // serialize op_name_
  paddle::platform::SerializeValue(&buffer, static_cast<int>(op_name_.size()));
  std::memcpy(buffer, op_name_.c_str(), op_name_.size());
  reinterpret_cast<char*&>(buffer) += op_name_.size();
  // serialize attrs_map_info_
  paddle::platform::SerializeValue(&buffer,
                                   static_cast<int>(attrs_map_info_.size()));
  std::memcpy(buffer, attrs_map_info_.c_str(), attrs_map_info_.size());
  reinterpret_cast<char*&>(buffer) += attrs_map_info_.size();
  // serialize inputs_type_info_
  paddle::platform::SerializeValue(&buffer,
                                   static_cast<int>(inputs_type_info_.size()));
  for (auto input_type_info : inputs_type_info_) {
    paddle::platform::SerializeValue(&buffer,
                                     static_cast<int>(input_type_info.size()));
    std::memcpy(buffer, input_type_info.c_str(), input_type_info.size());
    reinterpret_cast<char*&>(buffer) += input_type_info.size();
  }
  // serialize outputs_type_info_
  paddle::platform::SerializeValue(&buffer,
                                   static_cast<int>(outputs_type_info_.size()));
  for (auto output_type_info : outputs_type_info_) {
    paddle::platform::SerializeValue(&buffer,
                                     static_cast<int>(output_type_info.size()));
    std::memcpy(buffer, output_type_info.c_str(), output_type_info.size());
    reinterpret_cast<char*&>(buffer) += output_type_info.size();
  }
}

bool GenericPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  if (special_op_config_.find(op_name_) != special_op_config_.end() &&
      special_op_config_[op_name_]->HasFormatCombinationFunc()) {
    return special_op_config_[op_name_]->supportsFormatCombination(
        pos, in_out, nb_inputs, nb_outputs, isFp16Supported());
  } else {
    return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
            (isFp16Supported() &&
             in_out[pos].type == nvinfer1::DataType::kHALF)) &&
           (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR) &&
           (in_out[0].type == in_out[pos].type);
  }
}

nvinfer1::DataType GenericPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (special_op_config_.find(op_name_) != special_op_config_.end() &&
      special_op_config_.at(op_name_)->HasGetOutputDataTypeFunc()) {
    return special_op_config_.at(op_name_)->getOutputDataType(
        index, input_types, nb_inputs);
  }
  return input_types[0];
}

int GenericPlugin::initialize() TRT_NOEXCEPT {
  std::string kernel_func = op_yaml_info_->OpRuntimeInfo().kernel_func;

  PADDLE_ENFORCE_EQ(
      phi::KernelFactory::Instance().HasCompatiblePhiKernel(kernel_func),
      true,
      common::errors::Fatal("%s has no compatible phi kernel!",
                            op_name_.c_str()));

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  phi::GPUPlace place(phi::backends::gpu::GetCurrentDeviceId());
  auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));

  std::vector<phi::DataType> precision_types{phi::DataType::FLOAT32,
                                             phi::DataType::FLOAT16};
  for (auto& precision_type : precision_types) {
    phi::KernelKey phi_kernel_key(
        phi::Backend::GPU, phi::DataLayout::ANY, precision_type);

    auto nv_dtype = paddle::platform::PhiType2NvType(precision_type);
    phi_kernels_[nv_dtype] = std::make_unique<phi::Kernel>(
        phi::KernelFactory::Instance().SelectKernel(kernel_func,
                                                    phi_kernel_key));

    if (phi_kernel_contexts_.find(nv_dtype) == phi_kernel_contexts_.end() ||
        !phi_kernel_contexts_[nv_dtype]) {
      phi_kernel_contexts_[nv_dtype] =
          std::make_unique<phi::KernelContext>(dev_ctx);
    }
  }
  PADDLE_ENFORCE_EQ(
      phi_kernels_[nvinfer1::DataType::kFLOAT]->IsValid() ||
          phi_kernels_[nvinfer1::DataType::kHALF]->IsValid(),
      true,
      common::errors::Fatal("%s phi kernel is invalid!.", kernel_func));

  if (!dense_tensor_inputs_)
    dense_tensor_inputs_ = new std::vector<phi::DenseTensor>(getNbInputs());
  if (!dense_tensor_outputs_)
    dense_tensor_outputs_ = new std::vector<phi::DenseTensor>(getNbOutputs());
  return 0;
}

nvinfer1::DimsExprs GenericPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  CHECK(output_index < getNbOutputs());
  auto& dynamic_infermeta_factory = DynamicMetaFnFactory::Instance();
  auto op_name_without_dialect = op_name_;
  auto pos = op_name_.find_last_of(".");
  if (pos != std::string::npos) {
    op_name_without_dialect = op_name_.substr(pos + 1);
  }
  PADDLE_ENFORCE_EQ(
      dynamic_infermeta_factory.Contains(op_name_without_dialect),
      true,
      common::errors::InvalidArgument(
          "The %s op has no dynamic plugin infershape function!", op_name_));

  auto* infershape_func =
      dynamic_infermeta_factory.Get(op_name_without_dialect);
  return infershape_func(
      output_index, inputs, nb_inputs, expr_builder, attrs_map_);
}

void GenericPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) TRT_NOEXCEPT {
  CHECK(phi_kernels_[nvinfer1::DataType::kFLOAT]->IsValid() ||
        phi_kernels_[nvinfer1::DataType::kHALF]->IsValid());
  CHECK(nb_inputs == getNbInputs());
  CHECK(nb_outputs == getNbOutputs());
}

// Shutdown the layer. This is called when the engine is destroyed
void GenericPlugin::terminate() TRT_NOEXCEPT {
  delete dense_tensor_inputs_;
  delete dense_tensor_outputs_;
}

int GenericPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                           const nvinfer1::PluginTensorDesc* output_desc,
                           const void* const* inputs,
                           void* const* outputs,
                           void* workspace,
                           cudaStream_t stream) TRT_NOEXCEPT {
  phi::GPUPlace place(phi::backends::gpu::GetCurrentDeviceId());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  // TODO(inference): generic plugin do not support INT8 precision now.
  auto nvType2PhiType =
      [&](nvinfer1::DataType nv_dtype) -> std::pair<phi::DataType, int> {
    const std::map<nvinfer1::DataType, std::pair<phi::DataType, int>> _map{
        {nvinfer1::DataType::kFLOAT, {phi::DataType::FLOAT32, sizeof(float)}},
        {nvinfer1::DataType::kHALF, {phi::DataType::FLOAT16, sizeof(half)}},
        {nvinfer1::DataType::kINT32, {phi::DataType::INT32, sizeof(int32_t)}},
        {nvinfer1::DataType::kBOOL, {phi::DataType::BOOL, sizeof(bool)}},
    };
    CHECK(_map.count(nv_dtype))
        << "dtype [" << static_cast<int>(nv_dtype) << "] is not supported.";
    return _map.at(nv_dtype);
  };

  nvinfer1::DataType data_type;
  // input
  if (op_name_ == "pd_op.embedding") {
    data_type = input_desc[1].type;
  } else {
    data_type = input_desc[0].type;
  }
  CHECK((data_type == nvinfer1::DataType::kFLOAT) ||
        (data_type == nvinfer1::DataType::kHALF));

  phi_kernel_contexts_[data_type]->ClearInputOutput();

  auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));
  phi_kernel_contexts_[data_type]->SetDeviceContext(dev_ctx);

  auto& vec_kernel_fn_tensor_params = op_yaml_info_->TensorParams(true);
  int kernel_input_count = vec_kernel_fn_tensor_params.size();
  for (int i = 0; i < getNbInputs(); i++) {
    // Tensor Input
    if (!inputs_type_[i]) {
      phi_kernel_contexts_[data_type]->EmplaceBackInput(nullptr);
      continue;
    }
    auto const& input_dims = input_desc[i].dims;

    std::vector<int> input_shape;
    for (int j = 0; j < input_dims.nbDims; j++)
      input_shape.push_back(input_dims.d[j]);

    int input_numel = 1;
    for (int k = 0; k < input_shape.size(); k++) input_numel *= input_shape[k];
    auto data_type_and_size = nvType2PhiType(input_desc[i].type);
    phi::DenseTensorMeta input_meta(data_type_and_size.first,
                                    common::make_ddim(input_shape));
    std::shared_ptr<phi::Allocation> input_alloc(
        new phi::Allocation((void*)(inputs[i]),  // NOLINT
                            input_numel * data_type_and_size.second,
                            place));
    (*dense_tensor_inputs_)[i] =
        std::move(phi::DenseTensor(input_alloc, input_meta));
    if (i < kernel_input_count) {
      phi_kernel_contexts_[data_type]->EmplaceBackInput(
          &((*dense_tensor_inputs_)[i]));
    }
  }
  VLOG(8) << "EmplaceBackBackInput done";
  // attribute
  auto& name2id = op_yaml_info_->InputName2Id();
  auto& vec_kernel_fn_attr_params = op_yaml_info_->AttrParams(true);
  int tensor_attr_count = 0;
  for (auto& t : vec_kernel_fn_attr_params) {
    if (name2id.count(t)) {
      // tensor attribute, get information from input
      tensor_attr_count++;
      PADDLE_ENFORCE_LE(tensor_attr_count + kernel_input_count,
                        getNbInputs(),
                        common::errors::OutOfRange(
                            "The set input tensor number is %d, but got %d "
                            "that is greater than set input tensor num.",
                            getNbInputs(),
                            tensor_attr_count + kernel_input_count));
      auto operand_type = inputs_type_[name2id.at(t)];

      auto& tensor_attr_type = op_yaml_info_->TensorAttrTypeName(t);
      VLOG(6) << "ctx->EmplaceBack mutable attr: " << t;
      int tensor_index = kernel_input_count + tensor_attr_count - 1;
      if (tensor_attr_type == "paddle::dialect::IntArrayAttribute") {
        if (operand_type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
          phi::Attribute attr =
              phi::TensorRef(&((*dense_tensor_inputs_)[tensor_index]));
          phi_kernel_contexts_[data_type]->EmplaceBackAttr(attr);
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              " [%s] only support dense tensor ", tensor_attr_type));
        }
      } else if (tensor_attr_type == "paddle::dialect::ScalarAttribute") {
        phi::Attribute attr =
            phi::TensorRef(&((*dense_tensor_inputs_)[tensor_index]));

        phi_kernel_contexts_[data_type]->EmplaceBackAttr(attr);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "attr type not support [%s] ", tensor_attr_type));
      }

      continue;
    }

    PADDLE_ENFORCE_NE(
        attrs_map_.find(t),
        attrs_map_.end(),
        common::errors::NotFound("Not found %s in attrs_map_, please check "
                                 "attrs_map_info when construct GenericPlugin.",
                                 t));
    auto& attr_type_name = op_yaml_info_->AttrTypeName(t);
    if (attr_type_name == "paddle::dialect::IntArrayAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::DataTypeAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
    } else if (attr_type_name == "pir::Int32Attribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<::pir::Int32Attribute>().data());
    } else if (attr_type_name == "pir::Int64Attribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<::pir::Int64Attribute>().data());
    } else if (attr_type_name == "pir::FloatAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<::pir::FloatAttribute>().data());
    } else if (attr_type_name == "pir::DoubleAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<::pir::DoubleAttribute>().data());
    } else if (attr_type_name == "pir::BoolAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<::pir::BoolAttribute>().data());
    } else if (attr_type_name == "pir::StrAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<::pir::StrAttribute>().AsString());
    } else if (attr_type_name ==
               "pir::ArrayAttribute<paddle::dialect::ScalarAttribute>") {
      auto array_list =
          attrs_map_[t].dyn_cast<::pir::ArrayAttribute>().AsVector();
      std::vector<phi::Scalar> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<paddle::dialect::ScalarAttribute>(),
            true,
            common::errors::Unimplemented(
                "the 0th elementwise MUST be dialect::ScalarAttribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(array_list[i]
                                .dyn_cast<paddle::dialect::ScalarAttribute>()
                                .data());
        }
      }
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<::pir::Int32Attribute>") {
      auto array_list =
          attrs_map_[t].dyn_cast<::pir::ArrayAttribute>().AsVector();
      std::vector<int32_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<::pir::Int32Attribute>(),
            true,
            common::errors::Unimplemented(
                "the 0th elementwise MUST be ::pir::Int32Attribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<::pir::Int32Attribute>().data());
        }
      }
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<::pir::FloatAttribute>") {
      auto array_list =
          attrs_map_[t].dyn_cast<::pir::ArrayAttribute>().AsVector();
      std::vector<float> vec_res;
      if (array_list.size() > 0) {
        if (array_list[0].isa<::pir::FloatAttribute>()) {
          for (size_t i = 0; i < array_list.size(); ++i) {
            vec_res.push_back(
                array_list[i].dyn_cast<::pir::FloatAttribute>().data());
          }

        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "attr type not support [%s] ", attr_type_name));
        }
      }
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<::pir::Int64Attribute>") {
      auto array_list =
          attrs_map_[t].dyn_cast<::pir::ArrayAttribute>().AsVector();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<::pir::Int64Attribute>(),
            true,
            common::errors::PreconditionNotMet(
                "Element in array list MUST be ::pir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<::pir::Int64Attribute>().data());
        }
      }
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<::pir::Int64Attribute>") {
      auto array_list =
          attrs_map_[t].dyn_cast<::pir::ArrayAttribute>().AsVector();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<::pir::Int64Attribute>(),
            true,
            common::errors::PreconditionNotMet(
                "Element in array list MUST be ::pir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<::pir::Int64Attribute>().data());
        }
      }
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(vec_res);

    } else if (attr_type_name == "pir::ArrayAttribute<::pir::StrAttribute>") {
      auto array_list =
          attrs_map_[t].dyn_cast<::pir::ArrayAttribute>().AsVector();

      std::vector<std::string> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<::pir::StrAttribute>(),
            true,
            common::errors::PreconditionNotMet(
                "Element in array list MUST be ::pir::StrAttribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<::pir::StrAttribute>().AsString());
        }
      }
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(vec_res);

    } else if (attr_type_name == "paddle::dialect::PlaceAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::ScalarAttribute") {
      phi_kernel_contexts_[data_type]->EmplaceBackAttr(
          attrs_map_[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
    } else {
      PADDLE_THROW(common::errors::Unimplemented("attr type not support [%s] ",
                                                 attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }
  VLOG(8) << "EmplaceBackBackAttributes done";

  // output
  for (int i = 0; i < getNbOutputs(); i++) {
    auto const& output_dims = output_desc[i].dims;

    std::vector<int> output_shape;
    for (int j = 0; j < output_dims.nbDims; j++)
      output_shape.push_back(output_dims.d[j]);

    int output_numel = 1;
    for (int k = 0; k < output_shape.size(); k++)
      output_numel *= output_shape[k];

    auto data_type_and_size = nvType2PhiType(output_desc[i].type);
    phi::DenseTensorMeta output_meta(data_type_and_size.first,
                                     common::make_ddim(output_shape));
    std::shared_ptr<phi::Allocation> output_alloc(
        new phi::Allocation(reinterpret_cast<void*>(outputs[i]),
                            output_numel * data_type_and_size.second,
                            place));

    (*dense_tensor_outputs_)[i] =
        std::move(phi::DenseTensor(output_alloc, output_meta));

    phi_kernel_contexts_[data_type]->EmplaceBackOutput(
        &((*dense_tensor_outputs_)[i]));
  }
  VLOG(8) << "EmplaceBackBackOutput done";
  CHECK_EQ(phi_kernel_contexts_[data_type]->InputsSize(), getNbInputs());
  CHECK_EQ(phi_kernel_contexts_[data_type]->OutputsSize(), getNbOutputs());
  (*phi_kernels_[data_type])(phi_kernel_contexts_[data_type].get());

  if (special_op_config_.find(op_name_) != special_op_config_.end() &&
      special_op_config_[op_name_]->HasOutputsPostProcessFunc()) {
    special_op_config_[op_name_]->outputsPostProcess(
        pool, dense_tensor_outputs_, outputs);
  }
  return cudaGetLastError() != cudaSuccess;
}

nvinfer1::IPluginV2* PIRGenericPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
  std::string op_name;
  std::string attrs_map_info;
  std::vector<std::string> inputs_type_info;
  std::vector<std::string> outputs_type_info;
  bool with_fp16 = false;

  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    if (field_name.compare("op_name") == 0) {
      op_name = std::string(static_cast<const char*>(fc->fields[i].data),
                            fc->fields[i].length);
    } else if (field_name.compare("attrs_map_info") == 0) {
      attrs_map_info = std::string(static_cast<const char*>(fc->fields[i].data),
                                   fc->fields[i].length);
    } else if (field_name.compare("inputs_type_info") == 0) {
      std::string all_inputs_type_info(
          static_cast<const char*>(fc->fields[i].data), fc->fields[i].length);
      std::stringstream recovered_info(all_inputs_type_info);
      std::string item;
      while (std::getline(recovered_info, item, '\n')) {
        inputs_type_info.push_back(item);
      }
    } else if (field_name.compare("outputs_type_info") == 0) {
      std::string all_outputs_type_info(
          static_cast<const char*>(fc->fields[i].data), fc->fields[i].length);
      std::stringstream recovered_info(all_outputs_type_info);
      std::string item;
      while (std::getline(recovered_info, item, '\n')) {
        outputs_type_info.push_back(item);
      }
    } else if (field_name.compare("with_fp16") == 0) {
      with_fp16 = *static_cast<const bool*>(fc->fields[i].data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }
  return new GenericPlugin(
      op_name, attrs_map_info, inputs_type_info, outputs_type_info, with_fp16);
}

REGISTER_TRT_PLUGIN_V2(PIRGenericPluginCreator);
}  // namespace paddle::inference::tensorrt::pir

REGISTER_FILE_SYMBOLS(generic_plugin);
