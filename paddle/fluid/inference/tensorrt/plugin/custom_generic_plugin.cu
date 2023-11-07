// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/custom_generic_plugin.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/custom_generic_plugin_fn_factory.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

GenerateCustomPluginDataType ProtoTypeToGenerateCustomPluginDataType(
    framework::proto::VarType_Type proto_type) {
  using framework::proto::VarType_Type;
  switch (proto_type) {
    case VarType_Type::VarType_Type_BOOL:
      return GenerateCustomPluginDataType::PLUGIN_BOOL;
    case VarType_Type::VarType_Type_UINT8:
      return GenerateCustomPluginDataType::PLUGIN_UINT8;
    case VarType_Type::VarType_Type_INT8:
      return GenerateCustomPluginDataType::PLUGIN_INT8;
    case VarType_Type::VarType_Type_INT16:
      return GenerateCustomPluginDataType::PLUGIN_INT16;
    case VarType_Type::VarType_Type_INT32:
      return GenerateCustomPluginDataType::PLUGIN_INT32;
    case VarType_Type::VarType_Type_INT64:
      return GenerateCustomPluginDataType::PLUGIN_INT64;
    case VarType_Type::VarType_Type_FP16:
      return GenerateCustomPluginDataType::PLUGIN_FP16;
    case VarType_Type::VarType_Type_FP32:
      return GenerateCustomPluginDataType::PLUGIN_FP32;
    case VarType_Type::VarType_Type_FP64:
      return GenerateCustomPluginDataType::PLUGIN_FP64;
    case VarType_Type::VarType_Type_SIZE_T:
      return GenerateCustomPluginDataType::PLUGIN_SIZE_T;
    case VarType_Type::VarType_Type_BF16:
      return GenerateCustomPluginDataType::PLUGIN_BF16;
    case VarType_Type::VarType_Type_COMPLEX64:
      return GenerateCustomPluginDataType::PLUGIN_COMPLEX64;
    case VarType_Type::VarType_Type_COMPLEX128:
      return GenerateCustomPluginDataType::PLUGIN_COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "This data type is currently not supported"));
  }
}

CustomPlugin::CustomPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const InputOutPutVarInfo& in_out_info,
    bool with_fp16) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = framework::OpDesc(proto_op_desc_, nullptr);
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = in_out_info.inputs_data_type;
  outputs_data_type_ = in_out_info.outputs_data_type;
  with_fp16_ = with_fp16;
}

CustomPlugin::CustomPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const std::vector<GenerateCustomPluginDataType>& inputs_data_type,
    const std::vector<GenerateCustomPluginDataType>& outputs_data_type,
    bool with_fp16) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = framework::OpDesc(proto_op_desc_, nullptr);
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = inputs_data_type;
  outputs_data_type_ = outputs_data_type;
  with_fp16_ = with_fp16;
}

CustomPlugin::CustomPlugin(void const* serial_data, size_t serial_length) {
  DeserializeValue(&serial_data, &serial_length, &inputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &outputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &with_fp16_);

  std::string op_meta_data((char*)(serial_data), serial_length);  // NOLINT
  op_meta_data_ = std::move(op_meta_data);
  proto_op_desc_.ParseFromString(op_meta_data_);
  op_desc_ = framework::OpDesc(proto_op_desc_, nullptr);
}

int CustomPlugin::getNbOutputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Outputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

int CustomPlugin::getNbInputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Inputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

nvinfer1::IPluginV2DynamicExt* CustomPlugin::clone() const TRT_NOEXCEPT {
  nvinfer1::IPluginV2DynamicExt* plugin = new CustomPlugin(
      proto_op_desc_, inputs_data_type_, outputs_data_type_, with_fp16_);
  plugin->initialize();
  return plugin;
}

void CustomPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  // inputs_data_type_
  SerializeValue(&buffer, inputs_data_type_);
  // outputs_data_type_
  SerializeValue(&buffer, outputs_data_type_);
  // use fp16
  SerializeValue(&buffer, with_fp16_);
  // serialize op_meta_data_
  std::memcpy(buffer, op_meta_data_.c_str(), op_meta_data_.size());
  reinterpret_cast<char*&>(buffer) += op_meta_data_.size();
}

bool CustomPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  auto& supports_formate_factory =
      tensorrt::SupportsFormateFnFactory::Instance();
  if (!supports_formate_factory.Contains(op_desc_.Type()) &&
      FLAGS_enable_auto_generate_plugin_fn) {
    PADDLE_ENFORCE_EQ(supports_formate_factory.ContainsAuto(op_desc_.Type()),
                      true,
                      platform::errors::InvalidArgument(
                          "The %s op has no tensorrt plugin "
                          "supportsFormatCombination function!"
                          "Please use SetTrtSupportFormateFn to regiser.",
                          op_desc_.Type().c_str()));
    auto* supports_formate_fn =
        supports_formate_factory.GetAuto(op_desc_.Type());
    auto& op_meta_info_map = OpMetaInfoMap::Instance();
    const auto& meta_info_map = op_meta_info_map.GetMap();
    auto& op_info = meta_info_map.at(op_desc_.Type()).front();
    return supports_formate_fn(
        pos, in_out, nb_inputs, nb_outputs, op_info, op_desc_);
  } else {
    PADDLE_ENFORCE_EQ(supports_formate_factory.Contains(op_desc_.Type()),
                      true,
                      platform::errors::InvalidArgument(
                          "The %s op has no tensorrt plugin "
                          "supportsFormatCombination function!"
                          "Please use SetTrtSupportFormateFn to regiser.",
                          op_desc_.Type().c_str()));
    auto* supports_formate_fn = supports_formate_factory.Get(op_desc_.Type());
    return supports_formate_fn(pos, in_out, nb_inputs, nb_outputs);
  }
}

nvinfer1::DataType CustomPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_NE(
      input_types,
      nullptr,
      phi::errors::Unavailable("Input type should not be nullptr."));
  return input_types[0];
}

int CustomPlugin::initialize() TRT_NOEXCEPT {
  if (!tensor_inputs_)
    tensor_inputs_ = new std::vector<paddle::Tensor>(getNbInputs());
  if (!tensor_outputs_)
    tensor_outputs_ = new std::vector<paddle::Tensor>(getNbOutputs());
  return 0;
}

nvinfer1::DimsExprs CustomPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  CHECK(output_index < getNbOutputs());
  auto& get_output_dims_factory = tensorrt::GetOutputDimsFnFactory::Instance();
  if (!get_output_dims_factory.Contains(op_desc_.Type()) &&
      FLAGS_enable_auto_generate_plugin_fn) {
    PADDLE_ENFORCE_EQ(get_output_dims_factory.ContainsAuto(op_desc_.Type()),
                      true,
                      platform::errors::InvalidArgument(
                          "The %s op has no getOutputDimensions function!"
                          "Please use SetTrtInferShapeFn to regiser.",
                          op_desc_.Type().c_str()));

    auto* infershape_func = get_output_dims_factory.GetAuto(op_desc_.Type());
    auto& op_meta_info_map = OpMetaInfoMap::Instance();
    const auto& meta_info_map = op_meta_info_map.GetMap();
    auto& op_info = meta_info_map.at(op_desc_.Type()).front();
    return infershape_func(
        output_index, inputs, nb_inputs, expr_builder, op_info, op_desc_);
  } else {
    PADDLE_ENFORCE_EQ(get_output_dims_factory.Contains(op_desc_.Type()),
                      true,
                      platform::errors::InvalidArgument(
                          "The %s op has no getOutputDimensions function!"
                          "Please use SetTrtInferShapeFn to regiser.",
                          op_desc_.Type().c_str()));

    auto* infershape_func = get_output_dims_factory.Get(op_desc_.Type());
    return infershape_func(output_index, inputs, nb_inputs, expr_builder);
  }
}

void CustomPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                   int nb_inputs,
                                   const nvinfer1::DynamicPluginTensorDesc* out,
                                   int nb_outputs) TRT_NOEXCEPT {
  CHECK(nb_inputs == getNbInputs());
  CHECK(nb_outputs == getNbOutputs());
}

// Shutdown the layer. This is called when the engine is destroyed
void CustomPlugin::terminate() TRT_NOEXCEPT {
  delete tensor_inputs_;
  delete tensor_outputs_;
}

int CustomPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                          const nvinfer1::PluginTensorDesc* output_desc,
                          const void* const* inputs,
                          void* const* outputs,
                          void* workspace,
                          cudaStream_t stream) TRT_NOEXCEPT {
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  // TODO(inference): generic plugin do not support INT8 precision now.
  auto protoType2PhiType =
      [&](GenerateCustomPluginDataType proto_type,
          nvinfer1::DataType nv_dtype) -> std::pair<phi::DataType, int> {
    if (proto_type == GenerateCustomPluginDataType::PLUGIN_FP16) {
      return {phi::DataType::FLOAT16, sizeof(half)};
    } else if (proto_type == GenerateCustomPluginDataType::PLUGIN_FP32) {
      if (isFp16Supported() && nv_dtype == nvinfer1::DataType::kHALF) {
        return {phi::DataType::FLOAT16, sizeof(half)};
      } else {
        return {phi::DataType::FLOAT32, sizeof(float)};
      }
    } else if (proto_type == GenerateCustomPluginDataType::PLUGIN_INT64) {
      return {phi::DataType::INT64, sizeof(int64_t)};
    } else if (proto_type == GenerateCustomPluginDataType::PLUGIN_INT32) {
      return {phi::DataType::INT32, sizeof(int32_t)};
    } else if (proto_type == GenerateCustomPluginDataType::PLUGIN_BOOL) {
      return {phi::DataType::BOOL, sizeof(bool)};
    } else {
      CHECK(false) << "precision is not supported";
    }
  };

  nvinfer1::DataType data_type = input_desc[0].type;
  CHECK((data_type == nvinfer1::DataType::kFLOAT) ||
        (data_type == nvinfer1::DataType::kHALF));

  paddle::CustomOpKernelContext kernel_ctx;
  // input
  for (int i = 0; i < getNbInputs(); i++) {
    if (inputs_data_type_[i] == GenerateCustomPluginDataType::PLUGIN_OPTIONAL) {
      (*tensor_inputs_)[i] = paddle::Tensor();
      continue;
    }
    auto const& input_dims = input_desc[i].dims;

    std::vector<int> input_shape;
    for (int j = 0; j < input_dims.nbDims; j++)
      input_shape.push_back(input_dims.d[j]);

    int input_numel = 1;
    for (int k : input_shape) input_numel *= k;

    auto data_type_and_size =
        protoType2PhiType(inputs_data_type_[i], data_type);

    phi::DenseTensorMeta input_meta(data_type_and_size.first,
                                    phi::make_ddim(input_shape));
    std::shared_ptr<phi::Allocation> input_alloc(
        new phi::Allocation((void*)(inputs[i]),  // NOLINT
                            input_numel * data_type_and_size.second,
                            place));
    (*tensor_inputs_)[i] = paddle::Tensor(
        std::make_shared<phi::DenseTensor>(input_alloc, input_meta));
  }
  kernel_ctx.EmplaceBackInputs(*tensor_inputs_);

  // output
  for (int i = 0; i < getNbOutputs(); i++) {
    auto const& output_dims = output_desc[i].dims;

    std::vector<int> output_shape;
    for (int j = 0; j < output_dims.nbDims; j++)
      output_shape.push_back(output_dims.d[j]);

    int output_numel = 1;
    for (int k : output_shape) output_numel *= k;

    auto data_type_and_size =
        protoType2PhiType(outputs_data_type_[i], data_type);
    phi::DenseTensorMeta output_meta(data_type_and_size.first,
                                     phi::make_ddim(output_shape));
    std::shared_ptr<phi::Allocation> output_alloc(
        new phi::Allocation(reinterpret_cast<void*>(outputs[i]),
                            output_numel * data_type_and_size.second,
                            place));
    (*tensor_outputs_)[i] = paddle::Tensor(
        std::make_shared<phi::DenseTensor>(output_alloc, output_meta));
  }
  kernel_ctx.EmplaceBackOutputs(*tensor_outputs_);
  auto& op_meta_info_map = OpMetaInfoMap::Instance();
  const auto& meta_info_map = op_meta_info_map.GetMap();
  auto& op_info = meta_info_map.at(op_desc_.Type()).front();
  auto& op_attrs_names = OpMetaInfoHelper::GetAttrs(op_info);
  auto& attrs = op_desc_.GetAttrMap();
  for (auto& op_attrs_name : op_attrs_names) {
    auto attr_name_and_type = paddle::ParseAttrStr(op_attrs_name);
    auto attr_name = attr_name_and_type[0];
    auto attr_type_str = attr_name_and_type[1];
    if (attr_type_str == "bool") {
      kernel_ctx.EmplaceBackAttr(PADDLE_GET_CONST(bool, attrs.at(attr_name)));
    } else if (attr_type_str == "int") {
      kernel_ctx.EmplaceBackAttr(PADDLE_GET_CONST(int, attrs.at(attr_name)));
    } else if (attr_type_str == "float") {
      kernel_ctx.EmplaceBackAttr(PADDLE_GET_CONST(float, attrs.at(attr_name)));
    } else if (attr_type_str == "int64_t") {
      kernel_ctx.EmplaceBackAttr(
          PADDLE_GET_CONST(int64_t, attrs.at(attr_name)));
    } else if (attr_type_str == "std::string") {
      kernel_ctx.EmplaceBackAttr(
          PADDLE_GET_CONST(std::string, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<int>") {
      kernel_ctx.EmplaceBackAttr(
          PADDLE_GET_CONST(std::vector<int>, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<float>") {
      kernel_ctx.EmplaceBackAttr(
          PADDLE_GET_CONST(std::vector<float>, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<int64_t>") {
      kernel_ctx.EmplaceBackAttr(
          PADDLE_GET_CONST(std::vector<int64_t>, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<std::string>") {
      kernel_ctx.EmplaceBackAttr(
          PADDLE_GET_CONST(std::vector<std::string>, attrs.at(attr_name)));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, Please check whether "
          "the attribute data type and data type string are matched.",
          attr_type_str));
    }
  }
  auto kernel_fn = OpMetaInfoHelper::GetKernelFn(op_info);
  kernel_ctx.UpdatePlainOutputs(OpMetaInfoHelper::GetInputs(op_info),
                                OpMetaInfoHelper::GetOutputs(op_info),
                                OpMetaInfoHelper::GetInplaceMap(op_info));
  kernel_fn(&kernel_ctx);
  kernel_ctx.AssignInplaceOutputs();
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
