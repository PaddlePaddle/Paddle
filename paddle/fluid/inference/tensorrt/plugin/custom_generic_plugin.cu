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
#include "paddle/common/enforce.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

void validate(const std::string& op_type,
              const std::string& datatype,
              const std::string& tensor_format) {
  std::unordered_set<std::string> supports_dtypes = {
      "float32", "float16", "int8", "int32"};
  std::unordered_set<std::string> supports_tensor_formats = {
      "LINEAR", "CHW32", "CHW2", "HWC8", "CHW4"};
#if IS_TRT_VERSION_GE(7200)
  supports_tensor_formats.insert("DHWC8");
#endif
#if IS_TRT_VERSION_GE(8000)
  supports_tensor_formats.insert("HWC16");
#endif
  // refer to
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#ipluginv2
  PADDLE_ENFORCE_GE(supports_dtypes.count(datatype),
                    0,
                    common::errors::InvalidArgument(
                        "custom op [%s] has unsupported datatype: [%s], "
                        "now only support: [float32, float16, int8, int32].",
                        op_type,
                        datatype));
  PADDLE_ENFORCE_GE(
      supports_tensor_formats.count(tensor_format),
      0,
      common::errors::InvalidArgument(
          "custom op [%s] has unsupported tensor format: [%s], "
          "now only support: [LINEAR, CHW32, CHW2, HWC8, CHW4, DHWC8(TensorRT "
          "7.2 and after), HWC16(TensorRT 8.0 and after)].",
          op_type,
          tensor_format));

  if (datatype == "float32") {
    std::unordered_set<std::string> supports_formats_tmp = {"LINEAR", "CHW32"};
    PADDLE_ENFORCE_GE(
        supports_formats_tmp.count(tensor_format),
        0,
        common::errors::InvalidArgument(
            "custom op [%s]: float32 only supports [LINEAR, CHW32], "
            "but got tensor format: [%s], ",
            op_type,
            tensor_format));
  }
  if (datatype == "float16") {
    std::unordered_set<std::string> supports_formats_tmp = {
        "LINEAR", "CHW2", "HWC8", "CHW4"};
#if IS_TRT_VERSION_GE(7200)
    supports_formats_tmp.insert("DHWC8");
#endif
#if IS_TRT_VERSION_GE(8000)
    supports_formats_tmp.insert("HWC16");
#endif
    PADDLE_ENFORCE_GE(supports_formats_tmp.count(tensor_format),
                      0,
                      common::errors::InvalidArgument(
                          "custom op [%s]: float16 only supports [LINEAR, "
                          "CHW2, HWC8, CHW4, DHWC8(TensorRT 7.2 and after), "
                          "HWC16(TensorRT 8.0 and after)], "
                          "but got tensor format: [%s], ",
                          op_type,
                          tensor_format));
  }
  if (datatype == "int8") {
    std::unordered_set<std::string> supports_formats_tmp = {
        "LINEAR", "CHW32", "CHW4"};
    PADDLE_ENFORCE_GE(
        supports_formats_tmp.count(tensor_format),
        0,
        common::errors::InvalidArgument(
            "custom op [%s]: int8 only supports [LINEAR, CHW32, CHW4], "
            "but got tensor format: [%s], ",
            op_type,
            tensor_format));
  }
  if (datatype == "int32") {
    std::unordered_set<std::string> supports_formats_tmp = {"LINEAR"};
    PADDLE_ENFORCE_GE(supports_formats_tmp.count(tensor_format),
                      0,
                      common::errors::InvalidArgument(
                          "custom op [%s]: int32 only supports [LINEAR], "
                          "but got tensor format: [%s], ",
                          op_type,
                          tensor_format));
  }
}

std::vector<std::pair<std::string, std::string>> parseConfig(
    const std::string& op_type, const std::string& config) {
  std::vector<std::pair<std::string, std::string>> res;
  size_t start = 0;
  size_t seg = config.find("+", start);
  while (seg != std::string::npos) {
    std::string dtype_format = config.substr(start, seg - start);
    size_t split_pos = dtype_format.find(":");
    std::string dtype = dtype_format.substr(0, split_pos);
    std::string format;
    if (split_pos == std::string::npos) {
      format = "LINEAR";
    } else {
      format = dtype_format.substr(split_pos + 1);
    }
    transform(dtype.begin(), dtype.end(), dtype.begin(), ::tolower);
    transform(format.begin(), format.end(), format.begin(), ::toupper);
    validate(op_type, dtype, format);
    res.emplace_back(dtype, format);
    start = seg + 1;
    seg = config.find("+", start);
  }
  std::string dtype_format = config.substr(start);
  size_t split_pos = dtype_format.find(":");
  std::string dtype = dtype_format.substr(0, split_pos);
  std::string format;
  if (split_pos == std::string::npos) {
    format = "LINEAR";
  } else {
    format = dtype_format.substr(split_pos + 1);
  }
  transform(dtype.begin(), dtype.end(), dtype.begin(), ::tolower);
  transform(format.begin(), format.end(), format.begin(), ::toupper);
  validate(op_type, dtype, format);
  res.emplace_back(dtype, format);
  return res;
}

nvinfer1::DataType getTrtDtype(std::string dtype) {
  if (dtype == "float32") {
    return nvinfer1::DataType::kFLOAT;
  } else if (dtype == "float16") {
    return nvinfer1::DataType::kHALF;
  } else if (dtype == "int8") {
    return nvinfer1::DataType::kINT8;
  } else if (dtype == "int32") {
    return nvinfer1::DataType::kINT32;
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("Unsupported data type [%s]", dtype));
  }
}

nvinfer1::TensorFormat getTrtTensorFormat(std::string tensor_format) {
  if (tensor_format == "LINEAR") {
    return nvinfer1::TensorFormat::kLINEAR;
  } else if (tensor_format == "CHW32") {
    return nvinfer1::TensorFormat::kCHW32;
  } else if (tensor_format == "CHW2") {
    return nvinfer1::TensorFormat::kCHW2;
  } else if (tensor_format == "HWC8") {
    return nvinfer1::TensorFormat::kHWC8;
  } else if (tensor_format == "CHW4") {
    return nvinfer1::TensorFormat::kCHW4;
#if IS_TRT_VERSION_GE(7200)
  } else if (tensor_format == "DHWC8") {
    return nvinfer1::TensorFormat::kDHWC8;
#endif
#if IS_TRT_VERSION_GE(8000)
  } else if (tensor_format == "HWC16") {
    return nvinfer1::TensorFormat::kHWC16;
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported tensor format [%s]",
                                               tensor_format));
  }
}

GenerateCustomGenericPluginDataType
ProtoTypeToGenerateCustomGenericPluginDataType(
    framework::proto::VarType_Type proto_type) {
  using framework::proto::VarType_Type;
  switch (proto_type) {
    case VarType_Type::VarType_Type_BOOL:
      return GenerateCustomGenericPluginDataType::PLUGIN_BOOL;
    case VarType_Type::VarType_Type_UINT8:
      return GenerateCustomGenericPluginDataType::PLUGIN_UINT8;
    case VarType_Type::VarType_Type_INT8:
      return GenerateCustomGenericPluginDataType::PLUGIN_INT8;
    case VarType_Type::VarType_Type_INT16:
      return GenerateCustomGenericPluginDataType::PLUGIN_INT16;
    case VarType_Type::VarType_Type_INT32:
      return GenerateCustomGenericPluginDataType::PLUGIN_INT32;
    case VarType_Type::VarType_Type_INT64:
      return GenerateCustomGenericPluginDataType::PLUGIN_INT64;
    case VarType_Type::VarType_Type_FP16:
      return GenerateCustomGenericPluginDataType::PLUGIN_FP16;
    case VarType_Type::VarType_Type_FP32:
      return GenerateCustomGenericPluginDataType::PLUGIN_FP32;
    case VarType_Type::VarType_Type_FP64:
      return GenerateCustomGenericPluginDataType::PLUGIN_FP64;
    case VarType_Type::VarType_Type_SIZE_T:
      return GenerateCustomGenericPluginDataType::PLUGIN_SIZE_T;
    case VarType_Type::VarType_Type_BF16:
      return GenerateCustomGenericPluginDataType::PLUGIN_BF16;
    case VarType_Type::VarType_Type_COMPLEX64:
      return GenerateCustomGenericPluginDataType::PLUGIN_COMPLEX64;
    case VarType_Type::VarType_Type_COMPLEX128:
      return GenerateCustomGenericPluginDataType::PLUGIN_COMPLEX128;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "This data type is currently not supported"));
  }
}

CustomGenericPlugin::CustomGenericPlugin(
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

CustomGenericPlugin::CustomGenericPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const std::vector<GenerateCustomGenericPluginDataType>& inputs_data_type,
    const std::vector<GenerateCustomGenericPluginDataType>& outputs_data_type,
    bool with_fp16) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = framework::OpDesc(proto_op_desc_, nullptr);
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = inputs_data_type;
  outputs_data_type_ = outputs_data_type;
  with_fp16_ = with_fp16;
}

CustomGenericPlugin::CustomGenericPlugin(void const* serial_data,
                                         size_t serial_length) {
  DeserializeValue(&serial_data, &serial_length, &inputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &outputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &with_fp16_);

  std::string op_meta_data((char*)(serial_data), serial_length);  // NOLINT
  op_meta_data_ = std::move(op_meta_data);
  proto_op_desc_.ParseFromString(op_meta_data_);
  op_desc_ = framework::OpDesc(proto_op_desc_, nullptr);
}

int CustomGenericPlugin::getNbOutputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Outputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

int CustomGenericPlugin::getNbInputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Inputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

nvinfer1::IPluginV2DynamicExt* CustomGenericPlugin::clone() const TRT_NOEXCEPT {
  nvinfer1::IPluginV2DynamicExt* plugin = new CustomGenericPlugin(
      proto_op_desc_, inputs_data_type_, outputs_data_type_, with_fp16_);
  plugin->initialize();
  return plugin;
}

void CustomGenericPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
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

bool CustomGenericPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  auto& op_meta_info_map = OpMetaInfoMap::Instance();
  const auto& meta_info_map = op_meta_info_map.GetMap();
  auto& op_info = meta_info_map.at(op_desc_.Type()).front();
  auto& supports_format_config =
      OpMetaInfoHelper::GetTrtSupportsFormatConfig(op_info);
  PADDLE_ENFORCE_NE(supports_format_config.empty(),
                    true,
                    common::errors::InvalidArgument(
                        "The %s op has no tensorrt plugin "
                        "supportsFormatCombination config!"
                        "Please use SetTrtSupportsFormatConfig to set.",
                        op_desc_.Type().c_str()));
  // generate support format combination function by config
  size_t input_num = OpMetaInfoHelper::GetInputs(op_info).size();
  size_t output_num = OpMetaInfoHelper::GetOutputs(op_info).size();
  std::vector<std::vector<std::pair<std::string, std::string>>>
      format_combinations;
  for (auto& config : supports_format_config) {
    auto format_combination = parseConfig(op_desc_.Type(), config);
    PADDLE_ENFORCE_EQ(input_num + output_num,
                      format_combination.size(),
                      common::errors::InvalidArgument(
                          "Expected %d format_combination, but got %d.",
                          input_num + output_num,
                          format_combination.size()));
    format_combinations.emplace_back(format_combination);
  }

  bool is_supported = false;
  for (size_t i = 0; i < input_num + output_num; ++i) {
    if (i < input_num) {
      if (pos == i) {
        for (auto& format_combination : format_combinations) {
          is_supported |=
              (in_out[pos].type == getTrtDtype(format_combination[i].first) &&
               in_out[pos].format ==
                   getTrtTensorFormat(format_combination[i].second));
        }
      }
    } else {
      if (pos == i) {
        for (auto& format_combination : format_combinations) {
          bool is_supported_tmp = true;
          for (size_t j = 0; j < input_num; ++j) {
            is_supported_tmp &=
                (in_out[j].type == getTrtDtype(format_combination[j].first) &&
                 in_out[j].format ==
                     getTrtTensorFormat(format_combination[j].second));
          }
          is_supported_tmp &=
              (in_out[pos].type == getTrtDtype(format_combination[i].first) &&
               in_out[pos].format ==
                   getTrtTensorFormat(format_combination[i].second));
          is_supported |= is_supported_tmp;
        }
      }
    }
  }
  return is_supported;
}

nvinfer1::DataType CustomGenericPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_NE(
      input_types,
      nullptr,
      common::errors::Unavailable("Input type should not be nullptr."));
  return input_types[0];
}

int CustomGenericPlugin::initialize() TRT_NOEXCEPT {
  if (!tensor_inputs_)
    tensor_inputs_ = new std::vector<paddle::Tensor>(getNbInputs());
  if (!tensor_outputs_)
    tensor_outputs_ = new std::vector<paddle::Tensor>(getNbOutputs());
  return 0;
}

nvinfer1::DimsExprs CustomGenericPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_LT(
      output_index,
      getNbOutputs(),
      common::errors::InvalidArgument(
          "Output index (%d) must be less than the number of outputs (%d).",
          output_index,
          getNbOutputs()));
  auto& op_meta_info_map = OpMetaInfoMap::Instance();
  const auto& meta_info_map = op_meta_info_map.GetMap();
  auto& op_info = meta_info_map.at(op_desc_.Type()).front();
  auto& infer_shape_fn = OpMetaInfoHelper::GetTrtInferShapeFn(op_info);
  PADDLE_ENFORCE_NE(infer_shape_fn,
                    nullptr,
                    common::errors::InvalidArgument(
                        "The %s op has no getOutputDimensions function!"
                        "Please use SetTrtInferShapeFn to set.",
                        op_desc_.Type().c_str()));
  std::vector<paddle::any> custom_attrs;
  auto& attrs = op_desc_.GetAttrMap();
  auto& op_attrs_names = OpMetaInfoHelper::GetAttrs(op_info);
  for (auto& op_attrs_name : op_attrs_names) {
    auto attr_name_and_type = paddle::ParseAttrStr(op_attrs_name);
    auto attr_name = attr_name_and_type[0];
    auto attr_type_str = attr_name_and_type[1];
    if (attr_type_str == "bool") {
      custom_attrs.emplace_back(PADDLE_GET_CONST(bool, attrs.at(attr_name)));
    } else if (attr_type_str == "int") {
      custom_attrs.emplace_back(PADDLE_GET_CONST(int, attrs.at(attr_name)));
    } else if (attr_type_str == "float") {
      custom_attrs.emplace_back(PADDLE_GET_CONST(float, attrs.at(attr_name)));
    } else if (attr_type_str == "int64_t") {
      custom_attrs.emplace_back(PADDLE_GET_CONST(int64_t, attrs.at(attr_name)));
    } else if (attr_type_str == "std::string") {
      custom_attrs.emplace_back(
          PADDLE_GET_CONST(std::string, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<int>") {
      custom_attrs.emplace_back(
          PADDLE_GET_CONST(std::vector<int>, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<float>") {
      custom_attrs.emplace_back(
          PADDLE_GET_CONST(std::vector<float>, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<int64_t>") {
      custom_attrs.emplace_back(
          PADDLE_GET_CONST(std::vector<int64_t>, attrs.at(attr_name)));
    } else if (attr_type_str == "std::vector<std::string>") {
      custom_attrs.emplace_back(
          PADDLE_GET_CONST(std::vector<std::string>, attrs.at(attr_name)));
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, Please check whether "
          "the attribute data type and data type string are matched.",
          attr_type_str));
    }
  }
  return infer_shape_fn(
      {output_index, nb_inputs}, inputs, expr_builder, custom_attrs);
}

void CustomGenericPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    getNbInputs(),
                    common::errors::InvalidArgument(
                        "Number of inputs (%d) does not match the "
                        "expected number of inputs (%d).",
                        nb_inputs,
                        getNbInputs()));

  PADDLE_ENFORCE_EQ(nb_outputs,
                    getNbOutputs(),
                    common::errors::InvalidArgument(
                        "Number of outputs (%d) does not match the "
                        "expected number of outputs (%d).",
                        nb_outputs,
                        getNbOutputs()));
}

// Shutdown the layer. This is called when the engine is destroyed
void CustomGenericPlugin::terminate() TRT_NOEXCEPT {
  delete tensor_inputs_;
  delete tensor_outputs_;
}

int CustomGenericPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                 const nvinfer1::PluginTensorDesc* output_desc,
                                 const void* const* inputs,
                                 void* const* outputs,
                                 void* workspace,
                                 cudaStream_t stream) TRT_NOEXCEPT {
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  // TODO(inference): custom generic plugin do not support INT8 precision now.
  auto protoType2PhiType =
      [&](GenerateCustomGenericPluginDataType proto_type,
          nvinfer1::DataType nv_dtype) -> std::pair<phi::DataType, int> {
    if (proto_type == GenerateCustomGenericPluginDataType::PLUGIN_FP16) {
      return {phi::DataType::FLOAT16, sizeof(half)};
    } else if (proto_type == GenerateCustomGenericPluginDataType::PLUGIN_FP32) {
      if (isFp16Supported() && nv_dtype == nvinfer1::DataType::kHALF) {
        return {phi::DataType::FLOAT16, sizeof(half)};
      } else {
        return {phi::DataType::FLOAT32, sizeof(float)};
      }
    } else if (proto_type ==
               GenerateCustomGenericPluginDataType::PLUGIN_INT64) {
      return {phi::DataType::INT64, sizeof(int64_t)};
    } else if (proto_type ==
               GenerateCustomGenericPluginDataType::PLUGIN_INT32) {
      return {phi::DataType::INT32, sizeof(int32_t)};
    } else if (proto_type == GenerateCustomGenericPluginDataType::PLUGIN_BOOL) {
      return {phi::DataType::BOOL, sizeof(bool)};
    } else {
      PADDLE_ENFORCE_EQ(
          false,
          true,
          common::errors::InvalidArgument("Precision is not supported."));
    }
  };

  nvinfer1::DataType data_type = input_desc[0].type;
  PADDLE_ENFORCE_EQ(
      (data_type == nvinfer1::DataType::kFLOAT) ||
          (data_type == nvinfer1::DataType::kHALF),
      true,
      common::errors::InvalidArgument("The data type must be either kFLOAT or "
                                      "kHALF, but received data type %d.",
                                      static_cast<int>(data_type)));

  paddle::CustomOpKernelContext kernel_ctx;
  // input
  for (int i = 0; i < getNbInputs(); i++) {
    if (inputs_data_type_[i] ==
        GenerateCustomGenericPluginDataType::PLUGIN_OPTIONAL) {
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
    kernel_ctx.EmplaceBackInput(std::move((*tensor_inputs_)[i]));
  }

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
    kernel_ctx.EmplaceBackOutput(std::move((*tensor_outputs_)[i]));
  }

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
      PADDLE_THROW(common::errors::Unimplemented(
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

  // sync output tensor data into TensorRT output
  auto* calc_outs = kernel_ctx.AllMutableOutput();
  for (int i = 0; i < getNbOutputs(); i++) {
    auto calc_out =
        std::dynamic_pointer_cast<phi::DenseTensor>(calc_outs->at(i).impl());
    if (reinterpret_cast<void*>(calc_out->data()) !=
        reinterpret_cast<void*>(outputs[i])) {
      LOG_FIRST_N(WARNING, 1)
          << "You created new Tensor(s) in custom operator(s) used as "
             "output(s), "
             "we will do cudaMemcpy to synchronize the output(s) "
             "address needed by TensorRT plugin. "
             "Inplace operation is highly recommended for better performance.";
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
      phi::DenseTensor dense_output =
          std::move(phi::DenseTensor(output_alloc, output_meta));
      cudaMemcpy(static_cast<void*>(dense_output.data()),
                 static_cast<void*>(calc_out->data()),
                 output_numel * data_type_and_size.second,
                 cudaMemcpyDeviceToDevice);
    }
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
