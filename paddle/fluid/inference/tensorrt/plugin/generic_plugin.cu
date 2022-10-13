// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/generic_plugin.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_registry.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

void BuildPhiKernelContextAttr(const framework::OpDesc& op_desc,
                               phi::KernelContext* kernel_context,
                               const phi::KernelSignature& signature,
                               const phi::Kernel& phi_kernel) {
  const phi::KernelArgsDef& args_def = phi_kernel.args_def();
  const auto& attr_names = signature.attr_names;
  const auto& attr_defs = args_def.attribute_defs();

  PADDLE_ENFORCE_EQ(
      attr_names.size(),
      attr_defs.size(),
      platform::errors::InvalidArgument(
          "The attr_names.size() should be equal to attr_defs.size()."));

  framework::AttrReader attr_reader(op_desc.GetAttrMap());

  for (size_t k = 0; k < attr_names.size(); ++k) {
    auto attr_name = attr_names[k];
    auto* attr_ptr = attr_reader.GetAttr(attr_name);
    if (attr_ptr) {
      switch (attr_defs[k].type_index) {
        case phi::AttributeType::SCALAR: {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::FLOAT:
              return kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(float, attr)));
              break;
            case framework::proto::AttrType::INT:
              return kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(int, attr)));
              break;
            case framework::proto::AttrType::STRING:
              return kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(std::string, attr)));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to Scalar when "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        } break;

        case phi::AttributeType::INT_ARRAY: {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::INTS:
              kernel_context->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int32_t>, attr))));
              break;
            case framework::proto::AttrType::LONGS:
              kernel_context->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int64_t>, attr))));
              break;
            case framework::proto::AttrType::INT:
              kernel_context->EmplaceBackAttr(
                  phi::IntArray({PADDLE_GET_CONST(int, attr)}));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to IntArray when "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        } break;

        case phi::AttributeType::SCALARS: {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::INTS: {
              const auto& vec = PADDLE_GET_CONST(std::vector<int32_t>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::LONGS: {
              const auto& vec = PADDLE_GET_CONST(std::vector<int64_t>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::FLOATS: {
              const auto& vec = PADDLE_GET_CONST(std::vector<float>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::FLOAT64S: {
              const auto& vec = PADDLE_GET_CONST(std::vector<double>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to vector<Scalar> when "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        } break;

        default: {
          auto& attr = *attr_ptr;
          switch (attr_defs[k].type_index) {
            case phi::AttributeType::FLOAT32:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(float, attr));
              break;
            case phi::AttributeType::INT32:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(int, attr));
              break;
            case phi::AttributeType::BOOL:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(bool, attr));
              break;
            case phi::AttributeType::INT64:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(int64_t, attr));
              break;
            case phi::AttributeType::INT32S:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<int>, attr));
              break;
            case phi::AttributeType::DATA_TYPE: {
              auto data_type = paddle::framework::TransToPhiDataType(
                  static_cast<framework::proto::VarType::Type>(
                      PADDLE_GET_CONST(int, attr)));
              kernel_context->EmplaceBackAttr(data_type);
            } break;
            case phi::AttributeType::STRING:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::string, attr));
              break;
            case phi::AttributeType::INT64S:
              switch (AttrTypeID(attr)) {
                case framework::proto::AttrType::LONGS:
                  kernel_context->EmplaceBackAttr(
                      PADDLE_GET_CONST(std::vector<int64_t>, attr));
                  break;
                case framework::proto::AttrType::INTS: {
                  const auto& vector_int_attr =
                      PADDLE_GET_CONST(std::vector<int>, attr);
                  const std::vector<int64_t> vector_int64_attr(
                      vector_int_attr.begin(), vector_int_attr.end());
                  kernel_context->EmplaceBackAttr(vector_int64_attr);
                } break;
                default:
                  PADDLE_THROW(platform::errors::Unimplemented(
                      "Unsupported cast op attribute `%s` to vector<int64_t> "
                      "when ProtoAttr2PhiAttr.",
                      attr_name));
              }
              break;
            case phi::AttributeType::FLOAT32S:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<float>, attr));
              break;
            case phi::AttributeType::STRINGS:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<std::string>, attr));
              break;
            case phi::AttributeType::BOOLS:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<bool>, attr));
              break;
            case phi::AttributeType::FLOAT64S:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<double>, attr));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` when construct "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        }
      }
    }
  }
  CHECK_EQ(attr_names.size(), kernel_context->AttrsSize());
}

GenericPlugin::GenericPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const InputOutPutVarInfo& in_out_info) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = in_out_info.inputs_data_type;
  outputs_data_type_ = in_out_info.outputs_data_type;
}

GenericPlugin::GenericPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const std::vector<int>& inputs_data_type,
    const std::vector<int>& outputs_data_type) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = inputs_data_type;
  outputs_data_type_ = outputs_data_type;
}

GenericPlugin::GenericPlugin(void const* serial_data, size_t serial_length) {
  DeserializeValue(&serial_data, &serial_length, &inputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &outputs_data_type_);
  std::string op_meta_data((char*)(serial_data), serial_length);  // NOLINT
  op_meta_data_ = std::move(op_meta_data);
  proto_op_desc_.ParseFromString(op_meta_data_);
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
}

int GenericPlugin::getNbOutputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Outputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

int GenericPlugin::getNbInputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Inputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

nvinfer1::IPluginV2DynamicExt* GenericPlugin::clone() const TRT_NOEXCEPT {
  nvinfer1::IPluginV2DynamicExt* plugin =
      new GenericPlugin(proto_op_desc_, inputs_data_type_, outputs_data_type_);
  plugin->initialize();
  return plugin;
}

void GenericPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  // inputs_data_type_
  SerializeValue(&buffer, inputs_data_type_);
  // outputs_data_type_
  SerializeValue(&buffer, outputs_data_type_);
  // serialize op_meta_data_
  std::memcpy(buffer, op_meta_data_.c_str(), op_meta_data_.size());
  reinterpret_cast<char*&>(buffer) += op_meta_data_.size();
}

bool GenericPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  if (op_desc_.Type() == "gather_nd" || op_desc_.Type() == "yolo_box") {
    if (pos == 0) return in_out[pos].type == nvinfer1::DataType::kFLOAT;
    if (pos == 1) return in_out[pos].type == nvinfer1::DataType::kINT32;
  } else {
    return in_out[pos].type == nvinfer1::DataType::kFLOAT;
  }
}

nvinfer1::DataType GenericPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

int GenericPlugin::initialize() TRT_NOEXCEPT {
  std::string op_type = op_desc_.Type();

  phi::KernelSignature phi_kernel_signature;
  if (phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_type)) {
    const phi::ArgumentMappingFn* argument_mapping_func =
        phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_type);
    PluginArgumentMappingContext argument_mapping_context(&op_desc_);
    phi_kernel_signature = (*argument_mapping_func)(argument_mapping_context);
  } else {
    phi_kernel_signature =
        phi::DefaultKernelSignatureMap::Instance().Get(op_type);
  }

  phi::KernelKey phi_kernel_key(
      phi::Backend::GPU, phi::DataLayout::ANY, phi::DataType::FLOAT32);

  PADDLE_ENFORCE_EQ(
      phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type),
      true,
      platform::errors::Fatal("%s has no compatible phi kernel!",
                              op_type.c_str()));

  const phi::Kernel& phi_kernel = phi::KernelFactory::Instance().SelectKernel(
      phi_kernel_signature.name, phi_kernel_key);
  phi_kernel_ = &phi_kernel;

  PADDLE_ENFORCE_EQ(phi_kernel_->IsValid(),
                    true,
                    platform::errors::Fatal("%s phi kernel is invalid!.",
                                            phi_kernel_signature.name));

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));

  if (!phi_kernel_context_) {
    phi_kernel_context_ = new phi::KernelContext(dev_ctx);
    BuildPhiKernelContextAttr(
        op_desc_, phi_kernel_context_, phi_kernel_signature, phi_kernel);
  }
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
  auto& dynamic_infermeta_factory = tensorrt::DynamicMetaFnFactory::Instance();
  PADDLE_ENFORCE_EQ(dynamic_infermeta_factory.Contains(op_desc_.Type()),
                    true,
                    platform::errors::InvalidArgument(
                        "The %s op has no dynamic plugin infershape function!",
                        op_desc_.Type().c_str()));

  auto* infershape_func = dynamic_infermeta_factory.Get(op_desc_.Type());
  return infershape_func(
      output_index, inputs, nb_inputs, expr_builder, op_desc_);
}

void GenericPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) TRT_NOEXCEPT {
  CHECK(phi_kernel_context_);
  CHECK(phi_kernel_);
  CHECK(nb_inputs == getNbInputs());
  CHECK(nb_outputs == getNbOutputs());
}

// Shutdown the layer. This is called when the engine is destroyed
void GenericPlugin::terminate() TRT_NOEXCEPT {
  delete phi_kernel_context_;
  delete dense_tensor_inputs_;
  delete dense_tensor_outputs_;
}

int GenericPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                           const nvinfer1::PluginTensorDesc* output_desc,
                           const void* const* inputs,
                           void* const* outputs,
                           void* workspace,
                           cudaStream_t stream) TRT_NOEXCEPT {
  platform::CUDAPlace place(platform::GetCurrentDeviceId());

  // [TODO]now generic plugin do not support FP16 and INT8 precision
  auto protoType2PhiType = [](int proto_type) -> std::pair<phi::DataType, int> {
    if (proto_type ==
        static_cast<int>(framework::proto::VarType_Type::VarType_Type_FP32))
      return {phi::DataType::FLOAT32, sizeof(float)};
    else if (proto_type ==
                 static_cast<int>(
                     framework::proto::VarType_Type::VarType_Type_INT64) ||
             proto_type ==
                 static_cast<int>(
                     framework::proto::VarType_Type::VarType_Type_INT32))
      return {phi::DataType::INT32, sizeof(int32_t)};
    else if (proto_type ==
             static_cast<int>(
                 framework::proto::VarType_Type::VarType_Type_BOOL))
      return {phi::DataType::BOOL, sizeof(bool)};
    else
      CHECK(false) << "precision is not supported";
  };

  // input
  phi_kernel_context_->ClearInputOutput();

  for (int i = 0; i < getNbInputs(); i++) {
    auto const& input_dims = input_desc[i].dims;

    std::vector<int> input_shape;
    for (int j = 0; j < input_dims.nbDims; j++)
      input_shape.push_back(input_dims.d[j]);

    int input_numel = 1;
    for (int k = 0; k < input_shape.size(); k++) input_numel *= input_shape[k];

    auto data_type_and_size = protoType2PhiType(inputs_data_type_[i]);
    phi::DenseTensorMeta input_meta(data_type_and_size.first,
                                    phi::make_ddim(input_shape));
    std::shared_ptr<phi::Allocation> input_alloc(
        new phi::Allocation((void*)(inputs[i]),  // NOLINT
                            input_numel * data_type_and_size.second,
                            place));
    (*dense_tensor_inputs_)[i] =
        std::move(phi::DenseTensor(input_alloc, input_meta));
    phi_kernel_context_->EmplaceBackInput(&((*dense_tensor_inputs_)[i]));
  }

  // output
  for (int i = 0; i < getNbOutputs(); i++) {
    auto const& output_dims = output_desc[i].dims;

    std::vector<int> output_shape;
    for (int j = 0; j < output_dims.nbDims; j++)
      output_shape.push_back(output_dims.d[j]);

    int output_numel = 1;
    for (int k = 0; k < output_shape.size(); k++)
      output_numel *= output_shape[k];

    auto data_type_and_size = protoType2PhiType(inputs_data_type_[i]);
    phi::DenseTensorMeta output_meta(data_type_and_size.first,
                                     phi::make_ddim(output_shape));
    std::shared_ptr<phi::Allocation> output_alloc(
        new phi::Allocation(reinterpret_cast<void*>(outputs[i]),
                            output_numel * data_type_and_size.second,
                            place));
    phi::DenseTensor output_densetonsor(output_alloc, output_meta);
    (*dense_tensor_outputs_)[i] =
        std::move(phi::DenseTensor(output_alloc, output_meta));
    phi_kernel_context_->EmplaceBackOutput(&((*dense_tensor_outputs_)[i]));
  }

  CHECK_EQ(phi_kernel_context_->InputsSize(), getNbInputs());
  CHECK_EQ(phi_kernel_context_->OutputsSize(), getNbOutputs());

  (*phi_kernel_)(phi_kernel_context_);

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
