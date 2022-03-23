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

#include "paddle/infrt/kernel/phi/dense_tensor_kernels.h"
#include "paddle/infrt/common/string.h"
#include "paddle/infrt/dialect/phi/data_type.h"
#include "paddle/infrt/kernel/phi/context_kernels.h"
#include "paddle/infrt/paddle/model_parser.h"
#include "paddle/infrt/paddle/scope.h"
#include "paddle/infrt/tensor/tensor_map.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/place.h"

#ifdef INFRT_WITH_GPU
#include <cuda_runtime.h>
#endif

namespace paddle {
namespace platform {
using DeviceContext = ::phi::DeviceContext;
}  // namespace platform
namespace framework {
using LoDTensor = ::phi::DenseTensor;
void DeserializeFromStream(std::istream& is,
                           LoDTensor* tensor,
                           const platform::DeviceContext& dev_ctx);
}
}  // namespace paddle

namespace infrt {
namespace kernel {
namespace phi {

::phi::DenseTensor CreateDenseTensor(
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision) {
  return ::phi::DenseTensor(
      const_cast<::phi::Allocator*>(&context.GetAllocator()),
      ::phi::DenseTensorMeta(ConvertPrecisionToPhi(precision.get()),
                             ::phi::make_ddim(dims.get()),
                             ConvertLayoutToPhi(layout.get()),
                             {}));
}

::phi::DenseTensor CreateGPUDenseTensor(
    const ::phi::GPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision) {
  return ::phi::DenseTensor(
      const_cast<::phi::Allocator*>(&context.GetAllocator()),
      ::phi::DenseTensorMeta(ConvertPrecisionToPhi(precision.get()),
                             ::phi::make_ddim(dims.get()),
                             ConvertLayoutToPhi(layout.get()),
                             {}));
}

void FillDenseTensorF32(::phi::DenseTensor* dense_tensor,
                        host_context::Attribute<std::vector<float>> value) {
  auto place = dense_tensor->place();
  float* a_data = dense_tensor->mutable_data<float>(place);
  if (place.GetType() == ::phi::AllocationType::CPU) {
    for (int64_t i = 0; i < dense_tensor->numel(); ++i) {
      a_data[i] = (value.get())[i];
    }
  } else if (place.GetType() == ::phi::AllocationType::GPU) {
#ifdef INFRT_WITH_GPU
    // TODO(wilber): how to set the stream parameter to copy with stream.
    cudaMemcpy(a_data,
               value.get().data(),
               sizeof(float) * value.get().size(),
               cudaMemcpyHostToDevice);
#endif
  } else {
    llvm_unreachable("temporarily not support other target.");
  }
}

void PrintDenseTensor(::phi::DenseTensor* dense_tensor) {
#ifndef INFRT_WITH_GPU
#define PRINT_META_DATA(PHI_DATATYPE, DTYPE)                \
  case ::phi::DataType::PHI_DATATYPE: {                     \
    auto place = dense_tensor->place();                     \
    if (place.GetType() == ::phi::AllocationType::CPU) {    \
      DTYPE* data = dense_tensor->data<DTYPE>();            \
      if (dense_tensor->numel() == 0) break;                \
      std::cout << data[0];                                 \
      for (int64_t i = 1; i < dense_tensor->numel(); i++) { \
        std::cout << "," << data[i];                        \
      }                                                     \
    }                                                       \
    break;                                                  \
  }
#else
#define PRINT_META_DATA(PHI_DATATYPE, DTYPE)                     \
  case ::phi::DataType::PHI_DATATYPE: {                          \
    auto place = dense_tensor->place();                          \
    DTYPE* data = dense_tensor->data<DTYPE>();                   \
    if (dense_tensor->numel() == 0) break;                       \
    if (place.GetType() == ::phi::AllocationType::CPU) {         \
      std::cout << data[0];                                      \
      for (int64_t i = 1; i < dense_tensor->numel(); i++) {      \
        std::cout << "," << data[i];                             \
      }                                                          \
    } else if (place.GetType() == ::phi::AllocationType::GPU) {  \
      std::vector<DTYPE> host_data(dense_tensor->numel(), 0);    \
      cudaMemcpy(host_data.data(),                               \
                 data,                                           \
                 sizeof(DTYPE) * dense_tensor->numel(),          \
                 cudaMemcpyDeviceToHost);                        \
      std::cout << host_data[0];                                 \
      for (int64_t i = 1; i < dense_tensor->numel(); i++) {      \
        std::cout << "," << host_data[i];                        \
      }                                                          \
    } else {                                                     \
      llvm_unreachable("temporarily not support other target."); \
    }                                                            \
    break;                                                       \
  }
#endif

  ::phi::DDim dims = dense_tensor->dims();
  std::cout << "dense_tensor: shape=shape" << dims.to_str() << ","
            << " value=[";
  switch (dense_tensor->dtype()) {
    PRINT_META_DATA(FLOAT32, float);
    PRINT_META_DATA(INT32, int32_t);
    default:
      std::cout << "Error! Unsupported data type!\n";
  }
  std::cout << "]\n";
#undef PRINT_META_DATA
}

::infrt::phi::DenseTensorMap LoadParameters(const std::string& file_path) {
  std::cout << "loading params from: " << file_path << std::endl;
  ::infrt::phi::DenseTensorMap map;

  const std::string model_path = file_path + "/__model__";
  auto pb_proto_prog = paddle::LoadProgram(model_path);
  auto main_block = pb_proto_prog->blocks(0);

  for (auto& var : main_block.vars()) {
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable())
      continue;
    std::string param_path = file_path + "/" + var.name();
    std::ifstream param_file(param_path, std::ios::binary);
    switch (var.type().type()) {
      case ::paddle::framework::proto::VarType_Type_LOD_TENSOR: {
        std::unique_ptr<::phi::DenseTensor> tensor{
            std::make_unique<::phi::DenseTensor>()};
        ::phi::CPUContext ctx;
        ::paddle::framework::DeserializeFromStream(
            param_file, tensor.get(), ctx);
        map.SetDenseTensor(var.name(), std::move(tensor));
      } break;
      default: {
        LOG(WARNING) << "Var `" << var.name() << "` type `"
                     << static_cast<int>(var.type().type())
                     << "` has not been supported now.";
      }
    }
  }
  return map;
}

::infrt::phi::DenseTensorMap LoadParams(
    host_context::Attribute<std::string> path) {
  return LoadParameters(path.get());
}

::infrt::phi::DenseTensorMap LoadCombinedParameters(
    const std::string& model_path, const std::string& params_path) {
  std::cout << "loading params from: " << model_path << std::endl;
  ::infrt::phi::DenseTensorMap map;

  auto pb_proto_prog = paddle::LoadProgram(model_path);
  auto main_block = pb_proto_prog->blocks(0);

  std::ifstream param_file(params_path, std::ios::binary);

  std::set<std::string> tmp;
  for (auto& var : main_block.vars()) {
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable()) {
      continue;
    }
    if (var.type().type() ==
        ::paddle::framework::proto::VarType_Type_LOD_TENSOR) {
      tmp.emplace(var.name());
    } else {
      llvm_unreachable("the tensor type is illegal.");
    }
  }

  for (auto& var : tmp) {
    std::unique_ptr<::phi::DenseTensor> tensor{
        std::make_unique<::phi::DenseTensor>()};
    ::phi::CPUContext ctx;
    ::paddle::framework::DeserializeFromStream(param_file, tensor.get(), ctx);
    map.SetDenseTensor(var, std::move(tensor));
  }

  return map;
}

::infrt::phi::DenseTensorMap LoadCombinedParams(
    host_context::Attribute<std::string> model_path,
    host_context::Attribute<std::string> params_path) {
  return LoadCombinedParameters(model_path.get(), params_path.get());
}

::phi::DenseTensor TensorMapGetTensor(
    const ::infrt::phi::DenseTensorMap& map,
    host_context::Attribute<std::string> name) {
  auto* tensor = map.GetDenseTensor(name.get());
  CHECK(tensor);
  return *tensor;
}

int32_t TensorMapGetSize(const ::infrt::phi::DenseTensorMap& map) {
  return map.size();
}

::phi::DenseTensor ConvertHostTensor(::infrt::tensor::DenseHostTensor&& tensor,
                                     ::phi::Allocator* host_allocator) {
  std::vector<int64_t> dims;
  dims.reserve(tensor.shape().GetRank());
  for (const auto dim : tensor.shape()) {
    dims.emplace_back(dim);
  }
  CHECK(tensor.dtype().kind() == ::infrt::DType::Kind::F32);
  ::phi::DenseTensorMeta meta(::phi::DataType::FLOAT32, ::phi::make_ddim(dims));
  ::phi::DenseTensor phi_tensor(host_allocator, std::move(meta));
  void* src = tensor.raw_data();
  void* dst = phi_tensor.data();
  size_t bytes = sizeof(float) * tensor.shape().GetNumElements();
  std::memcpy(dst, src, bytes);
  return phi_tensor;
}

::infrt::phi::DenseTensorMap ConvertHostTensorMap(
    ::infrt::tensor::TensorMap&& map, ::phi::Allocator* host_allocator) {
  ::infrt::phi::DenseTensorMap phi_tensors;
  for (auto& pair : map) {
    phi_tensors.SetDenseTensor(
        pair.first,
        std::make_unique<::phi::DenseTensor>(
            ConvertHostTensor(std::move(*pair.second), host_allocator)));
  }
  return phi_tensors;
}

}  // namespace phi
}  // namespace kernel
}  // namespace infrt
