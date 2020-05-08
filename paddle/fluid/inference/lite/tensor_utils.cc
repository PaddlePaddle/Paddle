// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/lite/tensor_utils.h"
#include <map>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/inference/lite/engine.h"

namespace paddle {
namespace inference {
namespace lite {
namespace utils {

using paddle::lite_api::TargetType;
using paddle::lite_api::PrecisionType;
using paddle::lite_api::DataLayoutType;

template <typename DstLoD, typename SrcLoD>
void SetLoD(DstLoD* dst, const SrcLoD& src) {
  dst->reserve(src.size());
  dst->clear();
  for (auto&& v : src) {
    dst->emplace_back(v);
  }
}
template void SetLoD<paddle::lite::LoD, framework::LoD>(
    paddle::lite::LoD* dst, const framework::LoD& src);
template void SetLoD<framework::LoD, paddle::lite::LoD>(
    framework::LoD* dst, const paddle::lite::LoD& src);

platform::Place GetNativePlace(const TargetType& type, int id = 0) {
  switch (type) {
    case TargetType::kHost:
    case TargetType::kX86:
      return platform::CPUPlace();
    case TargetType::kCUDA:
      return platform::CUDAPlace(id);
    default:
      LOG(FATAL) << "Error target type.";
      return platform::Place();
  }
}

TargetType GetLiteTargetType(const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return TargetType::kHost;
  }
  return TargetType::kCUDA;
}

PrecisionType GetLitePrecisionType(framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType_Type_FP32:
      return PrecisionType::kFloat;
    case framework::proto::VarType_Type_INT8:
      return PrecisionType::kInt8;
    case framework::proto::VarType_Type_INT32:
      return PrecisionType::kInt32;
    case framework::proto::VarType_Type_INT64:
      return PrecisionType::kInt64;
    default:
      LOG(FATAL) << "Error precision type.";
      return PrecisionType::kUnk;
  }
}

framework::proto::VarType::Type GetNativePrecisionType(
    const PrecisionType& type) {
  switch (type) {
    case PrecisionType::kFloat:
      return framework::proto::VarType_Type_FP32;
    case PrecisionType::kInt8:
      return framework::proto::VarType_Type_INT8;
    case PrecisionType::kInt32:
      return framework::proto::VarType_Type_INT32;
    case PrecisionType::kInt64:
      return framework::proto::VarType_Type_INT64;
    default:
      LOG(FATAL) << "Error precision type.";
      return static_cast<framework::proto::VarType::Type>(-1);
  }
}

framework::DataLayout GetNativeLayoutType(const DataLayoutType& type) {
  switch (type) {
    case DataLayoutType::kNCHW:
      return framework::DataLayout::kNCHW;
    default:
      LOG(FATAL) << "Error layout type.";
      return static_cast<framework::DataLayout>(-1);
  }
}

void MemoryCopyAsync(const platform::Place& dst_place, void* dst_data,
                     const platform::Place& src_place, const void* src_data,
                     const size_t size, const platform::DeviceContext& ctx) {
  const platform::CPUPlace cpu_place;
  if (platform::is_cpu_place(dst_place) && platform::is_cpu_place(src_place)) {
    memory::Copy(cpu_place, dst_data, cpu_place, src_data, size);
  } else {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_cpu_place(dst_place) &&
        platform::is_gpu_place(src_place)) {
      LOG(FATAL) << "lite::MemoryCopy GPU->CPU is not yet implemented.";
    } else if (platform::is_gpu_place(dst_place) &&
               platform::is_cpu_place(src_place)) {
      LOG(FATAL) << "lite::MemoryCopy CPU->GPU is not yet implemented.";
    } else if (platform::is_gpu_place(dst_place) &&
               platform::is_gpu_place(src_place)) {
      auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, src_place);
      memory::Copy(
          gpu_place, dst_data, gpu_place, src_data, size,
          static_cast<const platform::CUDADeviceContext&>(ctx).stream());
    }
#else
    LOG(FATAL) << "You must define PADDLE_WITH_CUDA for using CUDAPlace.";
#endif
  }
}

void InitDstTensor(paddle::lite::Tensor* dst, const framework::LoDTensor& src) {
  // Currently, Lite needs to explicitly specify the target type of
  // the input tensor.
  constexpr int empty_size = 0;
  dst->mutable_data(GetLiteTargetType(src.place()), empty_size);
  dst->set_precision(GetLitePrecisionType(src.type()));
  SetLoD(dst->mutable_lod(), src.lod());
}

void InitDstTensor(framework::LoDTensor* dst, const paddle::lite::Tensor& src) {
  constexpr framework::proto::VarType::Type dtype =
      framework::proto::VarType_Type_FP32;
  dst->mutable_data(inference::lite::utils::GetNativePlace(src.target()),
                    dtype);
  SetLoD(dst->mutable_lod(), src.lod());
}

template <>
void TensorCopyAsync(paddle::lite::Tensor* dst, const framework::LoDTensor& src,
                     const platform::DeviceContext& ctx) {
  InitDstTensor(dst, src);
  const platform::Place& src_place = src.place();
  const platform::Place& dst_place = GetNativePlace(dst->target());
  const size_t bytes =
      static_cast<size_t>(src.numel()) * framework::SizeOfType(src.type());
  dst->Resize(framework::vectorize(src.dims()));
  const void* src_data = src.data<void>();
  void* dst_data = dst->mutable_data(bytes);
  VLOG(3) << "[CopyAsync fluid -> lite] Bytes = " << bytes << ", src = " << &src
          << ", dst = " << dst << ", src_type = " << src.type();
  MemoryCopyAsync(dst_place, dst_data, src_place, src_data, bytes, ctx);
  VLOG(3) << "[Lite memory size] Bytes = " << dst->memory_size();
}

template <>
void TensorCopyAsync(framework::LoDTensor* dst, const paddle::lite::Tensor& src,
                     const platform::DeviceContext& ctx) {
  dst->Resize(paddle::framework::make_ddim(src.dims().Vectorize()));
  InitDstTensor(dst, src);
  const platform::Place& src_place = GetNativePlace(src.target());
  const platform::Place& dst_place = dst->place();
  const size_t bytes =
      static_cast<size_t>(src.numel()) * framework::SizeOfType(dst->type());
  const void* src_data = src.raw_data();
  // When Lite is ready, the source type needs to be modified here.
  void* dst_data = dst->mutable_data(dst_place, dst->type());
  VLOG(3) << "[CopyAsync lite -> fluid] Bytes = " << bytes << ", src = " << &src
          << ", dst = " << dst << ", src_type = " << dst->type();
  MemoryCopyAsync(dst_place, dst_data, src_place, src_data, bytes, ctx);
  VLOG(3) << "[Lite memory size] Bytes = " << src.memory_size();
}

}  // namespace utils
}  // namespace lite
}  // namespace inference
}  // namespace paddle
