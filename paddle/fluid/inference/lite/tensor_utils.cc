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

#include <map>
#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/inference/lite/tensor_utils.h"
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace inference {
namespace lite {

namespace {

using paddle::lite_api::TargetType;
using paddle::lite_api::PrecisionType;
using paddle::lite_api::DataLayoutType;

platform::Place GetNativePlace(const TargetType& type) {
  switch (type) {
    case TargetType::kHost:
      return platform::CPUPlace();
    case TargetType::kCUDA:
      return platform::CUDAPlace();
    default:
      LOG(FATAL) << "Error target type.";
      return platform::Place();
  }
}

framework::proto::VarType::Type GetNativePrecisionType(const PrecisionType& type) {
  switch (type) {
    case PrecisionType::kFloat:
      return framework::proto::VarType_Type_FP32;
    case PrecisionType::kInt8:
      return framework::proto::VarType_Type_INT8;
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

void MemoryCopy(const platform::Place& dst_place, void* dst_data,
    const platform::Place& src_place, const void* src_data, const size_t size) {
  const platform::CPUPlace cpu_place;
  const platform::CUDAPlace gpu_place;
  if (platform::is_cpu_place(dst_place) && platform::is_cpu_place(src_place)) {
    memory::Copy(cpu_place, dst_data, cpu_place, src_data, size);
  } else {
#ifdef PADDLE_WITH_CUDA
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &ctx = *pool.Get(platform::CUDAPlace());
    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    if (platform::is_cpu_place(dst_place) && platform::is_gpu_place(src_place)) {
      memory::Copy(cpu_place, dst_data, gpu_place, src_data, size, stream);
    } else if (platform::is_gpu_place(dst_place) && platform::is_cpu_place(src_place)) {
      memory::Copy(gpu_place, dst_data, cpu_place, src_data, size, stream);
    } else if (platform::is_gpu_place(dst_place) && platform::is_gpu_place(src_place)) {
      memory::Copy(gpu_place, dst_data, gpu_place, src_data, size, stream);
    }
#else
    LOG(FATAL) << "You must define PADDLE_WITH_CUDA for using CUDAPlace.";
#endif
  }
}

} // namespace

template<>
void TensorCopy(paddle::lite::Tensor* dst, const framework::LoDTensor& src) {
  const platform::Place& src_place = src.place();
  const platform::Place& dst_place = GetNativePlace(dst->target());
  PADDLE_ENFORCE_EQ(src.type(), GetNativePrecisionType(dst->precision()));
  const size_t size = static_cast<size_t>(src.numel());
  dst->Resize(framework::vectorize(src.dims()));
  const void* src_data = src.data<void>();
  void* dst_data = dst->mutable_data(size);
  MemoryCopy(dst_place, dst_data, src_place, src_data, size);
}

template<>
void TensorCopy(framework::LoDTensor* dst, const paddle::lite::Tensor& src) {
  const platform::Place& src_place = GetNativePlace(src.target());
  const platform::Place& dst_place = dst->place();
  PADDLE_ENFORCE_EQ(dst->type(), GetNativePrecisionType(src.precision()));
  dst->Resize(paddle::framework::make_ddim(src.dims().Vectorize()));
  const size_t size = static_cast<size_t>(src.numel());
  const void* src_data = src.raw_data();
  void* dst_data = dst->mutable_data(dst_place, dst->type());
  MemoryCopy(dst_place, dst_data, src_place, src_data, size);
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle
