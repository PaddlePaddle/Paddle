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

using paddle::lite_api::TargetType;
using paddle::lite_api::PrecisionType;
using paddle::lite_api::DataLayoutType;
using platform::CPUPlace;
using platform::CUDAPlace;

namespace {

const platform::Place& GetNativePlace(TargetType) {
  switch (TargetType) {
    case TargetType::kHost:
      return CPUPlace();
    case TargetType::kCUDA:
      return CUDAPlace();
    default:
      LOG(FATAL) << "Error target type.";
      return platform::Place();
  }
}

const proto::VarType::Type& GetNativePrecisionType(PrecisionType) {
  switch (PrecisionType) {
    case PrecisionType::kFloat:
      return proto::VarType_Type_FP32;
    case PrecisionType::kInt8:
      return proto::VarType_Type_INT8;
    default:
      LOG(FATAL) << "Error precision type.";
      return static_cast<proto::VarType::Type>(-1);
  }
}

const framework::DataLayout& GetNativeLayoutType(DataLayoutType) {
  switch (DataLayoutType) {
    case DataLayoutType::kNCHW:
      return framework::DataLayout::kNCHW;
    default:
      LOG(FATAL) << "Error layout type.";
      return static_cast<framework::DataLayout>(-1);
  }
}

void MemoryCopy(const platform::Place& dst_place, void* dst_data,
    const platform::Place& src_place, const void* src_data, const size_t size) {
  if (platform::is_cpu_place(dst_place) && platform::is_cpu_place(src_place)) {
    Copy(dst_place, dst_data, src_place, src_data, SizeOfType(src.numel()));
  } else {
#ifdef PADDLE_WITH_CUDA
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &ctx = *pool.Get(place);
    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    Copy(dst_place, dst_data, src_place, src_data, SizeOfType(src.numel()), stream);
#else
    LOG(FATAL) << "You must define PADDLE_WITH_CUDA for using CUDAPlace.";
#endif
  }
}

} // namespace

template<paddle::lite::Tensor, framework::LoDTensor>
void TensorCopy(paddle::lite::Tensor* dst, const framework::LoDTensor& src) {
  const TargetType dst_target = dst->target();
  const platform::Place& src_place = src.place();
  const platform::Place& dst_place = GetNativePlace(dst_target);
  PADDLE_ENFORCE_EQ(src.type(), GetNativePrecisionType(dst->precision()));
  std::vector<int64_t> dims = framework::vectorize(src.dims());
  dst->Resize(dims);
  const void* src_data = src.data<void>();
  void* dst_data = dst->mutable_data<void>(dst_target);
  MemoryCopy(dst_place, dst_data, src_place, src_data, src.numel());
}

template<framework::LoDTensor, paddle::lite::Tensor>
void TensorCopy(framework::LoDTensor* dst, const paddle::lite::Tensor& src) {
  const platform::Place& src_place = GetNativePlace(src.target());
  const platform::Place& dst_place = dst->place();
  PADDLE_ENFORCE_EQ(dst->type(), GetNativePrecisionType(src.precision()));
  std::vector<int64_t> dims = src.dims().Vectorize();
  dst->Resize(dims);
  const void* src_data = src.raw_data();
  void* dst_data = dst->mutable_data<void>(dst_place, dst->type());
  MemoryCopy(dst_place, dst_data, src_place, src_data, src.numel());
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle