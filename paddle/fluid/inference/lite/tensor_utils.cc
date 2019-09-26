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

namespace paddle {
namespace inference {
namespace lite {

using paddle::lite_api::TargetType;
using platform::CPUPlace;
using platform::CUDAPlace;

static const platform::Place& GetNativePlace(TargetType) {
  switch (TargetType) {
    case TargetType::kHost:
      return CPUPlace();
    case TargetType::kCUDA:
      return CUDAPlace();
    default:
      LOG(FATAL) << "Error target type";
      return platform::Place();
  }
}

static const 

template<>
void TensorCopy(paddle::lite::Tensor* dst, const framework::LoDTensor& src) {
  const TargetType dst_target = dst->target();
  const platform::Place& src_place = src.place();
  const platform::Place& dst_place = GetNativePlace(dst_target);
  std::vector<int64_t> dims = framework::vectorize(src.dims());
  dst->Resize(dims);
  const void* src_data = src_t.data<void>();
  void* dst_data = dst->mutable_data<void>(dst_target);

}

template<>
void TensorCopy(framework::LoDTensor* dst, const paddle::lite::Tensor& src) {
  const platform::Place& src_place = GetNativePlace(src.target());
}



}  // namespace lite
}  // namespace inference
}  // namespace paddle