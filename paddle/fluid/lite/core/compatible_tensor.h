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

#pragma once

#include <vector>
#include "paddle/fluid/lite/core/target_wrapper.h"
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#include "paddle/fluid/lite/core/lite_tensor.h"
#else
#include "paddle/fluid/framework/lod_tensor.h"
#endif

namespace paddle {
namespace lite {

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
using Tensor = details::Tensor;
using DDim = details::DDim;
#else
using Tensor = framework::LoDTensor;
using DDim = framework::DDim;

static TargetType TensorGetTarget(const Tensor &x) {
  if (platform::is_gpu_place(x.place())) {
    return TARGET(kCUDA);
  } else if (platform::is_cpu_place(x.place())) {
    return TARGET(kX86);
  }
  return TARGET(kUnk);
}

template <typename T>
T *TensorMutableData(Tensor *x, TargetType target, size_t size) {
  if (target == TARGET(kX86) || target == TARGET(kHost)) {
    return x->mutable_data<T>(platform::CPUPlace(), memory::Allocator::kDefault,
                              size);
  } else if (target == TARGET(kCUDA)) {
    return x->mutable_data<T>(platform::CUDAPlace(),
                              memory::Allocator::kDefault, size);
  }
  LOG(FATAL) << "not valid target " << TargetToStr(target);
  return nullptr;
}
#endif

static int product(const DDim &dims, int start, int end) {
  int res = 1;
  for (int i = start; i < end; i++) {
    res *= dims[i];
  }
  return res;
}

static DDim SliceDims(const DDim &dims, int begin, int end) {
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  return DDim(dims[0] + begin, dims.begin() + end - 1);
#else
  auto vec = framework::vectorize(dims);
  return DDim(&vec[0] + begin, end - begin);
#endif
}

static std::vector<int64_t> DDimVectorize(const DDim &x) {
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  return x;
#else
  return framework::vectorize(x);
#endif
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
static int product(const DDim &dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         [](int a, int b) { return a * b; });
}
#endif

static DDim flatten_to_2d(const DDim &dims, int col) {
  return DDim({product(SliceDims(dims, 0, col)),
               product(SliceDims(dims, col, dims.size()))});
}

}  // namespace lite
}  // namespace paddle
