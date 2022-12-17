/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/softmax.h"

namespace paddle {
namespace operators {

template <typename T>
class CSoftmaxWithCrossEntropyOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unavailable(
        "Do not support c_embedding for cpu kernel now."));
  }
};

template <typename Context, typename T>
struct CSoftmaxWithCrossEntropyFunctor {
  void operator()(const framework::ExecutionContext& ctx);
};

template <typename Context, typename T>
struct CSoftmaxWithCrossEntropyProcessGroupFunctor {
  void operator()(const framework::ExecutionContext& ctx);
};

}  // namespace operators
}  // namespace paddle
