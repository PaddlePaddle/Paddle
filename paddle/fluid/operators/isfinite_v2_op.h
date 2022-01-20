// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/transform.h"

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace operators {

struct InfinityV2Functor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorContainsInfV2(tensor, out);
  }
};

struct NANV2Functor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorContainsNANV2(tensor, out);
  }
};

struct IsfiniteV2Functor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorIsfiniteV2(tensor, out);
  }
};

}  // namespace operators
}  // namespace paddle
