/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

enum TensorFormat {
  NHWC = 0,
  NCHW = 1,
};

inline TensorFormat StringToTensorFormat(const std::string& str) {
  if (str == "NHWC" || str == "nhwc") {
    return TensorFormat::NHWC;
  } else if (str == "NCHW" || str == "nchw") {
    return TensorFormat::NCHW;
  } else {
    PADDLE_THROW("Unknown storage order string: %s", str);
  }
}

template <typename Place, typename T>
class BatchNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename Place, typename T>
class BatchNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle
