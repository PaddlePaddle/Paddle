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
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class ConvCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ConvParam;

  void PrepareForRun() override;

  void Run() override;

  ~ConvCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  lite::arm::math::ImplBase<TARGET(kARM), PRECISION(kFloat), param_t>* impl_{
      nullptr};
};

template <PrecisionType Ptype_out>
class ConvComputeInt8 : public KernelLite<TARGET(kARM), PRECISION(kInt8)> {
 public:
  using param_t = operators::ConvParam;

  void PrepareForRun() override;

  void Run() override;

  ~ConvComputeInt8() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  lite::arm::math::ImplBase<TARGET(kARM), PRECISION(kInt8), param_t>* impl_{
      nullptr};
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
