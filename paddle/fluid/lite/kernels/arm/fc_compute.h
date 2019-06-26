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
#include <stdint.h>
#include "paddle/fluid/lite/arm/math/type_trans.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/operators/fc_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class FcCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::FcParam;

  void PrepareForRun() override;

  void Run() override;

  ~FcCompute() override {
    if (transed_weight_) {
      delete transed_weight_;
    }
  };

 private:
  lite::Tensor* transed_weight_{nullptr};
  int m_, n_, k_;
};

template <PrecisionType Ptype_out>
class FcComputeInt8 : public KernelLite<TARGET(kARM), PRECISION(kInt8)> {
 public:
  using param_t = operators::FcParam;

  void PrepareForRun() override;

  void Run() override;

  ~FcComputeInt8() override {
    if (transed_weight_) {
      delete transed_weight_;
    }
  };

 private:
  lite::Tensor* transed_weight_{nullptr};
  Tensor* tmp_int32_out_{nullptr};
  int m_, n_, k_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
