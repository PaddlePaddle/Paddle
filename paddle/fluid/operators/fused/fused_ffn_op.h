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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/matmul_v2_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CPUFusedFfnKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    printf("no support cpu...\n");
  }
};

template <typename DeviceContext, typename T>
class CPUFusedFfnGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

}  // namespace operators
}  // namespace paddle
