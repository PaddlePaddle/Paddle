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
#include <iostream>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/elementwise_op.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

#define EIGEN_MUL(x, y) ((x) * (y))

template <typename Place, typename T>
class ElementWiseMulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElementWiseCompute<EIGEN_FUNCTOR(mul, EIGEN_MUL), Place, T>(ctx);
  }
};

template <typename Place, typename T>
class ElementWiseMulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElementWiseCompute<EIGEN_GRAD_FUNCTOR(mul, EIGEN_MUL), Place, T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
