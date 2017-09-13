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
#include "paddle/operators/math/activation_functor.h"

#define ACTIVATION_KERNEL_NAME(ACTIVATION_NAME) ACTIVATION_NAME##Kernel

#define DEFINE_ACTIVATION_KERNEL(ACTIVATION_NAME)                              \
  template <typename Place, typename T>                                        \
  class ACTIVATION_KERNEL_NAME(ACTIVATION_NAME) : public framework::OpKernel { \
   public:                                                                     \
    void Compute(const framework::ExecutionContext& context) const override {  \
      auto* X = context.Input<framework::Tensor>("X");                         \
      auto* Y = context.Output<framework::Tensor>("Y");                        \
      Y->mutable_data<T>(context.GetPlace());                                  \
      math::ACTIVATION_NAME<Place, T> functor;                                 \
      auto* device_context = context.device_context();                         \
      functor(*device_context, *X, Y);                                         \
    }                                                                          \
  };

#define DEFINE_ACTIVATION_GRAD_KERNEL(ACTIVATION_GRAD_NAME)                   \
  template <typename Place, typename T>                                       \
  class ACTIVATION_KERNEL_NAME(ACTIVATION_GRAD_NAME)                          \
      : public framework::OpKernel {                                          \
   public:                                                                    \
    void Compute(const framework::ExecutionContext& context) const override { \
      auto* X = context.Input<framework::Tensor>("X");                        \
      auto* Y = context.Input<framework::Tensor>("Y");                        \
      auto* dY =                                                              \
          context.Input<framework::Tensor>(framework::GradVarName("Y"));      \
      auto* dX =                                                              \
          context.Output<framework::Tensor>(framework::GradVarName("X"));     \
      dX->mutable_data<T>(context.GetPlace());                                \
      math::ACTIVATION_GRAD_NAME<Place, T> functor;                           \
      auto* device_context = context.device_context();                        \
      functor(*device_context, *X, *Y, *dY, dX);                              \
    }                                                                         \
  };

namespace paddle {
namespace operators {

DEFINE_ACTIVATION_KERNEL(Sigmoid);

DEFINE_ACTIVATION_GRAD_KERNEL(SigmoidGrad);

DEFINE_ACTIVATION_KERNEL(Exp);

DEFINE_ACTIVATION_GRAD_KERNEL(ExpGrad);

DEFINE_ACTIVATION_KERNEL(Relu);

DEFINE_ACTIVATION_GRAD_KERNEL(ReluGrad);

}  // namespace operators
}  // namespace paddle
