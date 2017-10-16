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
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;
using framework::Tensor;

template <typename Place, typename T>
class LSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_t = ctx.Input<framework::LoDTensor>("Input");
    auto* batch_t = ctx.Input<framework::LoDTensor>("Batch");
    auto* bias_t = ctx.Input<framework::LoDTensor>("Bias");
    bool is_reverse = ctx.Attr<bool>("is_reverse");
    LoDTensor2BatchFunctor<Place, T> to_batch(ctx.device_context(), input_t,
                                              batch_t, is_reverse);

    auto in_dims = input_t->dims();
    int frame_size = in_dims[1];

    if (bias_t) {
      auto b = EigenMatrix<T>::From(*bias);
    }
  }
};

template <typename Place, typename T>
class LSTMGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
