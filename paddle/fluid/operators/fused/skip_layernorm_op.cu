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

#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SkipLayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    auto *X = context.Input<framework::Tensor>("X");
    auto *Y = context.Input<framework::Tensor>("Y");
    auto *scale = context.Input<framework::Tensor>("Scale");
    auto *bias = context.Input<framework::Tensor>("Bias");

    auto *X_d = X->data<T>();
    auto *Y_d = Y->data<T>();
    auto *scale_d = scale->data<T>();
    auto *bias_d = bias->data<T>();
    float epsilon = context.Attr<float>("epsilon");
    int begin_norm_axis = context.Attr<int>("begin_norm_axis");

    auto *out = context.Output<framework::Tensor>("Out");
    out->Resize(X->dims());
    auto *output_d = out->mutable_data<T>(context.GetPlace());

    size_t num = 1;
    for (size_t i = 0; i < X->dims().size(); i++) {
      num *= X->dims()[i];
    }
    int hidden = X->dims()[2];
    auto &device_ctx = context.template device_context<DeviceContext>();
    operators::math::SkipLayerNormFunctor<T> skip_layer_norm_func;

    skip_layer_norm_func(num, hidden, X_d, Y_d, scale_d, bias_d, output_d,
                         epsilon, device_ctx.stream());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    skip_layernorm,
    ops::SkipLayerNormKernel<paddle::platform::CUDADeviceContext, float>);
