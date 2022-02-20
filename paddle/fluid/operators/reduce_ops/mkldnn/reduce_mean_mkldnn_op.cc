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

#include "paddle/fluid/operators/reduce_ops/mkldnn/reduce_mkldnn_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ReduceMeanMKLDNNKernel : public ReduceMKLDNNKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx, dnnl::algorithm::reduction_mean);
  }
};

template <typename T>
class ReduceMeanGradMKLDNNKernel : public ReduceGradMKLDNNKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* input_x = ctx.Input<Tensor>("X");
    auto input_dims = phi::vectorize(input_x->dims());
    auto reduce_dims = ctx.Attr<std::vector<int>>("dim");

    int number_of_elements = 1;
    if (!ctx.Attr<bool>("reduce_all")) {
      for (size_t i = 0; i < reduce_dims.size(); ++i) {
        reduce_dims[i] = (reduce_dims[i] >= 0)
                             ? reduce_dims[i]
                             : input_dims.size() + reduce_dims[i];
        number_of_elements *= input_dims[reduce_dims[i]];
      }
    } else {
      number_of_elements = input_x->numel();
    }

    this->RunKernel(ctx, dnnl::algorithm::binary_add,
                    dnnl::algorithm::reduction_mean, 0.0f,
                    1.0L / number_of_elements);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(reduce_mean, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReduceMeanMKLDNNKernel<float>,
                   ops::ReduceMeanMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(reduce_mean_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReduceMeanGradMKLDNNKernel<float>,
                   ops::ReduceMeanGradMKLDNNKernel<paddle::platform::bfloat16>);
