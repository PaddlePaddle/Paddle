/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gaussian_random_op.h"

namespace paddle {
namespace operators {

template <typename T>
class GaussianMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    GaussianRandomKernel<platform::CPUDeviceContext, T> kernel;
    kernel.Compute(context);
    auto* tensor = context.Output<framework::Tensor>("Out");
    // The format of output is set as the mkldnn's format
    // TODO(@mozga-intel) The format of matrix sets inside the another layers.
    tensor->set_layout(framework::DataLayout::kMKLDNN);
    tensor->set_format(mkldnn::memory::format::oihw);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(gaussian_random, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::GaussianMKLDNNKernel<float>);
