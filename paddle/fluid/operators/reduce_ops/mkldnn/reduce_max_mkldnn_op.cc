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
class ReduceMaxMKLDNNKernel : public ReduceMKLDNNKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx, dnnl::algorithm::reduction_max);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(reduce_max,
                   MKLDNN,
                   paddle::platform::CPUPlace,
                   ops::ReduceMaxMKLDNNKernel<float>,
                   ops::ReduceMaxMKLDNNKernel<paddle::platform::bfloat16>);
