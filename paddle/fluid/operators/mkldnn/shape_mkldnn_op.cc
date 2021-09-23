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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/shape_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::framework::LoDTensor;

template <typename T>
class ShapeMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_var = ctx.InputVar("Input");
    auto* out = ctx.Output<Tensor>("Out");

    framework::DDim in_dims;
    if (in_var->IsType<SelectedRows>()) {
      in_dims = in_var->Get<SelectedRows>().value().dims();
    } else {
      in_dims = in_var->Get<LoDTensor>().dims();
    }

    out->Resize({in_dims.size()});
    auto out_data = out->mutable_data<int32_t>(platform::CPUPlace());
    for (int i = 0; i < in_dims.size(); ++i) {
      out_data[i] = in_dims[i];
    }

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetPlainMKLDNNFormat(in_dims.size()));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(shape, MKLDNN, paddle::platform::CPUPlace,
                   ops::ShapeMKLDNNKernel<float>,
                   ops::ShapeMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::ShapeMKLDNNKernel<int8_t>,
                   ops::ShapeMKLDNNKernel<uint8_t>);
