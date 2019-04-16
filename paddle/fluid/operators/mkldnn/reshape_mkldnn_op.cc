/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "mkldnn.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/reshape_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

template <typename T>
class ReshapeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ReshapeFunc(ctx);
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto out_dims = out->dims();

    mkldnn::memory::format dst_fmt = platform::MKLDNNFormatForSize(
        paddle::framework::vectorize2int(out_dims).size(),
        mkldnn::memory::format::nchw);
    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(dst_fmt);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(reshape2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ReshapeMKLDNNOpKernel<float>,
                   ops::ReshapeMKLDNNOpKernel<int8_t>,
                   ops::ReshapeMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL(reshape, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ReshapeMKLDNNOpKernel<float>);
