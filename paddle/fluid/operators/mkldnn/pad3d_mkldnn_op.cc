/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class Pad3dMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    const T& pad_value = static_cast<T>(ctx.Attr<float>("value"));
    const std::string& mode = ctx.Attr<std::string>("mode");
    const std::string& data_format = ctx.Attr<std::string>("data_format");

    auto src_tz = phi::vectorize(input->dims());
    auto dst_tz = phi::vectorize(output->dims());

    auto paddle_dt = framework::TransToProtoVarType(input->dtype());
    dnnl::memory::data_type onednn_dt = framework::ToMKLDNNDataType(paddle_dt);

    auto dims = phi::vectorize(output->dims());

  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(pad3d, MKLDNN, paddle::platform::CPUPlace,
                   ops::Pad3dMKLDNNKernel<float>,
                   ops::Pad3dMKLDNNKernel<int8_t>,
                   ops::Pad3dMKLDNNKernel<uint8_t>,
                   ops::Pad3dMKLDNNKernel<paddle::platform::bfloat16>);
