/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

template <typename Format = mkldnn::memory::format>
mkldnn::memory::desc type(const std::vector<int>& dims, Format&& f) {
  return platform::MKLDNNMemDesc(dims, mkldnn::memory::data_type::f32, f);
}

template <typename T>
class MulMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<Tensor>("X");
    auto weight = ctx.Input<Tensor>("Y");

    PADDLE_ENFORCE(input->dims().size() & (2 | 4),
                   "Input must be with 2 or 4 dimensions, i.e. NC or NCHW");
    PADDLE_ENFORCE(weight->dims().size() & (2 | 4),
                   "Weights must be with 2 or 4 dimensions, i.e. OI or OIHW");

    std::vector<int> w_tz = paddle::framework::vectorize2int(weight->dims());
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());

    auto src_md =
        src_tz.size() != 2
            ? type(src_tz, mkldnn::memory::format::nchw)
            : type({src_tz[0], src_tz[1]}, mkldnn::memory::format::nc);

    auto dst_md = type({src_tz[0], w_tz[1]}, mkldnn::memory::format::nc);

    auto weights_md =
        src_tz.size() != 2
            ? type({w_tz[1], src_tz[1], src_tz[2], src_tz[3]},
                   mkldnn::memory::format::oihw)
            : type({w_tz[1], src_tz[1]}, mkldnn::memory::format::oi);

    auto output = ctx.Output<Tensor>("Out");
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    const std::string key = ctx.op().Output("Out");
    const std::string key_fc_pd = key + "@mul_pd";

    const T* input_data = input->data<T>();
    const T* w_data = weight->data<T>();

    auto dst_memory = mkldnn::memory({dst_md, mkldnn_engine}, output_data);

    auto src_memory = mkldnn::memory({src_md, mkldnn_engine},
                                     platform::to_void_cast(input_data));

    auto weights_memory = mkldnn::memory({weights_md, mkldnn_engine},
                                         platform::to_void_cast(w_data));

    auto pd = platform::MKLDNNFwdPrimitiveDesc<mkldnn::inner_product_forward>(
        mkldnn_engine, src_md, weights_md, dst_md);

    dev_ctx.SetBlob(key_fc_pd, pd);

    auto forward = mkldnn::inner_product_forward(*pd, src_memory,
                                                 weights_memory, dst_memory);

    std::vector<mkldnn::primitive> pipeline = {forward};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }
};

template <typename T>
class MulMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("X");
    const Tensor* w = ctx.Input<Tensor>("Y");

    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* w_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));

    const std::string key = ctx.op().Input("Out");
    const std::string key_fc_pd = key + "@mul_pd";

    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    T* input_grad_data = nullptr;
    T* w_grad_data = nullptr;

    if (input_grad) {
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (w_grad) {
      w_grad_data = w_grad->mutable_data<T>(ctx.GetPlace());
    }

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> w_tz = paddle::framework::vectorize2int(w->dims());

    auto src_md =
        src_tz.size() != 2
            ? type(src_tz, mkldnn::memory::format::nchw)
            : type({src_tz[0], src_tz[1]}, mkldnn::memory::format::nc);

    auto dst_md = type({src_tz[0], w_tz[1]}, mkldnn::memory::format::nc);

    auto weights_md =
        src_tz.size() != 2
            ? type({w_tz[1], src_tz[1], src_tz[2], src_tz[3]},
                   mkldnn::memory::format::oihw)
            : type({w_tz[1], src_tz[1]}, mkldnn::memory::format::oi);

    auto src_memory = mkldnn::memory({src_md, mkldnn_engine},
                                     platform::to_void_cast(input_data));

    auto dst_memory = mkldnn::memory({dst_md, mkldnn_engine},
                                     platform::to_void_cast(out_grad_data));

    auto weight_memory = mkldnn::memory({weights_md, mkldnn_engine},
                                        platform::to_void_cast(w_data));

    auto pd =
        std::static_pointer_cast<mkldnn::inner_product_forward::primitive_desc>(
            dev_ctx.GetBlob(key_fc_pd));

    PADDLE_ENFORCE(pd != nullptr, "Fail to find pd in device context");

    if (w_grad) {
      auto weights_grad_memory = mkldnn::memory(
          {weights_md, mkldnn_engine}, platform::to_void_cast(w_grad_data));

      auto bwd_weight_pd = platform::MKLDNNBwdPrimitiveDesc<
          mkldnn::inner_product_backward_weights>(mkldnn_engine, *pd, src_md,
                                                  weights_md, dst_md);

      auto bwd_weights_prim = mkldnn::inner_product_backward_weights(
          bwd_weight_pd, src_memory, dst_memory, weights_grad_memory);

      std::vector<mkldnn::primitive> pipeline{bwd_weights_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }

    if (input_grad) {
      auto src_grad_memory = mkldnn::memory(
          {src_md, mkldnn_engine}, platform::to_void_cast(input_grad_data));

      auto bwd_data_pd =
          platform::MKLDNNBwdPrimitiveDesc<mkldnn::inner_product_backward_data>(
              mkldnn_engine, *pd, src_md, weights_md, dst_md);

      auto bwd_data_prim = mkldnn::inner_product_backward_data(
          bwd_data_pd, dst_memory, weight_memory, src_grad_memory);

      std::vector<mkldnn::primitive> pipeline{bwd_data_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(mul, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::MulMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(mul_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::MulMKLDNNGradOpKernel<float>);
