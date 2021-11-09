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

#include <memory>

#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T>
class FCMKLDNNHandler : public platform::MKLDNNHandlerNoCachingT<
                                   T, dnnl::inner_product_forward> {
 public:
  FCMKLDNNHandler(const Tensor* x, const Tensor* weights, const Tensor* bias,
                  Tensor* out, const int in_num_col_dims,
                  const mkldnn::engine engine, platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::inner_product_forward(
            engine, cpu_place) {

    auto md = dnnl::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);



      this->AcquireForwardPrimitiveDescriptor(
          dnnl::prop_kind::forward_inference, md, epsilon, flags);
  }
};

template <typename T, typename T_w>
class FCMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    this->RunKernel<T>(ctx);
  }

  template <typename Tout = T>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x = ctx.Input<LoDTensor>("Input");
    auto weights = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto out = ctx.Output<LoDTensor>("Out");

    auto in_col_dims = ctx.Attr<int>("in_num_col_dims");

    RecomputeOutputDims(ctx, x, weights, out);

    FCMKLDNNHandler handler(x, weights, bias, out, in_col_dims, engine, ctx.GetPlace());

  } 

  void RecomputeOutputDims(const ExecutionContext& ctx, const LoDTensor* x,
                           const Tensor* weights, LoDTensor* out) const {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights, false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w->dims(), output_dims, in_num_col_dims,
                 padding_weights);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());
  }
};



}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL(fc, MKLDNN, paddle::platform::CPUPlace,
                   ops::FCMKLDNNKernel<float, float>,
                   ops::FCMKLDNNKernel<paddle::platform::bfloat16, paddle::platform::bfloat16>
                   ops::FCMKLDNNKernel<uint8_t, int8_t>,
                   ops::FCMKLDNNKernel<int8_t, int8_t>);
