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

#include "paddle/fluid/framework/mkldnn_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::MKLDNNTensor;
using framework::MKLDNNTensorMutable;
using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

template <typename T>
class MKLDNNMD {
 public:
  explicit MKLDNNMD(const T* in, const T* w, bool bias)
      : in(paddle::framework::vectorize2int(in->dims())),
        w(paddle::framework::vectorize2int(w->dims())) {
    with_bias_ = bias;
  }

  mkldnn::memory::desc dst() const {
    return platform::MKLDNNMemDesc({in[0], w[1]},
                                   mkldnn::memory::data_type::f32,
                                   mkldnn::memory::format::any);
  }

  mkldnn::memory::desc src() const {
    return is_spatial()
               ? platform::MKLDNNMemDesc({in[0], in[1], in[2], in[3]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any)
               : platform::MKLDNNMemDesc({in[0], in[1]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any);
  }

  mkldnn::memory::desc weights() const {
    return is_spatial()
               ? platform::MKLDNNMemDesc({w[1], in[1], in[2], in[3]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any)
               : platform::MKLDNNMemDesc({w[1], in[1]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any);
  }

  mkldnn::memory::desc bias() const {
    return with_bias_
               ? platform::MKLDNNMemDesc({w[1]}, mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any)
               : platform::MKLDNNMemDesc({}, mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any);
  }

 private:
  bool is_spatial() const { return in.size() > 1 && w.size() > 1; }

  std::vector<int> in;
  std::vector<int> w;
  bool with_bias_;
  bool is_spatial_;
};

template <typename T>
class FCMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.MutableInput<Tensor>("Input");
    auto w = ctx.MutableInput<Tensor>("W");

    PADDLE_ENFORCE(input->dims().size() == 2 || input->dims().size() == 4,
                   "Input must be with 2 or 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(w->dims().size() == 2 || w->dims().size() == 4,
                   "Weights must be with 2 or 4 dimensions, i.e. OI or OIHW");

    bool with_bias = ctx.Attr<bool>("bias_attr");
    MKLDNNMD<Tensor> md(input, w, with_bias);

    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> pd =
        FcFwdPrimitiveDesc(md.src(), md.weights(), md.dst(), md.bias(),
                           with_bias, mkldnn_engine);

    const std::string key = ctx.op().Output("Out");
    const std::string key_fc_pd = key + "@fc_pd";

    dev_ctx.SetBlob(key_fc_pd, pd);

    auto output = ctx.Output<Tensor>("Out");

    MKLDNNTensorMutable input_mkldnn =
        MKLDNNTensorMutable::Create(input, mkldnn_engine);
    MKLDNNTensorMutable w_mkldnn =
        MKLDNNTensorMutable::Create(w, mkldnn_engine);
    MKLDNNTensorMutable output_mkldnn =
        MKLDNNTensorMutable::Create(output, mkldnn_engine);

    Reorder(pd, &input_mkldnn, &w_mkldnn, &output_mkldnn);

    auto dst_memory = output_mkldnn.GetMemory();
    auto src_memory = input_mkldnn.GetMemory();
    auto weights_memory = w_mkldnn.GetMemory();
    auto bias_memory = mkldnn::memory(
        mkldnn::memory::primitive_desc(md.bias(), mkldnn_engine));

    auto forward = with_bias ? mkldnn::inner_product_forward(
                                   *pd, src_memory, weights_memory, bias_memory,
                                   dst_memory)
                             : mkldnn::inner_product_forward(
                                   *pd, src_memory, weights_memory, dst_memory);

    std::vector<mkldnn::primitive> pipeline = {forward};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }

 private:
  std::unique_ptr<mkldnn::inner_product_forward::primitive_desc>
  FcFwdPrimitiveDesc(const mkldnn::memory::desc& src,
                     const mkldnn::memory::desc& weights,
                     const mkldnn::memory::desc& dst,
                     const mkldnn::memory::desc& bias, const bool with_bias,
                     const mkldnn::engine& engine) const {
    auto desc = with_bias
                    ? mkldnn::inner_product_forward::desc(
                          mkldnn::prop_kind::forward, src, weights, bias, dst)
                    : mkldnn::inner_product_forward::desc(
                          mkldnn::prop_kind::forward, src, weights, dst);

    auto pd = new mkldnn::inner_product_forward::primitive_desc(desc, engine);
    return std::unique_ptr<mkldnn::inner_product_forward::primitive_desc>(pd);
  }

  void Reorder(
      std::shared_ptr<const mkldnn::inner_product_forward::primitive_desc> pd,
      MKLDNNTensorMutable* input, MKLDNNTensorMutable* weights,
      MKLDNNTensorMutable* output) const {
    auto input_format = mkldnn::memory::primitive_desc(pd->src_primitive_desc())
                            .desc()
                            .data.format;
    auto weights_format =
        mkldnn::memory::primitive_desc(pd->weights_primitive_desc())
            .desc()
            .data.format;
    auto output_format =
        mkldnn::memory::primitive_desc(pd->dst_primitive_desc())
            .desc()
            .data.format;

    if (input->GetFormat() != input_format) {
      VLOG(3) << "input " << input->GetFormat() << " -> " << input_format;
    }
    if (weights->GetFormat() != weights_format) {
      VLOG(3) << "filter " << weights->GetFormat() << " -> " << weights_format;
    }
    if (output->GetFormat() != output_format) {
      VLOG(3) << "output " << output->GetFormat() << " -> " << output_format;
    }

    input->Reorder(static_cast<mkldnn::memory::format>(input_format));
    weights->Reorder(static_cast<mkldnn::memory::format>(weights_format));
    output->SetFormat(static_cast<mkldnn::memory::format>(output_format));
  }
};

template <typename T>
class FCMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* w_grad = ctx.Output<Tensor>(framework::GradVarName("W"));

    Tensor* input = ctx.MutableInput<Tensor>("Input");
    Tensor* w = ctx.MutableInput<Tensor>("W");
    Tensor* out_grad = ctx.MutableInput<Tensor>(framework::GradVarName("Out"));

    bool with_bias = ctx.Attr<bool>("bias_attr");

    MKLDNNMD<Tensor> md(input, w, with_bias);

    MKLDNNTensorMutable input_mkldnn =
        MKLDNNTensorMutable::Create(input, mkldnn_engine);
    MKLDNNTensorMutable w_mkldnn =
        MKLDNNTensorMutable::Create(w, mkldnn_engine);
    MKLDNNTensorMutable out_grad_mkldnn =
        MKLDNNTensorMutable::Create(out_grad, mkldnn_engine);

    auto bias_memory = mkldnn::memory(
        mkldnn::memory::primitive_desc(md.bias(), mkldnn_engine));

    const std::string key = ctx.op().Input("Out");
    const std::string key_fc_pd = key + "@fc_pd";

    auto pd =
        std::static_pointer_cast<mkldnn::inner_product_forward::primitive_desc>(
            dev_ctx.GetBlob(key_fc_pd));

    PADDLE_ENFORCE(pd != nullptr, "Fail to find key_fc_pd in device context");

    if (w_grad) {
      mkldnn::inner_product_backward_weights::primitive_desc bwd_weight_pd =
          FcBwdWeightsPrimitiveDesc(md.src(), md.weights(), md.dst(), md.bias(),
                                    with_bias, *pd, mkldnn_engine);

      MKLDNNTensorMutable weights_grad_mkldnn =
          MKLDNNTensorMutable::Create(w_grad, mkldnn_engine);

      // Reorder(bwd_weight_pd, input_mkldnn, out_grad_mkldnn,
      // weights_grad_mkldnn);

      auto bwd_weights_prim = mkldnn::inner_product_backward_weights(
          bwd_weight_pd, input_mkldnn.GetMemory(), out_grad_mkldnn.GetMemory(),
          w_mkldnn.GetMemory(), bias_memory);

      std::vector<mkldnn::primitive> pipeline{bwd_weights_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }

    if (input_grad) {
      mkldnn::inner_product_backward_data::primitive_desc bwd_data_pd =
          FcBwdDataPrimitiveDesc(md.src(), md.weights(), md.dst(), *pd,
                                 mkldnn_engine);

      MKLDNNTensorMutable src_grad_mkldnn =
          MKLDNNTensorMutable::Create(input_grad, mkldnn_engine);

      // Reorder(bwd_data_pd, out_grad_mkldnn, w_mkldnn, src_grad_mkldnn);

      auto bwd_data_prim = mkldnn::inner_product_backward_data(
          bwd_data_pd, out_grad_mkldnn.GetMemory(), w_mkldnn.GetMemory(),
          src_grad_mkldnn.GetMemory());

      std::vector<mkldnn::primitive> pipeline{bwd_data_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }
  }

 private:
  mkldnn::inner_product_backward_weights::primitive_desc
  FcBwdWeightsPrimitiveDesc(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& diff_weights,
      const mkldnn::memory::desc& diff_dst, const mkldnn::memory::desc& bias,
      const bool with_bias,
      const mkldnn::inner_product_forward::primitive_desc& pd,
      const mkldnn::engine& engine) const {
    auto bwd_weight_desc = with_bias
                               ? mkldnn::inner_product_backward_weights::desc(
                                     src, diff_weights, bias, diff_dst)
                               : mkldnn::inner_product_backward_weights::desc(
                                     src, diff_weights, bias, diff_dst);

    return mkldnn::inner_product_backward_weights::primitive_desc(
        bwd_weight_desc, engine, pd);
  }

  mkldnn::inner_product_backward_data::primitive_desc FcBwdDataPrimitiveDesc(
      const mkldnn::memory::desc& diff_src, const mkldnn::memory::desc& weights,
      const mkldnn::memory::desc& diff_dst,
      const mkldnn::inner_product_forward::primitive_desc& pd,
      const mkldnn::engine& engine) const {
    auto bwd_data_desc =
        mkldnn::inner_product_backward_data::desc(diff_src, weights, diff_dst);
    return mkldnn::inner_product_backward_data::primitive_desc(bwd_data_desc,
                                                               engine, pd);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(fc, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(fc_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNGradOpKernel<float>);
