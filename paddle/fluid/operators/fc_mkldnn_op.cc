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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

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
                                   mkldnn::memory::format::nc);
  }

  mkldnn::memory::desc src() const {
    return is_spatial()
               ? platform::MKLDNNMemDesc({in[0], in[1], in[2], in[3]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::nchw)
               : platform::MKLDNNMemDesc({in[0], in[1]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::nc);
  }

  mkldnn::memory::desc weights() const {
    return is_spatial()
               ? platform::MKLDNNMemDesc({w[1], in[1], in[2], in[3]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::oihw)
               : platform::MKLDNNMemDesc({w[1], in[1]},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::oi);
  }

  mkldnn::memory::desc bias() const {
    return with_bias_
               ? platform::MKLDNNMemDesc({w[1]}, mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::format_undef)
               : platform::MKLDNNMemDesc({}, mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::format_undef);
  }

 private:
  bool is_spatial() const { return in.size() > 1 && w.size() > 1; }

  std::vector<int> in;
  std::vector<int> w;
  bool with_bias_;
  bool is_spatial_;
};

class MKLDNNMemory {
 public:
  MKLDNNMemory(MKLDNNMD<Tensor>* t, const mkldnn::engine& e)
      : md_(t), engine_(e) {}
  virtual ~MKLDNNMemory() = default;

  template <typename Output>
  mkldnn::memory dst(const Output* out) {
    return mkldnn::memory({md_->dst(), engine_},
                          static_cast<void*>(const_cast<float*>(out)));
  }

  template <typename Output>
  mkldnn::memory dst(Output* out) {
    return mkldnn::memory({md_->dst(), engine_}, out);
  }

  template <typename Input>
  mkldnn::memory src(const Input* in) {
    return mkldnn::memory({md_->src(), engine_},
                          static_cast<void*>(const_cast<float*>(in)));
  }

  template <typename Weight>
  mkldnn::memory weights(const Weight* w) {
    return mkldnn::memory({md_->weights(), engine_},
                          static_cast<void*>(const_cast<float*>(w)));
  }

  mkldnn::memory bias() {
    return mkldnn::memory(mkldnn::memory::primitive_desc(md_->bias(), engine_));
  }

 private:
  MKLDNNMD<Tensor>* md_;
  const mkldnn::engine& engine_;
};

template <typename T>
class FCMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");

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

    MKLDNNMemory mem(&md, mkldnn_engine);

    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();

    auto output = ctx.Output<Tensor>("Out");
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    auto dst_memory = mem.dst(output_data);
    auto src_memory = mem.src(input_data);
    auto weights_memory = mem.weights(w_data);
    auto bias_memory = mem.bias();

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
};

template <typename T>
class FCMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    T* input_grad_data = nullptr;
    T* w_grad_data = nullptr;

    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* w_grad = ctx.Output<Tensor>(framework::GradVarName("W"));

    if (input_grad) {
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (w_grad) {
      w_grad_data = w_grad->mutable_data<T>(ctx.GetPlace());
    }

    const Tensor* input = ctx.Input<Tensor>("Input");
    const T* input_data = input->data<T>();

    const Tensor* w = ctx.Input<Tensor>("W");
    const T* w_data = w->data<T>();

    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    const T* out_grad_data = out_grad->data<T>();

    bool with_bias = ctx.Attr<bool>("bias_attr");

    MKLDNNMD<Tensor> md(input, w, with_bias);
    MKLDNNMemory mem(&md, mkldnn_engine);

    auto dst_memory = mem.dst(out_grad_data);
    auto src_memory = mem.src(input_data);
    auto weights_memory = mem.weights(w_data);
    auto bias_memory = mem.bias();

    const std::string key = ctx.op().Input("Out");
    const std::string key_fc_pd = key + "@fc_pd";

    auto pd =
        std::static_pointer_cast<mkldnn::inner_product_forward::primitive_desc>(
            dev_ctx.GetBlob(key_fc_pd));

    PADDLE_ENFORCE(pd != nullptr, "Fail to find key_fc_pd in device context");

    if (w_grad) {
      auto weights_grad_memory = mem.weights(w_grad_data);

      mkldnn::inner_product_backward_weights::primitive_desc bwd_weight_pd =
          FcBwdWeightsPrimitiveDesc(md.src(), md.weights(), md.dst(), md.bias(),
                                    with_bias, *pd, mkldnn_engine);

      auto bwd_weights_prim = mkldnn::inner_product_backward_weights(
          bwd_weight_pd, src_memory, dst_memory, weights_grad_memory,
          bias_memory);

      std::vector<mkldnn::primitive> pipeline{bwd_weights_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }

    if (input_grad) {
      auto src_grad_memory = mem.src(input_grad_data);

      mkldnn::inner_product_backward_data::primitive_desc bwd_data_pd =
          FcBwdDataPrimitiveDesc(md.src(), md.weights(), md.dst(), *pd,
                                 mkldnn_engine);

      auto bwd_data_prim = mkldnn::inner_product_backward_data(
          bwd_data_pd, dst_memory, weights_memory, src_grad_memory);

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
