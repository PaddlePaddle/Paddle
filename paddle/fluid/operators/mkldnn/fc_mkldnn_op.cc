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
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::vectorize;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T>
class FCMKLDNNHandler : public platform::MKLDNNHandlerNoCachingT<T, dnnl::inner_product_forward> {
 public:
  FCMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                  const platform::MKLDNNDeviceContext& dev_ctx, 
                  const Tensor* x, const Tensor* weights, const Tensor* bias,
                  Tensor* out, const int in_num_col_dims, dnnl::engine mkldnn_engine,
                  platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::inner_product_forward>(
            mkldnn_engine, cpu_place), dev_ctx_(dev_ctx) {
    this->memory_key_ = ctx.InputName("W");

    auto x_vec_dims = framework::vectorize(x->dims());
    auto weights_vec_dims = framework::vectorize(weights->dims());

    int MB = 1;

    for(int i = 0; i < in_num_col_dims; ++i) {
      MB *= x_vec_dims[i];
    }

    int IC = x_vec_dims[x_vec_dims.size() - 1];
    int OC = weights_vec_dims[1];

    dnnl::memory::desc bias_md;

    auto src_md = dnnl::memory::desc({MB, IC}, MKLDNNGetDataType<T>(), dnnl::memory::format_tag::any);
    auto weights_md = dnnl::memory::desc({OC, IC}, MKLDNNGetDataType<T>(), dnnl::memory::format_tag::any);
    auto dst_md = dnnl::memory::desc({MB, OC}, MKLDNNGetDataType<T>(), dnnl::memory::format_tag::any);
    if(bias) 
      bias_md = dnnl::memory::desc({bias->numel()}, MKLDNNGetDataType<float>(), dnnl::memory::format_tag::a);

    dnnl::primitive_attr attrs;
    HandlePostOps(ctx, attrs);

    this->AcquireForwardPrimitiveDescriptor(attrs, prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);
  }

private:
  void HandlePostOps(const paddle::framework::ExecutionContext& ctx, dnnl::primitive_attr& attrs) {
    static std::unordered_map<std::string, dnnl::algorithm> algo_map = {
      {"relu", dnnl::algorithm::eltwise_relu},
      {"gelu", dnnl::algorithm::eltwise_gelu},
      {"gelu_tanh", dnnl::algorithm::eltwise_gelu_tanh},
      {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
      {"tanh", dnnl::algorithm::eltwise_tanh},
      {"sigmoid", dnnl::algorithm::eltwise_logistic},
      {"hard_swish", dnnl::algorithm::eltwise_hardswish},
    };

    std::string activation_type = ctx.Attr<std::string>("activation_type");

    if(activation_type.empty() == false) {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;

      dnnl::post_ops post_ops;
      post_ops.append_eltwise(scale, algo_map[activation_type], alpha, beta);
      attrs.set_post_ops(post_ops);
    }
  }

  std::string memory_key_;
  const platform::MKLDNNDeviceContext& dev_ctx_;

 public:
  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(const Tensor* x) {
    const T* x_data = x->data<T>();

    auto user_md = dnnl::memory::desc(framework::vectorize(x->dims()), MKLDNNGetDataType<T>(), x->format());
    if(x->dims().size() != 2) {
      // reshape restrictions are always satisfied because in case of 3 or 4 dim input, plain layout is enforced
      user_md = user_md.reshape(this->fwd_pd_->src_desc().dims());
    }

    const auto src_md = this->fwd_pd_->src_desc();

    return this->AcquireMemoryWithReorder(user_md, src_md, to_void_cast<T>(x_data));
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemory(const Tensor* bias) {
    const float* bias_data = bias->data<float>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->bias_desc(),
                                            to_void_cast<float>(bias_data));
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(const Tensor* weights) {
    const std::string weights_key = this->memory_key_ + "@weights";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(weights_key));

    std::cout<<weights_key<<std::endl;

    if(!memory_p) {
      const T* weights_data = weights->data<T>();
      auto weights_dims = this->fwd_pd_->weights_desc().dims();

      auto user_md = dnnl::memory::desc(weights_dims, MKLDNNGetDataType<T>(), dnnl::memory::format_tag::io);
      memory_p = this->AcquireMemoryWithReorder(user_md, this->fwd_pd_->weights_desc(), to_void_cast<T>(weights_data));
      
      this->dev_ctx_.SetBlob(weights_key, memory_p);
    }
    return memory_p;
  }
};


template <typename T, typename T_w>
class FCMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
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

    FCMKLDNNHandler<T> handler(ctx, dev_ctx, x, weights, bias, out, in_col_dims, mkldnn_engine, ctx.GetPlace());

    //auto user_src_md = dnnl::memory::desc(framework::vectorize(x->dims(), MKLDNNGetDataType<T>(), x->format()));
    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
    auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(weights);

    auto dst_memory_p = handler.AcquireDstMemory(out);

    auto fc_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    
    std::unordered_map<int, dnnl::memory> fc_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}
    };

    if(bias) {
      auto bias_memory_p = handler.AcquireBiasMemory(bias);
      fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }
    
    fc_p->execute(astream, fc_args);
    astream.wait();
  } 

  void RecomputeOutputDims(const ExecutionContext& ctx, const LoDTensor* x,
                           const Tensor* weights, LoDTensor* out) const {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights, false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
    std::vector<int64_t> output_dims;
    FCOutputSize(x->dims(), weights->dims(), output_dims, in_num_col_dims,
                 padding_weights);
    out->Resize(framework::make_ddim(output_dims));
    out->set_lod(x->lod());
  }
};



}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    fc, MKLDNN, ::paddle::platform::CPUPlace, BF16, ops::kFCMKLDNNFP32,
    ops::FCMKLDNNKernel<paddle::platform::bfloat16,
                          paddle::platform::bfloat16>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNKernel<int8_t, int8_t>);
