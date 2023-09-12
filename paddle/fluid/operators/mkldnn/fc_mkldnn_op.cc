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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

using framework::ExecutionContext;
using phi::OneDNNContext;
using phi::funcs::OneDNNGetDataType;
using phi::funcs::to_void_cast;

struct InnerProductCache {
  dnnl::inner_product_forward inner_product_p;
  dnnl::memory src_mem;
  dnnl::memory weights_mem;
  dnnl::memory bias_mem;
  dnnl::memory dst_mem;
  dnnl::memory src_scales_mem;
  dnnl::memory wei_scales_mem;
  dnnl::memory dst_scales_mem;
};

std::tuple<std::vector<float>,
           std::vector<float>,
           std::vector<float>,
           std::vector<float>>
GetDNNLScales(const ExecutionContext& ctx) {
  auto scale_in_data = ctx.Attr<float>("Scale_in");
  auto scale_out = ctx.Attr<float>("Scale_out");
  auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");

  std::vector<float> dnnl_src_scales = {1.f / scale_in_data};
  size_t count = scale_weights_data.size();
  std::vector<float> dnnl_wei_scales(count);
#pragma omp parallel for if (count > 50)
  for (size_t i = 0; i < count; i++) {
    dnnl_wei_scales[i] = 1.f / scale_weights_data[i];
  }
  std::vector<float> dnnl_psum_scales = {1.f};
  std::vector<float> dnnl_dst_scales = {1.f / scale_out};

  return std::make_tuple(
      dnnl_src_scales, dnnl_wei_scales, dnnl_psum_scales, dnnl_dst_scales);
}

template <typename T_in, typename T_w, typename T_out>
class FCMKLDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T_in,
                                                 dnnl::inner_product_forward> {
 public:
  FCMKLDNNHandler(const ExecutionContext& ctx,
                  const OneDNNContext& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* weights,
                  const phi::DenseTensor* bias,
                  phi::DenseTensor* out UNUSED,
                  const int in_num_col_dims,
                  dnnl::engine onednn_engine,
                  platform::Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T_in, dnnl::inner_product_forward>(
            onednn_engine, cpu_place),
        dev_ctx_(dev_ctx) {
    this->memory_key_ = ctx.InputName("W");

    auto x_vec_dims = phi::vectorize(x->dims());
    auto weights_vec_dims = phi::vectorize(weights->dims());

    int MB = 1;
    for (int i = 0; i < in_num_col_dims; ++i) {
      MB *= x_vec_dims[i];
    }

    int IC = 1;
    for (size_t i = in_num_col_dims; i < x_vec_dims.size(); ++i) {
      IC *= x_vec_dims[i];
    }

    int OC = weights_vec_dims[1];

    dnnl::memory::desc bias_md;

    auto src_md = dnnl::memory::desc(
        {MB, IC}, OneDNNGetDataType<T_in>(), dnnl::memory::format_tag::any);
    auto weights_md = dnnl::memory::desc(
        {OC, IC}, OneDNNGetDataType<T_w>(), dnnl::memory::format_tag::any);
    auto dst_md = dnnl::memory::desc(
        {MB, OC}, OneDNNGetDataType<T_out>(), dnnl::memory::format_tag::any);
    if (bias) {
      bias_md = dnnl::memory::desc({bias->numel()},
                                   OneDNNGetDataType<float>(),
                                   dnnl::memory::format_tag::a);
    }

    const auto attrs = CreateFCAttrs(ctx);

    this->AcquireForwardPrimitiveDescriptor(attrs,
                                            dnnl::prop_kind::forward_inference,
                                            src_md,
                                            weights_md,
                                            bias_md,
                                            dst_md);
  }

 private:
  dnnl::primitive_attr CreateFCAttrs(const ExecutionContext& ctx) {
    dnnl::primitive_attr attributes;
    dnnl::post_ops post_operations;

    float activation_scale = 1.0f;
    if (phi::funcs::is_int8<T_w>()) {
      std::vector<float> src_scales, wei_scales, psum_scales, dst_scales;
      std::tie(src_scales, wei_scales, psum_scales, dst_scales) =
          GetDNNLScales(ctx);

      bool force_fp32_output = ctx.HasAttr("force_fp32_output") &&
                               ctx.Attr<bool>("force_fp32_output");

      attributes.set_scales_mask(DNNL_ARG_SRC, 0);

      dnnl::memory::desc src_scales_md(
          {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
      src_scales_mem_ = dnnl::memory(src_scales_md, this->engine_);
      memcpy(src_scales_mem_.get_data_handle(),
             src_scales.data(),
             src_scales.size() * sizeof(float));

      int mask = wei_scales.size() > 1 ? 1 : 0;
      attributes.set_scales_mask(DNNL_ARG_WEIGHTS, mask);

      dnnl::memory::desc wei_scales_md(
          {static_cast<int64_t>(wei_scales.size())},
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::x);
      wei_scales_mem_ = dnnl::memory(wei_scales_md, this->engine_);
      memcpy(wei_scales_mem_.get_data_handle(),
             wei_scales.data(),
             wei_scales.size() * sizeof(float));

      if (!force_fp32_output) {
        attributes.set_scales_mask(DNNL_ARG_DST, 0);

        dnnl::memory::desc dst_scales_md(
            {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
        dst_scales_mem_ = dnnl::memory(dst_scales_md, this->engine_);
        memcpy(dst_scales_mem_.get_data_handle(),
               dst_scales.data(),
               dst_scales.size() * sizeof(float));
      }
    }

    // ReLU from "fc_fuse_pass"
    if (ctx.Attr<std::string>("activation_type") == "relu") {
      post_operations.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    AppendActivation(ctx, post_operations, activation_scale);

    if (ctx.HasAttr("fused_output_scale")) {
      float scale_alpha = ctx.Attr<float>("fused_output_scale");
      post_operations.append_eltwise(
          dnnl::algorithm::eltwise_linear, scale_alpha, 0.0f);
    }

    attributes.set_post_ops(post_operations);
    return attributes;
  }

  void AppendActivation(const ExecutionContext& ctx,
                        dnnl::post_ops& post_ops,  // NOLINT
                        float activation_scale = 1.0f) {
    const auto invalid_attribute =
        ctx.HasAttr("fuse_activation")
            ? ctx.Attr<std::string>("fuse_activation").empty()
            : true;
    if (invalid_attribute) return;

    const auto fuse_activation = ctx.Attr<std::string>("fuse_activation");
    const auto fuse_alpha =
        ctx.HasAttr("fuse_alpha") ? ctx.Attr<float>("fuse_alpha") : 0.0f;
    const auto fuse_beta =
        ctx.HasAttr("fuse_beta") ? ctx.Attr<float>("fuse_beta") : 0.0f;

    const auto activation_map = phi::funcs::OneDNNActivationMap();
    const auto& activation_type = activation_map.find(fuse_activation);

    PADDLE_ENFORCE_NE(
        activation_type,
        activation_map.end(),
        phi::errors::InvalidArgument(
            "Activation '%s' not found in oneDNN algorithms mapper",
            fuse_activation));

    post_ops.append_eltwise(activation_type->second, fuse_alpha, fuse_beta);
    post_ops.append_eltwise(
        dnnl::algorithm::eltwise_linear, activation_scale, 0.0f);
  }

  // Computing oneDNN's scaling mask which determines along which dimension
  // slice should the scaling be applied.
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorderAndAttrs(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const dnnl::primitive_attr& attrs,
      const std::vector<float>& scale_data) {
    std::shared_ptr<dnnl::memory> target_memory_p;

    auto user_memory_p =
        std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
    target_memory_p = std::make_shared<dnnl::memory>(target_md, this->engine_);
    auto reorder_p = std::make_shared<dnnl::reorder>(
        *user_memory_p, *target_memory_p, attrs);

    auto scales_md =
        dnnl::memory::desc({static_cast<int64_t>(scale_data.size())},
                           dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::x);
    auto scale_mem =
        dnnl::memory(scales_md,
                     this->engine_,
                     phi::funcs::to_void_cast<float>(scale_data.data()));

    auto& astream = OneDNNContext::tls().get_stream();
    {
      reorder_p->execute(astream,
                         {{DNNL_ARG_FROM, *user_memory_p},
                          {DNNL_ARG_TO, *target_memory_p},
                          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem}});
      astream.wait();
    }

    return target_memory_p;
  }

  std::string memory_key_;
  const OneDNNContext& dev_ctx_;
  dnnl::memory src_scales_mem_;
  dnnl::memory wei_scales_mem_;
  dnnl::memory dst_scales_mem_;

 public:
  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* x) {
    const T_in* x_data = x->data<T_in>();

    auto user_md = x->mem_desc();
    if (x->dims().size() != 2) {
      // reshape restrictions are always satisfied because in case of 3 or 4 dim
      // input, plain layout is enforced
      user_md = user_md.reshape(this->fwd_pd_->src_desc().get_dims());
    }

    return this->AcquireMemoryWithReorder(
        user_md, this->fwd_pd_->src_desc(), to_void_cast<T_in>(x_data));
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const ExecutionContext& ctx, const phi::DenseTensor* bias) {
    const float* bias_data = bias->data<float>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->bias_desc(),
                                            to_void_cast<float>(bias_data));
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const phi::DenseTensor* weights, const std::vector<float>& scale_data) {
    const std::string weights_base_key = this->memory_key_ + "@weights";
    std::string weights_key;
    weights_key.reserve(128);
    weights_key = phi::funcs::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx_,
        phi::funcs::CreateKey(
            dev_ctx_, weights_base_key, this->fwd_pd_->weights_desc()));
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(weights_key));

    if (!memory_p) {
      const float* weights_data = weights->data<float>();
      auto weights_dims = this->fwd_pd_->weights_desc().get_dims();

      auto user_md = dnnl::memory::desc(weights_dims,
                                        OneDNNGetDataType<float>(),
                                        dnnl::memory::format_tag::io);

      if (phi::funcs::is_int8<T_w>()) {
        dnnl::primitive_attr attrs;
        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_scales_mask(DNNL_ARG_SRC, mask);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->weights_desc(),
            to_void_cast<float>(weights_data),
            attrs,
            scale_data);
      } else {
        memory_p =
            this->AcquireMemoryWithReorder(user_md,
                                           this->fwd_pd_->weights_desc(),
                                           to_void_cast<float>(weights_data));
      }

      this->dev_ctx_.SetBlob(weights_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireCustomDstMemory(
      const ExecutionContext& ctx, phi::DenseTensor* out) {
    return this->template AcquireDstMemory<T_out>(out);
  }  // namespace operators

  void SetScalesIfNeeded(std::unordered_map<int, dnnl::memory>* args) {
    if (src_scales_mem_.get_desc().is_zero() != true) {
      args->insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_mem_});
      args->insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem_});
    }
    // dst scales may be empty when force fp32 output
    if (dst_scales_mem_.get(true)) {
      args->insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_mem_});
    }
  }
};  // namespace paddle

#define IF_CHANGE_FC_TW_TYPENAME(condition, ...) \
  if (condition) {                               \
    using T_w = int8_t;                          \
    __VA_ARGS__();                               \
  } else {                                       \
    using T_w = T_in;                            \
    __VA_ARGS__();                               \
  }

template <typename T_in>
class FCMKLDNNKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const ExecutionContext& ctx) const override {
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";

    IF_CHANGE_FC_TW_TYPENAME((std::is_same<T_in, uint8_t>::value), ([&] {
                               if (force_fp32_output) {
                                 this->RunKernel<float, T_w>(ctx);
                               } else if (phi::funcs::is_int8<T_in>()) {
                                 if (fuse_relu) {
                                   this->RunKernel<uint8_t, T_w>(ctx);
                                 } else {
                                   this->RunKernel<int8_t, T_w>(ctx);
                                 }
                               } else {
                                 this->RunKernel<T_in, T_w>(ctx);
                               }
                             }));
  }

  void PrepareSrcMem(const std::shared_ptr<dnnl::inner_product_forward>& fc_p
                         UNUSED,
                     const std::shared_ptr<dnnl::memory>& src_mem,
                     const phi::DenseTensor* x,
                     const dnnl::engine& engine) const {
    auto x_md = x->mem_desc().reshape(src_mem->get_desc().get_dims());
    if (x_md != src_mem->get_desc()) {
      dnnl::memory x_mem(x_md, engine, to_void_cast<T_in>(x->data<T_in>()));
      auto reorder_p = dnnl::reorder(x_mem, *src_mem);

      auto& astream = OneDNNContext::tls().get_stream();
      reorder_p.execute(astream, x_mem, *src_mem);
      astream.wait();
    } else {
      src_mem->set_data_handle(to_void_cast<T_in>(x->data<T_in>()));
    }
  }

  template <typename T_out, typename T_w>
  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<phi::DenseTensor>("Input");
    const auto* weights = ctx.Input<phi::DenseTensor>("W");
    const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    const auto& scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");

    std::shared_ptr<dnnl::inner_product_forward> fc_p;
    std::shared_ptr<dnnl::memory> src_memory_p;
    std::shared_ptr<dnnl::memory> weights_memory_p;
    std::shared_ptr<dnnl::memory> bias_memory_p;
    std::shared_ptr<dnnl::memory> dst_memory_p;

    std::string cache_key;
    cache_key.reserve(64);
    cache_key = phi::funcs::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx,
        phi::funcs::CreateKey(dev_ctx,
                              ctx.InputName("Input"),
                              ctx.InputName("W"),
                              phi::vectorize(x->dims()),
                              phi::vectorize(weights->dims())));

    auto inner_product_cache =
        std::static_pointer_cast<InnerProductCache>(dev_ctx.GetBlob(cache_key));

    RecomputeOutputDims(ctx, x, weights, out);

    std::unordered_map<int, dnnl::memory> fc_args;

    if (inner_product_cache) {
      fc_p = std::make_shared<dnnl::inner_product_forward>(
          inner_product_cache->inner_product_p);
      src_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->src_mem);
      PrepareSrcMem(fc_p, src_memory_p, x, onednn_engine);

      weights_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->weights_mem);

      dst_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->dst_mem);

      auto out_ptr = out->mutable_data<T_out>(
          ctx.GetPlace(), dst_memory_p->get_desc().get_size());
      dst_memory_p->set_data_handle(out_ptr);

      fc_args.insert({DNNL_ARG_SRC, *src_memory_p});
      fc_args.insert({DNNL_ARG_WEIGHTS, *weights_memory_p});
      fc_args.insert({DNNL_ARG_DST, *dst_memory_p});

      if (bias) {
        bias_memory_p =
            std::make_shared<dnnl::memory>(inner_product_cache->bias_mem);
        fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
      }

      if (inner_product_cache->src_scales_mem.get(true)) {
        fc_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                        inner_product_cache->src_scales_mem});
        fc_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                        inner_product_cache->wei_scales_mem});
      }
      if (inner_product_cache->dst_scales_mem.get(true)) {
        fc_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                        inner_product_cache->dst_scales_mem});
      }
    } else {
      auto in_col_dims = ctx.Attr<int>("in_num_col_dims");

      FCMKLDNNHandler<T_in, T_w, T_out> handler(ctx,
                                                dev_ctx,
                                                x,
                                                weights,
                                                bias,
                                                out,
                                                in_col_dims,
                                                onednn_engine,
                                                ctx.GetPlace());

      src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
      weights_memory_p =
          handler.AcquireWeightsMemoryWithReorder(weights, scale_weights);
      dst_memory_p = handler.AcquireCustomDstMemory(ctx, out);
      fc_args.insert({DNNL_ARG_SRC, *src_memory_p});
      fc_args.insert({DNNL_ARG_WEIGHTS, *weights_memory_p});
      fc_args.insert({DNNL_ARG_DST, *dst_memory_p});

      if (bias) {
        bias_memory_p = handler.AcquireBiasMemoryWithReorder(ctx, bias);
        fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
      }

      if (phi::funcs::is_int8<T_in>()) {
        handler.SetScalesIfNeeded(&fc_args);
      }

      fc_p = handler.AcquireForwardPrimitive();
    }

    auto& astream = OneDNNContext::tls().get_stream();
    fc_p->execute(astream, fc_args);
    astream.wait();

    if (!inner_product_cache) {
      auto ip_cache = std::make_shared<InnerProductCache>();
      ip_cache->inner_product_p = *fc_p;
      ip_cache->src_mem = *src_memory_p;
      ip_cache->weights_mem = *weights_memory_p;
      ip_cache->dst_mem = *dst_memory_p;
      if (bias) {
        ip_cache->bias_mem = *bias_memory_p;
      }
      if (fc_args.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC)) {
        ip_cache->src_scales_mem =
            fc_args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
        ip_cache->wei_scales_mem =
            fc_args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
      }

      if (fc_args.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)) {
        ip_cache->dst_scales_mem =
            fc_args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
      }

      dev_ctx.SetBlob(cache_key, ip_cache);
    }

    const auto out_md =
        dst_memory_p->get_desc().reshape(phi::vectorize(out->dims()));

    if (ctx.HasAttr("fused_reshape2_shape")) {
      phi::funcs::SetOutMemDescWithReshape2FuseSupport(
          ctx.Attr<std::vector<int>>("fused_reshape2_shape"), out, out_md);
    } else {
      out->set_mem_desc(out_md);
    }
  }

  void RecomputeOutputDims(const ExecutionContext& ctx,
                           const phi::DenseTensor* x,
                           const phi::DenseTensor* weights,
                           phi::DenseTensor* out) const {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights,
                      false,
                      phi::errors::PermissionDenied(
                          "Weight padding in fc can not be used in oneDNN."));
    std::vector<int64_t> output_dims;
    FCOutputSize(x->dims(),
                 weights->dims(),
                 output_dims,
                 in_num_col_dims,
                 padding_weights);
    out->Resize(phi::make_ddim(output_dims));
    out->set_lod(x->lod());
  }
};

}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(fc,
                   MKLDNN,
                   ::phi::CPUPlace,
                   ops::FCMKLDNNKernel<float>,
                   ops::FCMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::FCMKLDNNKernel<uint8_t>,
                   ops::FCMKLDNNKernel<int8_t>);
