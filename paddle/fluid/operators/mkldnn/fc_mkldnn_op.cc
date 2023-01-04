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
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

using dnnl::inner_product_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;
using framework::DDim;
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
};
template <typename T_in, typename T_w, typename T_out>
class FCMKLDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T_in,
                                                 dnnl::inner_product_forward> {
 public:
  FCMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                  const OneDNNContext& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* weights,
                  const phi::DenseTensor* bias,
                  phi::DenseTensor* out,
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
                                            prop_kind::forward_inference,
                                            src_md,
                                            weights_md,
                                            bias_md,
                                            dst_md);
  }

 private:
  dnnl::primitive_attr CreateFCAttrs(const ExecutionContext& ctx) {
    dnnl::primitive_attr attributes;
    dnnl::post_ops post_operations;

    float sum_scale = 1.0f;
    float activation_scale = 1.0f;
    if (phi::funcs::is_int8<T_w>()) {
      std::vector<float> output_shift_scale;
      std::tie(output_shift_scale, sum_scale, activation_scale) =
          GetOutputScales(ctx);
      int mask = CreateMask(1, output_shift_scale.size() > 1);
      attributes.set_output_scales(mask, output_shift_scale);
    }

    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      post_operations.append_sum(sum_scale);
    }

    // ReLU from "fc_fuse_pass"
    if (ctx.Attr<std::string>("activation_type") == "relu") {
      post_operations.append_eltwise(
          activation_scale, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    AppendActivation(ctx, post_operations, activation_scale);

    if (ctx.HasAttr("fused_output_scale")) {
      float scale_alpha = ctx.Attr<float>("fused_output_scale");
      post_operations.append_eltwise(
          1.0, dnnl::algorithm::eltwise_linear, scale_alpha, 0.0f);
    }

    attributes.set_post_ops(post_operations);
    return attributes;
  }

  // Compute the bias scales so that its values correspond to the
  // scale of data being an output of weights and input multiplication
  std::vector<float> GetBiasScales(const framework::ExecutionContext& ctx) {
    if (ctx.HasAttr("Bias_scales")) {
      return ctx.Attr<std::vector<float>>("Bias_scales");
    } else {
      const float scale_in = ctx.Attr<float>("Scale_in");
      const auto& scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");
      std::vector<float> bias_scales(scale_weights.size());

      for (size_t i = 0; i < bias_scales.size(); ++i) {
        if (scale_weights[i] == 0.0)
          bias_scales[i] = 1.0f;
        else
          bias_scales[i] = scale_in * scale_weights[i];
      }
      return bias_scales;
    }
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

    if (fuse_activation == "hard_sigmoid") {
      post_ops.append_eltwise(activation_scale,
                              dnnl::algorithm::eltwise_linear,
                              fuse_alpha,
                              fuse_beta);
      post_ops.append_eltwise(
          activation_scale, dnnl::algorithm::eltwise_clip, 0.0f, 1.0f);
    } else {
      const std::unordered_map<std::string, dnnl::algorithm> activation_map = {
          {"abs", dnnl::algorithm::eltwise_abs},
          {"clip", dnnl::algorithm::eltwise_clip},
          {"gelu", dnnl::algorithm::eltwise_gelu_erf},
          {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
          {"gelu_tanh", dnnl::algorithm::eltwise_gelu_tanh},
          {"hard_swish", dnnl::algorithm::eltwise_hardswish},
          {"leaky_relu", dnnl::algorithm::eltwise_relu},
          {"mish", dnnl::algorithm::eltwise_mish},
          {"relu", dnnl::algorithm::eltwise_relu},
          {"relu6", dnnl::algorithm::eltwise_bounded_relu},
          {"sigmoid", dnnl::algorithm::eltwise_logistic},
          {"sqrt", dnnl::algorithm::eltwise_sqrt},
          {"swish", dnnl::algorithm::eltwise_swish},
          {"tanh", dnnl::algorithm::eltwise_tanh}};

      const auto& activation_type = activation_map.find(fuse_activation);

      PADDLE_ENFORCE_NE(
          activation_type,
          activation_map.end(),
          platform::errors::InvalidArgument(
              "Activation '%s' not found in oneDNN algorithms mapper",
              fuse_activation));

      post_ops.append_eltwise(
          activation_scale, activation_type->second, fuse_alpha, fuse_beta);
    }
  }

  // Correct output scale, to take into account scaling of input and weights
  // Since the data that comes out of input and weight multiplication is
  // scaled with its own scales, this data needs to be divided by
  // those scales to normalise them back to what their floating-point range
  // was. Then we multiply them by desired output scale we want on the output.
  std::tuple<std::vector<float>, float, float> GetOutputScales(
      const ExecutionContext& ctx) {
    if (ctx.HasAttr("Sum_scale")) {
      return std::make_tuple(ctx.Attr<std::vector<float>>("Output_shift_scale"),
                             ctx.Attr<float>("Sum_scale"),
                             ctx.Attr<float>("Activation_scale"));
    } else {
      auto scale_in_data = ctx.Attr<float>("Scale_in");
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      bool has_activation = !ctx.Attr<std::string>("activation_type").empty();
      bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
      bool fuse_residual_conn = ctx.HasAttr("fuse_residual_connection") &&
                                ctx.Attr<bool>("fuse_residual_connection");
      auto scale_in_eltwise_data = ctx.HasAttr("Scale_in_eltwise")
                                       ? ctx.Attr<float>("Scale_in_eltwise")
                                       : 1.0f;

      // If the output will be in floats, we don't multiply by scale_out.

      float activation_scale = (!force_fp32_output && has_activation)
                                   ? ctx.Attr<float>("Scale_out")
                                   : 1.0f;
      float scale_out_data = (force_fp32_output || has_activation)
                                 ? 1.0f
                                 : ctx.Attr<float>("Scale_out");
      float sum_scale =
          fuse_residual_conn ? scale_out_data / scale_in_eltwise_data : 1.0f;
      const size_t weight_scales_num = scale_weights_data.size();

      for (size_t i = 0; i < weight_scales_num; ++i) {
        if (scale_weights_data[i] == 0.0)
          scale_weights_data[i] = scale_out_data;
        else
          scale_weights_data[i] =
              scale_out_data / (scale_in_data * scale_weights_data[i]);
      }
      return std::make_tuple(scale_weights_data, sum_scale, activation_scale);
    }
  }

  // Computing MKL-DNN's scaling mask which determines along which dimension
  // slice should the scaling be applied. For more data plase refer to:
  // https://intel.github.io/mkl-dnn/group__c__api__attributes.html
  // Section dnnl_status_t DNNL_API dnnl_primitive_attr_set_output_scales
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorderAndAttrs(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const dnnl::primitive_attr& attrs) {
    std::shared_ptr<dnnl::memory> target_memory_p;

    auto user_memory_p =
        std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
    target_memory_p = std::make_shared<dnnl::memory>(target_md, this->engine_);
    auto reorder_p = std::make_shared<dnnl::reorder>(
        *user_memory_p, *target_memory_p, attrs);

    auto& astream = OneDNNContext::tls().get_stream();
    {
      platform::RecordEvent record_reorder(
          "int_reorder",
          platform::TracerEventType::UserDefined,
          1,
          platform::EventRole::kUniqueOp);
      reorder_p->execute(
          astream,
          {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
      astream.wait();
    }

    return target_memory_p;
  }

  std::string memory_key_;
  const OneDNNContext& dev_ctx_;

 public:
  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* x) {
    const T_in* x_data = x->data<T_in>();

    auto user_md = x->mem_desc();
    if (x->dims().size() != 2) {
      // reshape restrictions are always satisfied because in case of 3 or 4 dim
      // input, plain layout is enforced
      user_md = user_md.reshape(this->fwd_pd_->src_desc().dims());
    }

    return this->AcquireMemoryWithReorder(
        user_md, this->fwd_pd_->src_desc(), to_void_cast<T_in>(x_data));
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const framework::ExecutionContext& ctx, const phi::DenseTensor* bias) {
    const float* bias_data = bias->data<float>();

    if (phi::funcs::is_int8<T_w>() == false) {
      // for BF16/FP32 bias is 1D and has no scales, so reorder is not needed
      return this->AcquireMemoryFromPrimitive(this->fwd_pd_->bias_desc(),
                                              to_void_cast<float>(bias_data));
    } else {
      const std::string bias_key = this->memory_key_ + "@bias";
      auto memory_p = std::static_pointer_cast<dnnl::memory>(
          this->dev_ctx_.GetBlob(bias_key));

      if (!memory_p) {
        const auto& scale_data = GetBiasScales(ctx);
        dnnl::primitive_attr attrs;

        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_output_scales(mask, scale_data);

        auto user_md = dnnl::memory::desc({bias->dims()[0]},
                                          OneDNNGetDataType<float>(),
                                          dnnl::memory::format_tag::a);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->bias_desc(),
            to_void_cast<float>(bias_data),
            attrs);
        this->dev_ctx_.SetBlob(bias_key, memory_p);
      }
      return memory_p;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const phi::DenseTensor* weights, const std::vector<float>& scale_data) {
    const std::string weights_key = this->memory_key_ + "@weights";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(weights_key));

    if (!memory_p) {
      const float* weights_data = weights->data<float>();
      auto weights_dims = this->fwd_pd_->weights_desc().dims();

      auto user_md = dnnl::memory::desc(weights_dims,
                                        OneDNNGetDataType<float>(),
                                        dnnl::memory::format_tag::io);

      if (phi::funcs::is_int8<T_w>()) {
        dnnl::primitive_attr attrs;
        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_output_scales(mask, scale_data);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->weights_desc(),
            to_void_cast<float>(weights_data),
            attrs);
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
    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      auto* residual_param = ctx.Input<phi::DenseTensor>("ResidualData");

      PADDLE_ENFORCE_EQ(
          out->dims(),
          residual_param->dims(),
          platform::errors::InvalidArgument(
              "Output and elementwise parameter need to have the "
              "same dimension sizes, but got output's dimension = %d"
              " and residual param's dimension =%d .",
              out->dims().size(),
              residual_param->dims().size()));

      out->ShareDataWith(*residual_param);
    }
    return this->template AcquireDstMemory<T_out>(out);
  }  // namespace operators
};   // namespace paddle

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
  void Compute(const framework::ExecutionContext& ctx) const override {
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

  void PrepareSrcMem(const std::shared_ptr<inner_product_forward>& fc_p,
                     const std::shared_ptr<dnnl::memory>& src_mem,
                     const phi::DenseTensor* x,
                     const dnnl::engine& engine) const {
    auto x_md = x->mem_desc().reshape(src_mem->get_desc().dims());
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

  void SetOutMemDescWithUnsqueeze2FuseSupport(
      const framework::ExecutionContext& ctx,
      phi::DenseTensor* out,
      const dnnl::memory::desc& out_md) const {
    const std::vector<int>& fused_unsqueeze2_axes =
        ctx.Attr<std::vector<int>>("fused_unsqueeze2_axes");
    const std::vector<int64_t>& op_tz = out_md.dims();
    std::vector<int64_t> unsqueezed_op_tz(
        op_tz.size() + fused_unsqueeze2_axes.size(), 0);

    for (const auto& axis : fused_unsqueeze2_axes) {
      int positive_axis = axis < 0 ? unsqueezed_op_tz.size() + axis : axis;
      unsqueezed_op_tz[positive_axis] = 1;
    }

    int j = 0;
    for (size_t i = 0; i < unsqueezed_op_tz.size(); ++i) {
      if (unsqueezed_op_tz[i] == 0) {
        unsqueezed_op_tz[i] = op_tz[j++];
      }
    }
    out->set_mem_desc(out_md.reshape(unsqueezed_op_tz));
    out->Resize(phi::make_ddim(unsqueezed_op_tz));
  }

  void SetOutMemDescWithReshape2FuseSupport(
      const framework::ExecutionContext& ctx,
      phi::DenseTensor* out,
      const dnnl::memory::desc& out_md) const {
    std::vector<int64_t> fused_reshape2_shape(
        ctx.Attr<std::vector<int>>("fused_reshape2_shape").begin(),
        ctx.Attr<std::vector<int>>("fused_reshape2_shape").end());

    const int out_shape_numel = out->numel();
    const int new_shape_numel = std::accumulate(fused_reshape2_shape.begin(),
                                                fused_reshape2_shape.end(),
                                                1,
                                                std::multiplies<int64_t>());

    for (size_t i = 0; i < fused_reshape2_shape.size(); ++i) {
      if (fused_reshape2_shape[i] == -1) {
        fused_reshape2_shape[i] = -out_shape_numel / new_shape_numel;
        break;
      }
    }

    out->set_mem_desc(out_md.reshape(fused_reshape2_shape));
    out->Resize(phi::make_ddim(fused_reshape2_shape));
  }

  void SetOutMemDescWithLogicalLayoutFusesSupport(
      const framework::ExecutionContext& ctx,
      phi::DenseTensor* out,
      const dnnl::memory::desc& out_md) const {
    if (ctx.HasAttr("fused_unsqueeze2_axes")) {
      SetOutMemDescWithUnsqueeze2FuseSupport(ctx, out, out_md);
    } else if (ctx.HasAttr("fused_reshape2_shape")) {
      SetOutMemDescWithReshape2FuseSupport(ctx, out, out_md);
    } else if (ctx.HasAttr("fused_squeeze2_axes")) {
      out->set_mem_desc(out_md);
      out->Resize(phi::make_ddim(out_md.dims()));
    } else {
      out->set_mem_desc(out_md);
    }
  }

  template <typename T_out, typename T_w>
  void RunKernel(const framework::ExecutionContext& ctx) const {
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
                              phi::vectorize(x->dims())));

    auto inner_product_cache =
        std::static_pointer_cast<InnerProductCache>(dev_ctx.GetBlob(cache_key));

    RecomputeOutputDims(ctx, x, weights, out);

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
      if (ctx.HasAttr("fuse_residual_connection") &&
          ctx.Attr<bool>("fuse_residual_connection")) {
        auto* residual_param = ctx.Input<phi::DenseTensor>("ResidualData");
        out->ShareDataWith(*residual_param);
      }
      auto out_ptr = out->mutable_data<T_out>(
          ctx.GetPlace(), dst_memory_p->get_desc().get_size());
      dst_memory_p->set_data_handle(out_ptr);

      if (bias) {
        bias_memory_p =
            std::make_shared<dnnl::memory>(inner_product_cache->bias_mem);
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

      if (bias) {
        bias_memory_p = handler.AcquireBiasMemoryWithReorder(ctx, bias);
      }

      fc_p = handler.AcquireForwardPrimitive();
    }

    auto& astream = OneDNNContext::tls().get_stream();

    std::unordered_map<int, dnnl::memory> fc_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    if (bias) {
      fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }

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
      dev_ctx.SetBlob(cache_key, ip_cache);
    }

    SetOutMemDescWithLogicalLayoutFusesSupport(
        ctx,
        out,
        dst_memory_p->get_desc().reshape(phi::vectorize(out->dims())));
  }

  void RecomputeOutputDims(const ExecutionContext& ctx,
                           const phi::DenseTensor* x,
                           const phi::DenseTensor* weights,
                           phi::DenseTensor* out) const {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights,
                      false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
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
