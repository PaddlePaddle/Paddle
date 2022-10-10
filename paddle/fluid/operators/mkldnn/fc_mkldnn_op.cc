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

using dnnl::inner_product_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;
using framework::DataLayout;
using framework::DDim;
using framework::ExecutionContext;
using framework::LoDTensor;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::MKLDNNGetDataType;
using platform::to_void_cast;

template <typename T>
constexpr bool IsInt8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}

template <typename T_in, typename T_w, typename T_out>
class FCMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T_in,
                                               dnnl::inner_product_forward> {
 public:
  FCMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                  const platform::MKLDNNDeviceContext& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* weights,
                  const phi::DenseTensor* bias,
                  phi::DenseTensor* out,
                  const int in_num_col_dims,
                  dnnl::engine mkldnn_engine,
                  platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T_in, dnnl::inner_product_forward>(
            mkldnn_engine, cpu_place),
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
        {MB, IC}, MKLDNNGetDataType<T_in>(), dnnl::memory::format_tag::any);
    auto weights_md = dnnl::memory::desc(
        {OC, IC}, MKLDNNGetDataType<T_w>(), dnnl::memory::format_tag::any);
    auto dst_md = dnnl::memory::desc(
        {MB, OC}, MKLDNNGetDataType<T_out>(), dnnl::memory::format_tag::any);
    if (bias) {
      bias_md = dnnl::memory::desc({bias->numel()},
                                   MKLDNNGetDataType<float>(),
                                   dnnl::memory::format_tag::a);
    }

    dnnl::primitive_attr attrs;
    HandlePostOps(ctx, &attrs);

    this->AcquireForwardPrimitiveDescriptor(attrs,
                                            prop_kind::forward_inference,
                                            src_md,
                                            weights_md,
                                            bias_md,
                                            dst_md);
  }

 private:
  void HandlePostOps(const paddle::framework::ExecutionContext& ctx,
                     dnnl::primitive_attr* attrs) {
    static std::unordered_map<std::string, dnnl::algorithm> algo_map = {
        {"relu", dnnl::algorithm::eltwise_relu},
        {"gelu", dnnl::algorithm::eltwise_gelu},
        {"gelu_tanh", dnnl::algorithm::eltwise_gelu_tanh},
        {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
        {"tanh", dnnl::algorithm::eltwise_tanh},
        {"sigmoid", dnnl::algorithm::eltwise_logistic},
        {"hard_swish", dnnl::algorithm::eltwise_hardswish},
        {"mish", dnnl::algorithm::eltwise_mish}};

    std::vector<float> output_shift_scale;
    float scale = 1.0f;
    if (IsInt8<T_w>()) {
      std::tie(output_shift_scale, scale) = ComputeOutputShiftScale(ctx);
      int mask = CreateMask(1, output_shift_scale.size() > 1);
      attrs->set_output_scales(mask, output_shift_scale);
    }

    dnnl::post_ops post_ops;

    constexpr float sum_scale = 1.0f;
    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      post_ops.append_sum(sum_scale);
    }

    std::string activation_type = ctx.Attr<std::string>("activation_type");

    if (activation_type.empty() == false) {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;

      post_ops.append_eltwise(scale, algo_map[activation_type], alpha, beta);
    }

    attrs->set_post_ops(post_ops);
  }

  // Compute the bias scales so that its values correspond to the
  // scale of data being an output of weights and input multiplication
  std::vector<float> ComputeBiasScales(
      const float scale_in, const std::vector<float>& scale_weights) {
    std::vector<float> bias_scales(scale_weights.size());

    for (size_t i = 0; i < bias_scales.size(); ++i) {
      if (scale_weights[i] == 0.0)
        bias_scales[i] = 1.0f;
      else
        bias_scales[i] = scale_in * scale_weights[i];
    }

    return bias_scales;
  }

  // Correct output scale, to take into account scaling of input and weights
  // Since the data that comes out of input and weight multiplication is
  // scaled with its own scales, this data needs to be divided by
  // those scales to normalise them back to what their floating-point range
  // was. Then we multiply them by desired output scale we want on the output.
  std::tuple<std::vector<float>, float> ComputeOutputShiftScale(
      const ExecutionContext& ctx) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    bool has_activation = !ctx.Attr<std::string>("activation_type").empty();
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    // If the output will be in floats, we don't multiply by scale_out.

    float scale = (!force_fp32_output && has_activation)
                      ? ctx.Attr<float>("Scale_out")
                      : 1.0f;
    float inner_scale = (force_fp32_output || has_activation)
                            ? 1.0f
                            : ctx.Attr<float>("Scale_out");
    const size_t weight_scales_num = scale_weights_data.size();
    std::vector<float> output_shift_scale(weight_scales_num);

    for (size_t i = 0; i < weight_scales_num; i++) {
      if (scale_weights_data[i] == 0.0)
        output_shift_scale[i] = inner_scale;
      else
        output_shift_scale[i] =
            inner_scale / (scale_in_data * scale_weights_data[i]);
    }

    return make_tuple(output_shift_scale, scale);
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

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(
        astream,
        {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();

    return target_memory_p;
  }

  std::string memory_key_;
  const platform::MKLDNNDeviceContext& dev_ctx_;

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
      const phi::DenseTensor* bias,
      const float scale_in,
      const std::vector<float>& scale_weights) {
    const float* bias_data = bias->data<float>();

    if (IsInt8<T_w>() == false) {
      // for BF16/FP32 bias is 1D and has no scales, so reorder is not needed
      return this->AcquireMemoryFromPrimitive(this->fwd_pd_->bias_desc(),
                                              to_void_cast<float>(bias_data));
    } else {
      const std::string bias_key = this->memory_key_ + "@bias";
      auto memory_p = std::static_pointer_cast<dnnl::memory>(
          this->dev_ctx_.GetBlob(bias_key));

      if (!memory_p) {
        const auto& scale_data = ComputeBiasScales(scale_in, scale_weights);
        dnnl::primitive_attr attrs;

        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_output_scales(mask, scale_data);

        auto user_md = dnnl::memory::desc({bias->dims()[0]},
                                          MKLDNNGetDataType<float>(),
                                          dnnl::memory::format_tag::a);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->bias_desc(),
            to_void_cast<float>(bias_data),
            attrs);
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
                                        MKLDNNGetDataType<float>(),
                                        dnnl::memory::format_tag::io);

      if (IsInt8<T_w>()) {
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
      auto* residual_param = ctx.Output<phi::DenseTensor>("ResidualData");

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

template <typename T_in, typename T_w>
class FCMKLDNNKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";

    if (force_fp32_output) {
      this->RunKernel<float>(ctx);
    } else if (IsInt8<T_in>()) {
      if (fuse_relu) {
        this->RunKernel<uint8_t>(ctx);
      } else {
        this->RunKernel<int8_t>(ctx);
      }
    } else {
      this->RunKernel<T_in>(ctx);
    }
  }

  template <typename T_out = T_w>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<LoDTensor>("Input");
    const auto* weights = ctx.Input<phi::DenseTensor>("W");
    const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto out = ctx.Output<LoDTensor>("Out");

    auto in_col_dims = ctx.Attr<int>("in_num_col_dims");

    const float scale_in = ctx.Attr<float>("Scale_in");
    const auto& scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");

    RecomputeOutputDims(ctx, x, weights, out);

    FCMKLDNNHandler<T_in, T_w, T_out> handler(ctx,
                                              dev_ctx,
                                              x,
                                              weights,
                                              bias,
                                              out,
                                              in_col_dims,
                                              mkldnn_engine,
                                              ctx.GetPlace());

    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
    auto weights_memory_p =
        handler.AcquireWeightsMemoryWithReorder(weights, scale_weights);
    auto dst_memory_p = handler.AcquireCustomDstMemory(ctx, out);

    auto fc_p = handler.AcquireForwardPrimitive();
    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();

    std::unordered_map<int, dnnl::memory> fc_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    if (bias) {
      auto bias_memory_p =
          handler.AcquireBiasMemoryWithReorder(bias, scale_in, scale_weights);
      fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }

    fc_p->execute(astream, fc_args);
    astream.wait();

    out->set_mem_desc(
        dst_memory_p->get_desc().reshape(phi::vectorize(out->dims())));
  }

  void RecomputeOutputDims(const ExecutionContext& ctx,
                           const LoDTensor* x,
                           const phi::DenseTensor* weights,
                           LoDTensor* out) const {
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
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FP32,
                                    ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    fc,
    MKLDNN,
    ::paddle::platform::CPUPlace,
    BF16,
    ops::kFCMKLDNNFP32,
    ops::FCMKLDNNKernel<paddle::platform::bfloat16,
                        paddle::platform::bfloat16>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8,
                                    ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8,
                                    ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNKernel<int8_t, int8_t>);
