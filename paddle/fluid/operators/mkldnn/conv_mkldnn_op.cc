/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tuple>

#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {
namespace {

inline MKLDNNMemoryFormat GetWeightsFormat(const MKLDNNMemoryFormat format,
                                           const int groups,
                                           const bool is_conv3d) {
  if (is_conv3d) {
    return (groups == 1) ? format : MKLDNNMemoryFormat::goidhw;
  } else {
    return (groups == 1) ? format : MKLDNNMemoryFormat::goihw;
  }
}

static mkldnn::memory::data_type GetDstType(bool is_int8, bool is_bfloat16,
                                            bool force_fp32_output,
                                            std::string fuse_activation,
                                            bool fuse_residual_conn,
                                            const Tensor* residual_param) {
  auto dst_dt = mkldnn::memory::data_type::f32;
  if (is_int8) {
    dst_dt = (fuse_activation == "relu" || fuse_activation == "relu6")
                 ? mkldnn::memory::data_type::u8
                 : mkldnn::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = mkldnn::memory::data_type::f32;
    }
    if (fuse_residual_conn && residual_param) {
      auto residual_dt = framework::ToMKLDNNDataType(residual_param->type());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  } else {
    if (!force_fp32_output && is_bfloat16) {
      dst_dt = mkldnn::memory::data_type::bf16;
      if (fuse_residual_conn && residual_param) {
        dst_dt = framework::ToMKLDNNDataType(residual_param->type());
      }
    }
  }
  return dst_dt;
}

template <typename T, typename K, typename T_out>
class ConvMKLDNNHandlerT
    : public platform::MKLDNNHandlerNoCachingT<
          T, mkldnn::convolution_forward, mkldnn::convolution_backward_data,
          mkldnn::convolution_backward_weights> {
 public:
  ConvMKLDNNHandlerT(const framework::ExecutionContext& ctx,
                     const mkldnn::engine mkldnn_engine, const Tensor* input,
                     const Tensor* filter, const Tensor* bias, Tensor* output)
      : platform::MKLDNNHandlerNoCachingT<T, mkldnn::convolution_forward,
                                          mkldnn::convolution_backward_data,
                                          mkldnn::convolution_backward_weights>(
            mkldnn_engine, ctx.GetPlace()),
        is_test_(ctx.Attr<bool>("is_test")) {
    PADDLE_ENFORCE_EQ(input->layout(), framework::DataLayout::kMKLDNN,
                      platform::errors::InvalidArgument(
                          "The input tensor's layout should be %d, but got %d.",
                          framework::DataLayout::kMKLDNN, input->layout()));
    PADDLE_ENFORCE_NE(
        input->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for Input tensor"));

    PADDLE_ENFORCE_EQ(
        filter->layout(), framework::DataLayout::kMKLDNN,
        platform::errors::InvalidArgument(
            "The Filter tensor's layout should be %d, but got %d.",
            framework::DataLayout::kMKLDNN, filter->layout()));
    PADDLE_ENFORCE_NE(filter->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Wrong format set for Filter tensor"));

    PADDLE_ENFORCE_GE(input->dims().size(), 4,
                      platform::errors::InvalidArgument(
                          "Input must be with 4 or 5 dimensions, i.e. NCHW or "
                          "NCDHW, but got dimension = %d .",
                          input->dims().size()));
    PADDLE_ENFORCE_LE(input->dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "Input must be with 4 or 5 dimensions, i.e. NCHW or "
                          "NCDHW, but got dimension = %d .",
                          input->dims().size()));

    PADDLE_ENFORCE_GE(filter->dims().size(), 4,
                      platform::errors::InvalidArgument(
                          "Filter must be with 4 or 5 dimensions, i.e. OIHW or "
                          "OIDHW, but got dimension = %d .",
                          filter->dims().size()));
    PADDLE_ENFORCE_LE(filter->dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "Filter must be with 4 or 5 dimensions, i.e. OIHW or "
                          "OIDHW, but got dimension = %d .",
                          filter->dims().size()));

    if (bias) {
      PADDLE_ENFORCE_EQ(
          bias->layout(), framework::DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The Bias tensor's layout should be %d, but got %d.",
              framework::DataLayout::kMKLDNN, bias->layout()));
      PADDLE_ENFORCE_NE(bias->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Got wrong format for Bias tensor."));

      PADDLE_ENFORCE_EQ(
          bias->dims().size(), 1,
          platform::errors::InvalidArgument("Bias must only have 1 dimension, "
                                            "i.e. X, but got dimension = %d .",
                                            bias->dims().size()));
    }

    const std::string fuse_activation =
        ctx.Attr<std::string>("fuse_activation");
    const float fuse_alpha = ctx.Attr<float>("fuse_alpha");
    const float fuse_beta = ctx.Attr<float>("fuse_beta");
    const bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    const int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    const auto input_dims = input->dims();
    const auto data_dims =
        framework::slice_ddim(input_dims, 2, input_dims.size());
    const auto filter_dims = filter->dims();
    const auto filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());

    const auto ksize = framework::vectorize(filter_data_dims);

    auto strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    auto paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    auto dilations_temp = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int64_t> dilations(begin(dilations_temp), end(dilations_temp));

    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             data_dims, strides, ksize);

    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                   [](int64_t i) { return i - 1; });

    const auto src_tz = framework::vectorize(input->dims());

    auto weights_tz = framework::vectorize(filter->dims());
    platform::GetGroupConvWeightsTz(weights_tz, groups);

    const auto dst_tz = framework::vectorize(output->dims());

    const mkldnn::memory::dims stride_dims = strides;
    const auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);
    const mkldnn::memory::dims dilations_dims = dilations;

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    auto chosen_memory_format = MKLDNNMemoryFormat::any;
    auto data_type = mkldnn::memory::data_type::f32;
    if (ctx.Attr<std::string>("mkldnn_data_type") == "bfloat16" ||
        std::is_same<T_out, platform::bfloat16>::value)
      data_type = mkldnn::memory::data_type::bf16;

    mkldnn::memory::desc src_md, weights_md;
    if (platform::is_int8<T>()) {
      src_md = platform::MKLDNNMemDesc(
          src_tz, framework::ToMKLDNNDataType(input->type()),
          chosen_memory_format);
      weights_md = platform::MKLDNNMemDesc(
          weights_tz, mkldnn::memory::data_type::s8, chosen_memory_format);
    } else {
      src_md = platform::MKLDNNMemDesc(src_tz, data_type, chosen_memory_format);
      weights_md = platform::MKLDNNMemDesc(weights_tz, data_type,
                                           MKLDNNMemoryFormat::any);
    }

    const auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);
    const auto fwd_prop_kind = is_test_ ? mkldnn::prop_kind::forward_inference
                                        : mkldnn::prop_kind::forward_training;
    float sum_scale = 1.0f;
    std::vector<float> output_shift_scale;
    if (platform::is_int8<T>())
      std::tie(sum_scale, output_shift_scale) = get_int8_scales(ctx);

    const mkldnn::primitive_attr conv_attr = CreatePostOps(
        fuse_activation, fuse_alpha, fuse_beta, fuse_residual_conn,
        output_shift_scale, sum_scale);  // for INT8 only!

    if (bias) {
      auto bias_tz = framework::vectorize(bias->dims());
      mkldnn::memory::desc bias_md;
      if (platform::is_int8<T>()) {
        bias_md = platform::MKLDNNMemDesc(
            bias_tz, mkldnn::memory::data_type::s32, MKLDNNMemoryFormat::x);
      } else {
        bias_md =
            platform::MKLDNNMemDesc(bias_tz, data_type, MKLDNNMemoryFormat::x);
      }

      this->AcquireForwardPrimitiveDescriptor(
          conv_attr, fwd_prop_kind, dnnl::algorithm::convolution_direct, src_md,
          weights_md, bias_md, dst_md, stride_dims, dilations_dims,
          mkldnn_paddings[0], mkldnn_paddings[1]);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          conv_attr, fwd_prop_kind, dnnl::algorithm::convolution_direct, src_md,
          weights_md, dst_md, stride_dims, dilations_dims, mkldnn_paddings[0],
          mkldnn_paddings[1]);
    }
  }

  ConvMKLDNNHandlerT(const framework::ExecutionContext& ctx,
                     const mkldnn::engine mkldnn_engine, const Tensor* in,
                     const Tensor* filter, const Tensor* bias,
                     const Tensor* out_grad, Tensor* filter_grad,
                     Tensor* in_x_grad)
      : platform::MKLDNNHandlerNoCachingT<T, mkldnn::convolution_forward,
                                          mkldnn::convolution_backward_data,
                                          mkldnn::convolution_backward_weights>(
            mkldnn_engine, ctx.GetPlace()),
        is_test_(false) {
    PADDLE_ENFORCE_EQ(in->layout(), framework::DataLayout::kMKLDNN,
                      platform::errors::InvalidArgument(
                          "The input tensor's layout should be %d, but got %d.",
                          framework::DataLayout::kMKLDNN, in->layout()));
    PADDLE_ENFORCE_NE(in->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Got wrong format for Input tensor."));

    PADDLE_ENFORCE_EQ(
        filter->layout(), framework::DataLayout::kMKLDNN,
        platform::errors::InvalidArgument(
            "The filter tensor's layout should be %d, but got %d.",
            framework::DataLayout::kMKLDNN, filter->layout()));
    PADDLE_ENFORCE_NE(filter->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Got wrong format for Filter tensor."));

    PADDLE_ENFORCE_EQ(
        out_grad->layout(), framework::DataLayout::kMKLDNN,
        platform::errors::InvalidArgument(
            "The output_grad tensor's layout should be %d, but got %d.",
            framework::DataLayout::kMKLDNN, out_grad->layout()));
    PADDLE_ENFORCE_NE(out_grad->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Wrong format set for output_grad tensor"));

    PADDLE_ENFORCE_EQ(
        is_test_, false,
        platform::errors::InvalidArgument(
            "is_test attribute should be set to False in training phase."));

    std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int64_t> dilations(begin(dilations_temp), end(dilations_temp));

    auto input_dims = in->dims();
    auto data_dims = framework::slice_ddim(input_dims, 2, input_dims.size());
    auto filter_dims = filter->dims();
    auto filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    auto ksize = framework::vectorize(filter_data_dims);

    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             data_dims, strides, ksize);

    auto src_tz = framework::vectorize(in->dims());
    auto weights_tz = framework::vectorize(filter->dims());

    int groups = ctx.Attr<int>("groups");
    int g = std::max(groups, 1);
    platform::GetGroupConvWeightsTz(weights_tz, g);
    auto dst_tz = framework::vectorize(out_grad->dims());

    /* create memory descriptor for conv backward without specified format
     * ('any') which lets a primitive (conv backward in this case) choose
     * the memory format preferred for best performance
     */
    const auto chosen_memory_format = MKLDNNMemoryFormat::any;
    const auto weights_format = MKLDNNMemoryFormat::any;

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    const auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);
    auto diff_src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    auto diff_weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                   [](int64_t i) { return i - 1; });
    const mkldnn::memory::dims dilations_dims = dilations;

    const mkldnn::memory::dims stride_dims = strides;
    // Recreating FWD PD. For training there are no post ops in convolution
    mkldnn::primitive_attr conv_attr;
    if (bias) {
      auto bias_tz = framework::vectorize(bias->dims());
      mkldnn::memory::desc bias_md;
      if (platform::is_int8<T>()) {
        bias_md = platform::MKLDNNMemDesc(
            bias_tz, mkldnn::memory::data_type::s32, MKLDNNMemoryFormat::x);
      } else {
        bias_md = platform::MKLDNNMemDesc(
            bias_tz, mkldnn::memory::data_type::f32, MKLDNNMemoryFormat::x);
      }

      this->AcquireForwardPrimitiveDescriptor(
          conv_attr, mkldnn::prop_kind::forward_training,
          dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md,
          dst_md, stride_dims, dilations_dims, mkldnn_paddings[0],
          mkldnn_paddings[1]);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          conv_attr, mkldnn::prop_kind::forward_training,
          dnnl::algorithm::convolution_direct, src_md, weights_md, dst_md,
          stride_dims, dilations_dims, mkldnn_paddings[0], mkldnn_paddings[1]);
    }

    this->AcquireBackwardPrimitiveDescriptor(
        mkldnn::algorithm::convolution_direct, diff_src_md, weights_md,
        diff_dst_md, strides, dilations_dims, mkldnn_paddings[0],
        mkldnn_paddings[1]);

    this->AcquireBackwardWeightsPrimitiveDescriptor(
        mkldnn::algorithm::convolution_direct, src_md, diff_weights_md,
        diff_dst_md, strides, dilations_dims, mkldnn_paddings[0],
        mkldnn_paddings[1]);
  }

  std::tuple<float, std::vector<float>> get_int8_scales(
      const framework::ExecutionContext& ctx) const {
    const auto* filter = ctx.Input<Tensor>("Filter");
    const auto& weights_tz = framework::vectorize(filter->dims());

    const bool& force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    const bool& fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    const int groups = std::max(ctx.Attr<int>("groups"), 1);

    const auto& scale_in_data = ctx.Attr<float>("Scale_in");
    const auto& scale_in_eltwise_data = ctx.Attr<float>("Scale_in_eltwise");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    bool is_multi_channel = scale_weights_data.size() > 1;
    auto scale_out_data =
        force_fp32_output ? 1.0f : ctx.Attr<float>("Scale_out");
    float sum_scale =
        fuse_residual_conn ? scale_out_data / scale_in_eltwise_data : 1.0f;
    int count =
        is_multi_channel
            ? (groups > 1 ? (weights_tz)[1] * (weights_tz)[0] : (weights_tz)[0])
            : 1;
    std::vector<float> output_shift_scale(count);

#pragma omp parallel for if (count > 50)
    for (int i = 0; i < count; i++) {
      if (scale_weights_data[i] == 0.0)
        // weights data will contain 0 in some models, then weights
        // scale couldn't be calculated
        output_shift_scale[i] = scale_out_data;
      else
        output_shift_scale[i] =
            static_cast<float>(static_cast<double>(scale_out_data) /
                               (static_cast<double>(scale_in_data) *
                                static_cast<double>(scale_weights_data[i])));
    }

    return std::make_tuple(sum_scale, output_shift_scale);
  }

  std::shared_ptr<std::tuple<float, std::vector<float>>> get_int8_bias_scales(
      const framework::ExecutionContext& ctx,
      const platform::MKLDNNDeviceContext& dev_ctx,
      const std::string& key) const {
    const auto* filter = ctx.Input<Tensor>("Filter");
    const auto& weights_tz = framework::vectorize(filter->dims());
    const int groups = std::max(ctx.Attr<int>("groups"), 1);

    // Get scales int8 bias key
    const std::string key_bs = key + "@bs";

    // Scales for int8 bias are to be cached to avoid
    // computing them each iteration
    auto bias_scale_tuple =
        std::static_pointer_cast<std::tuple<float, std::vector<float> > >(dev_ctx.GetBlob(key_bs));
    if (bias_scale_tuple)
      return bias_scale_tuple;

    const auto& scale_weights_data =
        ctx.Attr<std::vector<float>>("Scale_weights");
    const auto& scale_in_data = ctx.Attr<float>("Scale_in");

    bool is_multi_channel = scale_weights_data.size() > 1;
    int mask_reorder = is_multi_channel ? 1 << 0 : 1;
    const int count =
        is_multi_channel
            ? (groups > 1 ? (weights_tz)[1] * (weights_tz)[0] : (weights_tz)[0])
            : 1;

    bias_scale_tuple = std::make_shared<std::tuple<float, std::vector<float>>>(std::make_tuple(static_cast<float>(mask_reorder), std::vector<float>(count)));
    for (int i = 0; i < count; i++) {
      std::get<1>(*bias_scale_tuple)[i] = scale_in_data * scale_weights_data[i];
    }

    dev_ctx.SetBlob(key_bs, bias_scale_tuple);

    return bias_scale_tuple;
  }

  mkldnn::primitive_attr CreatePostOps(
      std::string fuse_activation, float fuse_alpha, float fuse_beta,
      bool fuse_residual_conn, const std::vector<float> output_shift_scale = {},
      float sum_scale = 1.0f) {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;
    if (output_shift_scale.size() > 0) {
      int mask = output_shift_scale.size() > 1 ? 1 << 1 : 0;
      conv_attr.set_output_scales(mask, output_shift_scale);
    }

    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }
    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_activation == "relu" || fuse_activation == "leaky_relu") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "relu6") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale,
                                     mkldnn::algorithm::eltwise_bounded_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_swish,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "hard_swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(
          scale, mkldnn::algorithm::eltwise_hardswish, fuse_alpha, fuse_beta);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<mkldnn::memory>
  AcquireWeightsMemoryWithReorderFromDataPrimitive(
      const platform::MKLDNNDeviceContext& dev_ctx, const std::string& key,
      const framework::Tensor* filter, const int groups, const bool is_conv3d) {
    const K* filter_data = filter->data<K>();
    auto weights_tz = framework::vectorize(filter->dims());
    platform::GetGroupConvWeightsTz(weights_tz, groups);

    auto user_src_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<K>(),
        GetWeightsFormat(filter->format(), groups, is_conv3d));

    return this->AcquireMemoryWithReorder(
        dev_ctx, user_src_md, this->bwd_pd_->weights_desc(),
        platform::to_void_cast<K>(filter_data), key, "@weights_mem_d_p", false);
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryWithReorder(
      const framework::Tensor* input) {
    return this->AcquireMemoryWithReorderPrimitive(input,
                                                   this->fwd_pd_->src_desc());
  }

  std::shared_ptr<mkldnn::memory>
  AcquireSrcMemoryWithReorderFromWeightsPrimitive(
      const framework::Tensor* input) {
    return this->AcquireMemoryWithReorderPrimitive(input,
                                                   this->bwd_w_pd_->src_desc());
  }

  std::shared_ptr<mkldnn::memory>
  AcquireDiffDstMemoryWithReorderFromWeightsPrimitive(
      const framework::Tensor* out_grad) {
    return this->AcquireMemoryWithReorderPrimitive(
        out_grad, this->bwd_w_pd_->diff_dst_desc());
  }

  std::shared_ptr<mkldnn::memory>
  AcquireDiffDstMemoryWithReorderMemoryFromDataPrimitive(
      const framework::Tensor* out_grad) {
    return this->AcquireMemoryWithReorderPrimitive(
        out_grad, this->bwd_pd_->diff_dst_desc());
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryWithReorderPrimitive(
      const framework::Tensor* in_mem, const mkldnn::memory::desc& mem_md) {
    const T* in_mem_data = in_mem->data<T>();

    auto user_mem_md = platform::MKLDNNMemDesc(
        framework::vectorize(in_mem->dims()), platform::MKLDNNGetDataType<T>(),
        in_mem->format());

    return platform::MKLDNNHandlerNoCachingT<
        T, mkldnn::convolution_forward, mkldnn::convolution_backward_data,
        mkldnn::convolution_backward_weights>::
        AcquireMemoryWithReorder(user_mem_md, mem_md,
                                 platform::to_void_cast<T>(in_mem_data),
                                 is_test_);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryWithReorder(
      const platform::MKLDNNDeviceContext& dev_ctx, const std::string& key,
      const framework::Tensor* filter, const int groups, const bool is_conv3d,
      const std::vector<float>& scale_data = {1.0f}, int mask = 0) {
    const K* filter_data = filter->data<K>();
    auto weights_tz = framework::vectorize(filter->dims());
    platform::GetGroupConvWeightsTz(weights_tz, groups);

    auto user_src_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<K>(),
        GetWeightsFormat(filter->format(), groups, is_conv3d));

    return this->AcquireMemoryWithReorder(
        dev_ctx, user_src_md, this->fwd_pd_->weights_desc(),
        platform::to_void_cast<K>(filter_data), key, "@weights_mem_p", is_test_,
        {}, scale_data, mask);
  }

  template <typename F = T>
  std::shared_ptr<mkldnn::memory> AcquireMemoryWithReorder(
      const platform::MKLDNNDeviceContext& dev_ctx,
      const mkldnn::memory::desc& user_md,
      const mkldnn::memory::desc& target_md, void* ptr, const std::string& key,
      const std::string& suffix, bool is_persistent = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {},
      const std::vector<float>& scale_data = {1.0f}, int mask = 0) {
    const auto target_key = key + suffix + "_target";
    const auto key_reorder_p = key + suffix + "reorder_p";
    const auto user_key = key + suffix + "_user";

    auto target_memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(target_key));

    if (target_memory_p == nullptr) {
      if (custom_reorder_func) {
        auto reordered_data =
            custom_reorder_func(reinterpret_cast<const F*>(ptr));
        dev_ctx.SetBlob(key_reorder_p + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }
      auto user_memory_p =
          std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
      if (user_md != target_md) {
        target_memory_p =
            std::make_shared<mkldnn::memory>(target_md, this->engine_);
        dnnl::reorder::primitive_desc reorder_pdesc;
        if (platform::is_int8<T>()) {
          dnnl::primitive_attr attr;
          attr.set_output_scales(mask, scale_data);
          reorder_pdesc = dnnl::reorder::primitive_desc(*user_memory_p,
                                                        *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx.SetBlob(key_reorder_p, reorder_p);

        auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                     {MKLDNN_ARG_TO, *target_memory_p}});
        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx.SetBlob(user_key, user_memory_p);
      dev_ctx.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      // TODO(jczaja): Here we detect if reorder is cached it means it is needed
      // need to change this to get rid of keys
      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                     {MKLDNN_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryWithReorder(
      const framework::ExecutionContext& ctx,
      const platform::MKLDNNDeviceContext& dev_ctx, const std::string& key,
      const framework::Tensor* bias) {
    const K* bias_data = bias->data<K>();
    auto user_bias_md = platform::MKLDNNMemDesc(
        framework::vectorize(bias->dims()), platform::MKLDNNGetDataType<K>(),
        MKLDNNMemoryFormat::x);
      
    // Get Bias scales for int8
//                                                                                        std::make_shared<std::tuple<float, std::vector<float>>>(std::make_tuple(static_cast<float>(mask_reorder), std::vector<float>(count)));
    auto p_tupple = platform::is_int8<T>() ?  (get_int8_bias_scales(ctx, dev_ctx, key)) : std::make_shared<std::tuple<float, std::vector<float>>>(std::make_tuple(0.0f, std::vector<float>(1,1.0f)));

    return this->AcquireMemoryWithReorder(
        dev_ctx, user_bias_md, this->fwd_pd_->bias_desc(),
        platform::to_void_cast<K>(bias_data), key, "@bias_mem_p", is_test_, {},
        std::get<1>(*p_tupple), std::get<0>(*p_tupple));
  }

  std::shared_ptr<mkldnn::memory> AcquireResidualMemory(
      const framework::Tensor* residual_param) {
    void* residual_data =
        residual_param->type() == framework::DataTypeTrait<T_out>::DataType()
            ? platform::to_void_cast<T_out>(residual_param->data<T_out>())
            : platform::to_void_cast<T>(residual_param->data<T>());
    auto user_residual_md = platform::MKLDNNMemDesc(
        framework::vectorize(residual_param->dims()),
        framework::ToMKLDNNDataType(residual_param->type()),
        residual_param->format());
    return this->AcquireMemoryFromPrimitive(user_residual_md, residual_data);
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryWithResidual(
      framework::Tensor* output, const framework::Tensor* residual_param) {
    std::shared_ptr<dnnl::memory> dst_memory_p;
    if (residual_param->format() !=
        platform::GetMKLDNNFormat(this->fwd_pd_->dst_desc())) {
      auto residual_memory_p = this->AcquireResidualMemory(residual_param);
      dst_memory_p = this->template AcquireDstMemory<T_out>(output);
      this->AcquireReorder(residual_memory_p, dst_memory_p);
    } else {
      // Changing ShareDataWith to TensorCopy results in performance drop
      // on ResNet architectures
      // (https://github.com/PaddlePaddle/Paddle/issues/22964)
      output->ShareDataWith(*residual_param);
      dst_memory_p = this->template AcquireDstMemory<T_out>(output);
    }
    return dst_memory_p;
  }

 private:
  const bool is_test_;
};

}  // anonymous namespace

template <typename T, typename K>
class ConvMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "Operator DNNL Conv must use CPUPlace"));
    bool is_INT8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    bool is_BFLOAT16 = ctx.Attr<std::string>("mkldnn_data_type") == "bfloat16";
    auto residual_param = ctx.Input<Tensor>("ResidualData");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    std::string fuse_activation = ctx.Attr<std::string>("fuse_activation");
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    auto dst_dt =
        GetDstType(is_INT8, is_BFLOAT16, force_fp32_output, fuse_activation,
                   fuse_residual_conn, residual_param);
    if (!is_INT8) {
      if (dst_dt == mkldnn::memory::data_type::f32) {
        ComputeFP32<float>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::bf16) {
        ComputeFP32<platform::bfloat16>(ctx);
      }
    } else {
      if (dst_dt == mkldnn::memory::data_type::f32) {
        ComputeINT8<float>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::u8) {
        ComputeINT8<uint8_t>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::s8) {
        ComputeINT8<int8_t>(ctx);
      }
    }
  }

  template <typename T_out>
  void ComputeFP32(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const bool is_conv3d = ctx.Attr<std::vector<int>>("strides").size() == 3U;
    const bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");

    const auto* input = ctx.Input<Tensor>("Input");
    const auto* filter = ctx.Input<Tensor>("Filter");
    const auto* bias =
        ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    ConvMKLDNNHandlerT<T, K, T_out> handler(ctx, mkldnn_engine, input, filter,
                                            bias, output);

    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);

    // Key is needed for params (weights and biases) to have reordered weights
    // cached
    std::string key = platform::CreateKey(dev_ctx, ctx.InputName("Input"),
                                          ctx.InputName("Filter"));
    key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

    auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
        dev_ctx, key, filter, ctx.Attr<int>("groups"), is_conv3d);
    std::shared_ptr<dnnl::memory> dst_memory_p;
    if (fuse_residual_conn) {
      auto* residual_param = ctx.Input<Tensor>("ResidualData");
      dst_memory_p =
          handler.AcquireDstMemoryWithResidual(output, residual_param);
    } else {
      dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
    }

    auto conv_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> args = {
        {MKLDNN_ARG_SRC, *src_memory_p},
        {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
        {MKLDNN_ARG_DST, *dst_memory_p}};

    if (bias) {
      auto bias_memory_p =
          handler.AcquireBiasMemoryWithReorder(ctx, dev_ctx, key, bias);
      args.insert({MKLDNN_ARG_BIAS, *bias_memory_p});
    }

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    conv_p->execute(astream, args);
    astream.wait();

    output->set_layout(framework::DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }

  template <typename T_out>
  void ComputeINT8(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const std::string& fuse_activation =
        ctx.Attr<std::string>("fuse_activation");
    const bool& fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    const bool& force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    const bool is_conv3d = ctx.Attr<std::vector<int>>("strides").size() == 3U;

    bool unsigned_output =
        (fuse_activation == "relu" || fuse_activation == "relu6");
    bool need_s8_to_u8 = false;

    PADDLE_ENFORCE_NE(
        is_conv3d, true,
        platform::errors::Unimplemented(
            "OneDNN int8 convolution does not support 3D inputs currently"));
    PADDLE_ENFORCE_EQ(
        fuse_residual_conn && force_fp32_output, false,
        platform::errors::Unimplemented(
            "residual fusion does not support force output with fp32"));

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    ConvMKLDNNHandlerT<T, K, T_out> handler(ctx, mkldnn_engine, input, filter,
                                            bias, output);

    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);

    const auto& scale_weights_data =
        ctx.Attr<std::vector<float>>("Scale_weights");
    const bool is_multi_channel = scale_weights_data.size() > 1;
    const int& groups = ctx.Attr<int>("groups");
    int mask_reorder =
        is_multi_channel ? ((groups != 1) ? (1 << 1) + (1 << 0) : 1 << 0) : 0;

    // Key is needed for params (weights and biases) to have reordered weights
    // cached
    std::string key = platform::CreateKey(dev_ctx, ctx.InputName("Input"),
                                          ctx.InputName("Filter"));
    key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

    auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
        dev_ctx, key, filter, groups, false, scale_weights_data, mask_reorder);

    std::shared_ptr<dnnl::memory> dst_memory_p;
    if (fuse_residual_conn) {
      auto* residual_param = ctx.Input<Tensor>("ResidualData");
      PADDLE_ENFORCE_EQ(
          output->dims(), residual_param->dims(),
          platform::errors::InvalidArgument(
              "Output and elementwise parameter need to have the "
              "same dimension sizes, but got output's dimension = %d"
              " and residual param's dimension =%d .",
              output->dims().size(), residual_param->dims().size()));
      dst_memory_p =
          handler.AcquireDstMemoryWithResidual(output, residual_param);
      need_s8_to_u8 = (platform::MKLDNNGetDataType<T_out>() ==
                       mkldnn::memory::data_type::s8) &&
                      unsigned_output;
    } else {
      dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
    }

    auto conv_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> args = {
        {MKLDNN_ARG_SRC, *src_memory_p},
        {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
        {MKLDNN_ARG_DST, *dst_memory_p}};

    if (bias) {

      auto bias_memory_p = handler.AcquireBiasMemoryWithReorder(
          ctx, dev_ctx, key, bias);
      args.insert({MKLDNN_ARG_BIAS, *bias_memory_p});
    }

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    conv_p->execute(astream, args);
    astream.wait();

    if (need_s8_to_u8) {
      output->mutable_data<uint8_t>(ctx.GetPlace());
    }

    output->set_layout(framework::DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }
};

template <typename T, typename K>
class ConvMKLDNNGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "Operator DNNL ConvGrad must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* bias =
        ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if (!input_grad && !filter_grad) return;

    // TODO(jczaja): Are all tensors really needed?
    ConvMKLDNNHandlerT<T, K, T> handler(ctx, dev_ctx.GetEngine(), input, filter,
                                        bias, output_grad, filter_grad,
                                        input_grad);

    // Key is needed for params (weights and biases) to have reordered weights
    // cached
    std::string key = platform::CreateKey(dev_ctx, ctx.InputName("Input"),
                                          ctx.InputName("Filter"));
    key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

    // create mkldnn memory from input tensors (data/weights)
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (filter_grad) {
      auto src_memory_p =
          handler.AcquireSrcMemoryWithReorderFromWeightsPrimitive(input);
      auto diff_dst_memory_p =
          handler.AcquireDiffDstMemoryWithReorderFromWeightsPrimitive(
              output_grad);

      // For convoluition with groups write filter grad into
      // oneDNN buffer and then we reorder it into filter_grad tensor
      int g = std::max(ctx.Attr<int>("groups"), 1);
      auto diff_weights_memory_p =
          g > 1 ? handler.AcquireDiffWeightsMemory()
                : handler.AcquireDiffWeightsMemory(filter_grad);

      auto conv_bwd_weights_p = handler.AcquireBackwardWeightsPrimitive();

      // TODO(grygielski) why no bias_diff?
      conv_bwd_weights_p->execute(
          astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                    {MKLDNN_ARG_DIFF_WEIGHTS, *diff_weights_memory_p}});
      astream.wait();

      filter_grad->set_layout(framework::DataLayout::kMKLDNN);
      // in OneDNN groups in convolution are treated as separate dimension
      // which is not the case in paddlepaddle
      auto filter_fmt = platform::GetMKLDNNFormat(*diff_weights_memory_p);

      // For convolution with groups convert from blocked to NCHW
      // otherwise there will be problems in next operators working on this data
      if (g > 1) {
        mkldnn::memory::data_type in_type =
            framework::ToMKLDNNDataType(filter->type());
        // for 3d conv with groups (six dimensional data reorder to goidhw)
        // for 2d conv with groups (five dimensional data reorder to goihw)
        // auto weights_tz = framework::vectorize(filter->dims());

        auto weights_tz = diff_weights_memory_p->get_desc().dims();
        mkldnn::memory::format_tag out_format =
            weights_tz.size() == 6 ? mkldnn::memory::format_tag::goidhw
                                   : mkldnn::memory::format_tag::goihw;
        platform::ReorderMKLDNNHandler handler(weights_tz, filter->type(),
                                               in_type, mkldnn_engine);
        auto reorder_dst_memory_p =
            handler.AcquireDstMemory(filter_grad, out_format, ctx.GetPlace());

        auto reorder_p =
            handler.AcquireReorder(reorder_dst_memory_p, diff_weights_memory_p);

        {
          platform::RecordEvent record_reorder("int_reorder",
                                               platform::EventRole::kUniqueOp);
          reorder_p->execute(astream, *diff_weights_memory_p,
                             *reorder_dst_memory_p);
          astream.wait();
        }

        // So here we have a data in goihw , which can be interpreted as OIHW
        // (OIDHW for conv3d)
        // because filter_grad shape is set for OIHW (OIDHW for conv3d)
        mkldnn::memory::format_tag target_format =
            weights_tz.size() == 6 ? mkldnn::memory::format_tag::oidhw
                                   : mkldnn::memory::format_tag::oihw;
        filter_grad->set_format(target_format);
      } else {
        filter_grad->set_format(filter_fmt);
      }
    }
    if (input_grad) {
      auto weights_memory_p =
          handler.AcquireWeightsMemoryWithReorderFromDataPrimitive(
              dev_ctx, key, filter, ctx.Attr<int>("groups"),
              ctx.Attr<std::vector<int>>("strides").size() == 3U);

      auto diff_dst_memory_p =
          handler.AcquireDiffDstMemoryWithReorderMemoryFromDataPrimitive(
              output_grad);
      auto diff_src_memory_p = handler.AcquireDiffSrcMemory(input_grad);

      auto conv_bwd_data_p = handler.AcquireBackwardPrimitive();

      conv_bwd_data_p->execute(astream,
                               {{MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                                {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
      astream.wait();

      input_grad->set_layout(framework::DataLayout::kMKLDNN);
      input_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory_p));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    conv2d, MKLDNN, ::paddle::platform::CPUPlace, BF16, ops::kConvMKLDNNFP32,
    ops::ConvMKLDNNOpKernel<paddle::platform::bfloat16, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, U8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, S8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float, float>);
