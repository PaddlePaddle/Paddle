// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/common/macros.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {
namespace onednn {

inline funcs::OneDNNMemoryFormat GetWeightsFormat(int groups, bool is_conv3d) {
  if (is_conv3d) {
    return (groups == 1) ? funcs::OneDNNMemoryFormat::oidhw
                         : funcs::OneDNNMemoryFormat::goidhw;
  } else {
    return (groups == 1) ? funcs::OneDNNMemoryFormat::oihw
                         : funcs::OneDNNMemoryFormat::goihw;
  }
}

template <typename T, typename K, typename T_out>
class ConvOneDNNHandlerT
    : public funcs::OneDNNHandlerT<T,
                                   dnnl::convolution_forward,
                                   dnnl::convolution_backward_data,
                                   dnnl::convolution_backward_weights> {
 public:
  ConvOneDNNHandlerT(const OneDNNContext& dev_ctx,
                     const dnnl::engine onednn_engine,
                     Place cpu_place,
                     const phi::DenseTensor* input,
                     const phi::DenseTensor* filter,
                     const phi::DenseTensor* bias,
                     const std::vector<int>& strides_in,
                     const std::vector<int>& paddings_in,
                     const std::string& padding_algorithm,
                     const std::vector<int>& dilations_in,
                     int groups,
                     const std::string& data_format UNUSED,
                     bool is_test,
                     bool is_BFLOAT16,
                     const std::string& fuse_activation,
                     bool fuse_residual_conn,
                     bool force_fp32_output,
                     phi::DenseTensor* output,
                     const std::string& unique_name)
      : funcs::OneDNNHandlerT<T,
                              dnnl::convolution_forward,
                              dnnl::convolution_backward_data,
                              dnnl::convolution_backward_weights>(
            dev_ctx,
            onednn_engine,
            cpu_place,
            funcs::CreateKey(
                dev_ctx, common::vectorize(input->dims()), unique_name)) {
    if (unlikely(!this->isCached())) {
      PADDLE_ENFORCE_EQ(
          input->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The input tensor's layout should be %d, but got %d.",
              DataLayout::ONEDNN,
              input->layout()));

      PADDLE_ENFORCE_EQ(
          filter->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The Filter tensor's layout should be %d, but got %d.",
              DataLayout::ONEDNN,
              filter->layout()));

      PADDLE_ENFORCE_GE(
          input->dims().size(),
          4,
          phi::errors::InvalidArgument(
              "Input must be with 4 or 5 dimensions, i.e. NCHW or "
              "NCDHW, but got dimension = %d .",
              input->dims().size()));
      PADDLE_ENFORCE_LE(
          input->dims().size(),
          5,
          phi::errors::InvalidArgument(
              "Input must be with 4 or 5 dimensions, i.e. NCHW or "
              "NCDHW, but got dimension = %d .",
              input->dims().size()));

      PADDLE_ENFORCE_GE(
          filter->dims().size(),
          4,
          phi::errors::InvalidArgument(
              "Filter must be with 4 or 5 dimensions, i.e. OIHW or "
              "OIDHW, but got dimension = %d .",
              filter->dims().size()));
      PADDLE_ENFORCE_LE(
          filter->dims().size(),
          5,
          phi::errors::InvalidArgument(
              "Filter must be with 4 or 5 dimensions, i.e. OIHW or "
              "OIDHW, but got dimension = %d .",
              filter->dims().size()));

      if (bias) {
        PADDLE_ENFORCE_EQ(
            bias->layout(),
            DataLayout::ONEDNN,
            phi::errors::InvalidArgument(
                "The Bias tensor's layout should be %d, but got %d.",
                DataLayout::ONEDNN,
                bias->layout()));

        PADDLE_ENFORCE_EQ(
            bias->dims().size(),
            1,
            phi::errors::InvalidArgument("Bias must only have 1 dimension, "
                                         "i.e. X, but got dimension = %d .",
                                         bias->dims().size()));
      }
      const auto input_dims = input->dims();
      const auto data_dims =
          common::slice_ddim(input_dims, 2, input_dims.size());
      const auto filter_dims = filter->dims();
      const auto filter_data_dims =
          common::slice_ddim(filter_dims, 2, filter_dims.size());
      const auto ksize = common::vectorize(filter_data_dims);
      std::vector<int64_t> strides(begin(strides_in), end(strides_in));
      std::vector<int64_t> paddings(begin(paddings_in), end(paddings_in));
      std::vector<int64_t> dilations(begin(dilations_in), end(dilations_in));
      UpdatePaddingAndDilation(
          &paddings, &dilations, padding_algorithm, data_dims, strides, ksize);
      std::transform(
          dilations.begin(), dilations.end(), dilations.begin(), [](int64_t i) {
            return i - 1;
          });

      const auto src_tz = common::vectorize(input->dims());

      auto weights_tz = common::vectorize(filter->dims());
      funcs::GetGroupConvWeightsTz(weights_tz, groups);

      const auto dst_tz = common::vectorize(output->dims());

      const dnnl::memory::dims stride_dims = strides;
      const auto onednn_paddings = funcs::ToOneDNNPadding(paddings);
      const dnnl::memory::dims dilations_dims = dilations;
      /* create memory descriptor for convolution without specified format
       * ('any') which lets a primitive (convolution in this case) choose
       * the memory format preferred for best performance
       */
      auto chosen_memory_format = funcs::OneDNNMemoryFormat::any;
      auto data_type = dnnl::memory::data_type::f32;
      if (is_BFLOAT16 || std::is_same<T_out, dtype::bfloat16>::value) {
        data_type = dnnl::memory::data_type::bf16;
      }

      dnnl::memory::desc src_md, weights_md;
      if (funcs::is_int8<T>()) {
        src_md = funcs::OneDNNMemDesc(src_tz,
                                      funcs::ToOneDNNDataType(input->dtype()),
                                      chosen_memory_format);
        weights_md = funcs::OneDNNMemDesc(
            weights_tz, dnnl::memory::data_type::s8, chosen_memory_format);
      } else {
        src_md = funcs::OneDNNMemDesc(src_tz, data_type, chosen_memory_format);
        weights_md = funcs::OneDNNMemDesc(
            weights_tz, data_type, funcs::OneDNNMemoryFormat::any);
      }
      if (input->dims().size() == 4 && input->dims()[1] <= 4) {
        chosen_memory_format = funcs::OneDNNMemoryFormat::nhwc;
      }
      const auto dst_md = funcs::OneDNNMemDesc(
          dst_tz, funcs::OneDNNGetDataType<T_out>(), chosen_memory_format);
      const auto fwd_prop_kind = dnnl::prop_kind::forward_inference;
      const dnnl::primitive_attr conv_attr = CreateConvAttrs(filter,
                                                             groups,
                                                             force_fp32_output,
                                                             fuse_residual_conn,
                                                             fuse_activation);

      if (bias) {
        auto bias_tz = common::vectorize(bias->dims());
        dnnl::memory::desc bias_md =
            funcs::OneDNNMemDesc(bias_tz,
                                 dnnl::memory::data_type::f32,
                                 funcs::OneDNNMemoryFormat::x);

        this->AcquireForwardPrimitiveDescriptor(
            conv_attr,
            fwd_prop_kind,
            dnnl::algorithm::convolution_direct,
            src_md,
            weights_md,
            bias_md,
            dst_md,
            stride_dims,
            dilations_dims,
            onednn_paddings[0],
            onednn_paddings[1]);
      } else {
        this->AcquireForwardPrimitiveDescriptor(
            conv_attr,
            fwd_prop_kind,
            dnnl::algorithm::convolution_direct,
            src_md,
            weights_md,
            dst_md,
            stride_dims,
            dilations_dims,
            onednn_paddings[0],
            onednn_paddings[1]);
      }
    }
  }

  ConvOneDNNHandlerT(const OneDNNContext& dev_ctx,
                     Place cpu_place,
                     const phi::DenseTensor* in,
                     const phi::DenseTensor* filter,
                     const phi::DenseTensor* bias,
                     const phi::DenseTensor* out_grad,
                     const std::vector<int>& strides_in,
                     const std::vector<int>& paddings_in,
                     const std::string& padding_algorithm,
                     const std::vector<int>& dilations_in,
                     int groups,
                     const std::string& data_format UNUSED,
                     bool is_test,
                     phi::DenseTensor* filter_grad UNUSED,
                     phi::DenseTensor* in_x_grad UNUSED,
                     const std::string& unique_name)
      : funcs::OneDNNHandlerT<T,
                              dnnl::convolution_forward,
                              dnnl::convolution_backward_data,
                              dnnl::convolution_backward_weights>(
            dev_ctx,
            dev_ctx.GetEngine(),
            cpu_place,
            funcs::CreateKey(
                dev_ctx, common::vectorize(in->dims()), unique_name)) {
    if (unlikely(!this->isBwdCached())) {
      PADDLE_ENFORCE_EQ(
          in->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The input tensor's layout should be %d, but got %d.",
              DataLayout::ONEDNN,
              in->layout()));

      PADDLE_ENFORCE_EQ(
          filter->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The filter tensor's layout should be %d, but got %d.",
              DataLayout::ONEDNN,
              filter->layout()));

      PADDLE_ENFORCE_EQ(
          out_grad->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The output_grad tensor's layout should be %d, but got %d.",
              DataLayout::ONEDNN,
              out_grad->layout()));

      PADDLE_ENFORCE_EQ(
          is_test,
          false,
          phi::errors::InvalidArgument(
              "is_test attribute should be set to False in training phase."));

      std::vector<int64_t> strides(begin(strides_in), end(strides_in));
      std::vector<int64_t> paddings(begin(paddings_in), end(paddings_in));
      std::vector<int64_t> dilations(begin(dilations_in), end(dilations_in));

      auto input_dims = in->dims();
      auto data_dims = common::slice_ddim(input_dims, 2, input_dims.size());
      auto filter_dims = filter->dims();
      auto filter_data_dims =
          common::slice_ddim(filter_dims, 2, filter_dims.size());
      auto ksize = common::vectorize(filter_data_dims);

      UpdatePaddingAndDilation(
          &paddings, &dilations, padding_algorithm, data_dims, strides, ksize);

      auto src_tz = common::vectorize(in->dims());
      auto weights_tz = common::vectorize(filter->dims());

      int g = std::max(groups, 1);
      funcs::GetGroupConvWeightsTz(weights_tz, g);
      auto dst_tz = common::vectorize(out_grad->dims());

      /* create memory descriptor for conv backward without specified format
       * ('any') which lets a primitive (conv backward in this case) choose
       * the memory format preferred for best performance
       */
      const auto chosen_memory_format = funcs::OneDNNMemoryFormat::any;
      const auto weights_format = funcs::OneDNNMemoryFormat::any;

      auto src_md = funcs::OneDNNMemDesc(
          src_tz, funcs::OneDNNGetDataType<T>(), chosen_memory_format);
      const auto dst_md = funcs::OneDNNMemDesc(
          dst_tz, funcs::OneDNNGetDataType<T_out>(), chosen_memory_format);
      auto diff_src_md = funcs::OneDNNMemDesc(
          src_tz, funcs::OneDNNGetDataType<T>(), chosen_memory_format);
      auto weights_md = funcs::OneDNNMemDesc(
          weights_tz, funcs::OneDNNGetDataType<T>(), weights_format);
      auto diff_weights_md = funcs::OneDNNMemDesc(
          weights_tz, funcs::OneDNNGetDataType<T>(), weights_format);
      auto diff_dst_md = funcs::OneDNNMemDesc(
          dst_tz, funcs::OneDNNGetDataType<T>(), chosen_memory_format);

      auto onednn_paddings = funcs::ToOneDNNPadding(paddings);
      std::transform(
          dilations.begin(), dilations.end(), dilations.begin(), [](int64_t i) {
            return i - 1;
          });
      const dnnl::memory::dims dilations_dims = dilations;

      const dnnl::memory::dims stride_dims = strides;
      // Recreating FWD PD. For training there are no post ops in convolution
      dnnl::primitive_attr conv_attr;
      if (bias) {
        auto bias_tz = common::vectorize(bias->dims());
        dnnl::memory::desc bias_md =
            funcs::OneDNNMemDesc(bias_tz,
                                 dnnl::memory::data_type::f32,
                                 funcs::OneDNNMemoryFormat::x);

        this->AcquireForwardPrimitiveDescriptor(
            conv_attr,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            src_md,
            weights_md,
            bias_md,
            dst_md,
            stride_dims,
            dilations_dims,
            onednn_paddings[0],
            onednn_paddings[1]);
      } else {
        this->AcquireForwardPrimitiveDescriptor(
            conv_attr,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            src_md,
            weights_md,
            dst_md,
            stride_dims,
            dilations_dims,
            onednn_paddings[0],
            onednn_paddings[1]);
      }

      this->AcquireBackwardPrimitiveDescriptor(
          dnnl::algorithm::convolution_direct,
          diff_src_md,
          weights_md,
          diff_dst_md,
          strides,
          dilations_dims,
          onednn_paddings[0],
          onednn_paddings[1]);

      this->AcquireBackwardWeightsPrimitiveDescriptor(
          dnnl::algorithm::convolution_direct,
          src_md,
          diff_weights_md,
          diff_dst_md,
          strides,
          dilations_dims,
          onednn_paddings[0],
          onednn_paddings[1]);
    }
  }

  dnnl::primitive_attr CreateConvAttrs(const DenseTensor* filter,
                                       int groups,
                                       bool force_fp32_output,
                                       bool fuse_residual_conn,
                                       const std::string& fuse_activation) {
    dnnl::primitive_attr conv_attr;
    dnnl::post_ops post_operations;

    float sum_scale = 1.0f;
    std::vector<float> output_shift_scale;
    if (funcs::is_int8<T>()) {
      conv_attr.set_scales_mask(DNNL_ARG_SRC, 0);

      auto wei_scales = ConvertToDNNLScales("Scale_weights");
      // By oneDNN API definition:
      // - For per-tensor quantization: the mask should be 0
      // - For per-dimension quantization: the mask should be 1 <<
      // dimension_index Here, wei_scales.size() != 1 means per-channel
      // quantization, the channel index in oneDNN is always 0, so we use mask =
      // 1 << 0. If the conv is group, the weights shape will be [g, oc/g, ic,
      // h, w], we need to do scaling along both group dim and oc dim, so the
      // mask = (1 << 0) + (1 << 1).
      int mask = wei_scales.size() == 1
                     ? 0
                     : (groups > 1 ? ((1 << 0) + (1 << 1)) : 1 << 0);
      conv_attr.set_scales_mask(DNNL_ARG_WEIGHTS, mask);

      if (!force_fp32_output) {
        conv_attr.set_scales_mask(DNNL_ARG_DST, 0);
      }

      auto psum_scales = ConvertToDNNLScales("Scale_in_eltwise");
      sum_scale = psum_scales[0];
    }

    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }

    funcs::AppendActivation(this->dev_ctx_, post_operations);

    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<dnnl::memory>
  AcquireWeightsMemoryWithReorderFromDataPrimitive(
      const phi::DenseTensor* filter, const int groups, const bool is_conv3d) {
    const K* filter_data = filter->data<K>();
    auto weights_tz = common::vectorize(filter->dims());
    funcs::GetGroupConvWeightsTz(weights_tz, groups);

    auto user_src_md =
        funcs::OneDNNMemDesc(weights_tz,
                             funcs::OneDNNGetDataType<K>(),
                             GetWeightsFormat(groups, is_conv3d));

    return this->AcquireMemoryWithReorder(user_src_md,
                                          this->bwd_pd_->weights_desc(),
                                          funcs::to_void_cast<K>(filter_data),
                                          "@weights_mem_d_p",
                                          false);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* input) {
    return this->AcquireMemoryWithReorderPrimitive(input,
                                                   "@src_mem_p_user",
                                                   "@src_mem_p_target",
                                                   "@src_mem_p",
                                                   this->fwd_pd_->src_desc());
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorderFromWeightsPrimitive(
      const phi::DenseTensor* input) {
    return this->AcquireMemoryWithReorderPrimitive(input,
                                                   "@src_mem_w_p_user",
                                                   "@src_mem_w_p_target",
                                                   "@src_mem_w_p",
                                                   this->bwd_w_pd_->src_desc());
  }

  std::shared_ptr<dnnl::memory>
  AcquireDiffDstMemoryWithReorderFromWeightsPrimitive(
      const phi::DenseTensor* out_grad) {
    return this->AcquireMemoryWithReorderPrimitive(
        out_grad,
        "@diff_dst_mem_w_p_user",
        "@diff_dst_mem_w_p_target",
        "@diff_dst_mem_w_p",
        this->bwd_w_pd_->diff_dst_desc());
  }

  std::shared_ptr<dnnl::memory>
  AcquireDiffDstMemoryWithReorderMemoryFromDataPrimitive(
      const phi::DenseTensor* out_grad) {
    return this->AcquireMemoryWithReorderPrimitive(
        out_grad,
        "@diff_dst_mem_p_user",
        "@diff_dst_mem_p_target",
        "@diff_dst_mem_p",
        this->bwd_pd_->diff_dst_desc());
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorderPrimitive(
      const phi::DenseTensor* in_mem,
      const char* key_mem_user,
      const char* key_mem_target,
      const char* key_mem,
      const dnnl::memory::desc& mem_md) {
    const T* in_mem_data = in_mem->data<T>();
    const std::string user_key_suffix{key_mem_user};
    auto user_mem_p = this->AcquireMemory(user_key_suffix);

    if (!user_mem_p) {
      return this->AcquireMemoryWithReorder(in_mem->mem_desc(),
                                            mem_md,
                                            funcs::to_void_cast<T>(in_mem_data),
                                            key_mem);
    } else {
      const std::string target_key_suffix{key_mem_target};
      const auto target_mem_p = this->AcquireMemory(target_key_suffix);
      user_mem_p->set_data_handle(funcs::to_void_cast<T>(in_mem_data));
      if (user_mem_p != target_mem_p) {
        this->AcquireReorder(user_mem_p, target_mem_p);
      }
      return target_mem_p;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const phi::DenseTensor* filter,
      const int groups,
      const bool is_conv3d,
      const bool is_test,
      const std::vector<float>& scale_data = {1.0f},
      int mask = 0) {
    // This is workaround to make execution faster, delete
    // if statement after including md inside Tensor
    auto weights_mem_p = this->AcquireMemory("@weights_mem_p_target");
    if (is_test && weights_mem_p) {
      return weights_mem_p;
    } else if (is_test) {
      const K* filter_data = filter->data<K>();
      auto weights_tz = common::vectorize(filter->dims());
      funcs::GetGroupConvWeightsTz(weights_tz, groups);

      auto user_src_md =
          funcs::OneDNNMemDesc(weights_tz,
                               funcs::OneDNNGetDataType<K>(),
                               GetWeightsFormat(groups, is_conv3d));

      return this->AcquireMemoryWithReorder(user_src_md,
                                            this->fwd_pd_->weights_desc(),
                                            funcs::to_void_cast<K>(filter_data),
                                            "@weights_mem_p",
                                            is_test,
                                            {},
                                            scale_data,
                                            mask);
    } else {
      const T* filter_data = filter->data<T>();
      auto weights_tz = common::vectorize(filter->dims());
      funcs::GetGroupConvWeightsTz(weights_tz, groups);

      auto user_src_md =
          funcs::OneDNNMemDesc(weights_tz,
                               funcs::OneDNNGetDataType<T>(),
                               GetWeightsFormat(groups, is_conv3d));

      return this->AcquireMemoryWithReorder(user_src_md,
                                            this->fwd_pd_->weights_desc(),
                                            funcs::to_void_cast<T>(filter_data),
                                            "@weights_mem_p",
                                            is_test,
                                            {},
                                            scale_data,
                                            mask);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const phi::DenseTensor* bias,
      const bool is_test,
      const std::vector<float>& scale_data = {1.0f},
      int mask = 0) {
    auto bias_mem_p = this->AcquireMemory("@bias_mem_p_target");
    if (is_test && bias_mem_p) {
      return bias_mem_p;
    } else {
      // if K is int8 (weights are int8) then biases are int32
      using K_Bias = typename std::
          conditional<std::is_same<K, int8_t>::value, int32_t, K>::type;
      if (std::is_same<K_Bias, int32_t>::value &&
          bias->dtype() != phi::DataType::INT32) {
        LOG(ERROR) << "Bias should be of type int32 but is " << bias->dtype();
      }
      const K_Bias* bias_data = bias->data<K_Bias>();

      return this->AcquireMemoryWithReorder(
          bias->mem_desc(),
          this->fwd_pd_->bias_desc(),
          funcs::to_void_cast<K_Bias>(bias_data),
          "@bias_mem_p",
          is_test,
          {},
          scale_data,
          mask);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireResidualMemory(
      const phi::DenseTensor* residual_param) {
    void* residual_data =
        residual_param->dtype() == phi::CppTypeToDataType<T_out>::Type()
            ? funcs::to_void_cast<T_out>(residual_param->data<T_out>())
            : funcs::to_void_cast<T>(residual_param->data<T>());
    auto residual_mem_p = this->AcquireMemory("@user_residual_data_mem_p");
    if (residual_mem_p) {
      residual_mem_p->set_data_handle(residual_data);
      return residual_mem_p;
    } else {
      return this->AcquireMemoryFromPrimitive(residual_param->mem_desc(),
                                              residual_data,
                                              "@user_residual_data_mem_p");
    }
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemoryWithResidual(
      phi::DenseTensor* output, const phi::DenseTensor* residual_param) {
    std::shared_ptr<dnnl::memory> dst_memory_p;
    auto residual_memory_p = this->AcquireResidualMemory(residual_param);
    dst_memory_p = this->template AcquireDstMemory<T_out>(output);
    this->AcquireReorder(residual_memory_p, dst_memory_p);
    return dst_memory_p;
  }

  // Currently, 4 kind of onednn scales are supported: src scales, weight
  // scales, post-sum scales and dst scales. This function is used to convert
  // paddle scales to onednn scales
  std::vector<float> ConvertToDNNLScales(const std::string& attr_name) {
    std::vector<float> paddle_scales;
    // weight scales is vector but other scales are scalar
    if (attr_name == "Scale_weights") {
      paddle_scales =
          this->dev_ctx_.HasDnnAttr(attr_name)
              ? PADDLE_GET_CONST(std::vector<float>,
                                 this->dev_ctx_.GetDnnAttr(attr_name))
              : std::vector<float>{1.0f};
    } else {
      float scale =
          this->dev_ctx_.HasDnnAttr(attr_name)
              ? PADDLE_GET_CONST(float, this->dev_ctx_.GetDnnAttr(attr_name))
              : 1.0f;
      paddle_scales = std::vector<float>{scale};
    }

    size_t count = paddle_scales.size();
    std::vector<float> dnnl_scales(count);
#pragma omp parallel for if (count > 50)
    for (size_t i = 0; i < count; i++) {
      dnnl_scales[i] = 1.f / paddle_scales[i];
    }
    return dnnl_scales;
  }

  std::shared_ptr<dnnl::memory> AcquireScalesMemory(int dnnl_arg) {
    // <dnnl_arg, {cache_key_suffix, attr_name}>
    std::unordered_map<int, std::pair<std::string, std::string>> map = {
        {DNNL_ARG_SRC, {"@src_scales", "Scale_in"}},
        {DNNL_ARG_WEIGHTS, {"@wei_scales", "Scale_weights"}},
        {DNNL_ARG_DST, {"@dst_scales", "Scale_out"}},
    };

    std::string cache_key_suffix, attr_name;
    std::tie(cache_key_suffix, attr_name) = map.at(dnnl_arg);

    // first look up the cache
    auto dnnl_scales_mem = this->AcquireMemory(cache_key_suffix);

    if (!dnnl_scales_mem) {
      // cache miss, so construct scales memory from the paddle scales
      // attributes
      auto dnnl_scales = ConvertToDNNLScales(attr_name);
      dnnl::memory::desc dnnl_scales_md(
          {static_cast<int64_t>(dnnl_scales.size())},
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::x);
      dnnl_scales_mem =
          std::make_shared<dnnl::memory>(dnnl_scales_md, this->engine_);
      memcpy(dnnl_scales_mem->get_data_handle(),
             dnnl_scales.data(),
             dnnl_scales.size() * sizeof(float));
      // cache the constructed memory
      this->CacheMemory(cache_key_suffix, dnnl_scales_mem);
    }

    return dnnl_scales_mem;
  }
};

}  // namespace onednn
}  // namespace phi
