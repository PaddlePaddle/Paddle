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

#include "paddle/phi/kernels/conv_transpose_kernel.h"

#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace phi {

inline dnnl::memory::dims GetWeightsTz(const phi::DenseTensor* filter,
                                       const int groups) {
  auto weights_tz = phi::vectorize(filter->dims());
  int g = std::max(groups, 1);
  int g_dim = (g > 1) ? 1 : 0;
  funcs::GetGroupConvWeightsTz(weights_tz, g);
  // gIOHW -> gOIHW || IOHW -> OIHW
  std::swap(weights_tz[g_dim + 0], weights_tz[g_dim + 1]);
  return weights_tz;
}

template <typename T, typename K, typename T_out>
class ConvTransposeOneDNNHandlerT
    : public funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward> {
 private:
  const bool is_test_;

 public:
  ConvTransposeOneDNNHandlerT(const OneDNNContext& dev_ctx,
                              const DenseTensor* x,
                              const DenseTensor* filter,
                              const DenseTensor* bias,
                              const std::vector<int>& strides_in,
                              const std::vector<int>& paddings_in,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations_in,
                              const std::string& data_format,
                              DenseTensor* out)
      : funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward>(
            dev_ctx.GetEngine(), dev_ctx.GetPlace()),
        is_test_(dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false) {
    PADDLE_ENFORCE_EQ(is_test_,
                      true,
                      phi::errors::InvalidArgument(
                          "ConvTransposeOneDNN works only for inference. "
                          "The attribute \'is_test\' value should be set to "
                          "True, but got is_test=False."));

    PADDLE_ENFORCE_EQ(
        x->layout(),
        DataLayout::ONEDNN,
        phi::errors::InvalidArgument("Got wrong layout = %d for Input tensor.",
                                     x->layout()));

    PADDLE_ENFORCE_EQ(
        filter->layout(),
        DataLayout::ONEDNN,
        phi::errors::InvalidArgument(
            "The filter tensor's layout should be %d, but got %d.",
            DataLayout::ONEDNN,
            filter->layout()));

    PADDLE_ENFORCE_EQ(
        x->dims().size(),
        4,
        phi::errors::InvalidArgument("Input must be with 4 dimensions, "
                                     "i.e. NCHW. but got dimension =%d",
                                     x->dims().size()));
    PADDLE_ENFORCE_EQ(
        filter->dims().size(),
        4,
        phi::errors::InvalidArgument("Filter must be with 4 dimensions, "
                                     "i.e. OIHW, but got dimension =%d",
                                     filter->dims().size()));

    if (bias) {
      PADDLE_ENFORCE_EQ(
          bias->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The bias tensor's laytout should be %d, but got %d.",
              DataLayout::ONEDNN,
              bias->layout()));

      PADDLE_ENFORCE_EQ(
          bias->dims().size(),
          1,
          phi::errors::InvalidArgument("Bias must only have 1 dimension, "
                                       "i.e. X, but got dimension = %d .",
                                       bias->dims().size()));
    }

    dnnl::memory::dims strides(begin(strides_in), end(strides_in));
    dnnl::memory::dims paddings(begin(paddings_in), end(paddings_in));
    dnnl::memory::dims dilations(begin(dilations_in), end(dilations_in));

    PADDLE_ENFORCE_EQ(
        strides.size(),
        2,
        phi::errors::Unimplemented(
            "Now we only support 2d oneDNN convolution transpose op"));

    const auto x_dims = x->dims();
    const auto x_data_dims = phi::slice_ddim(x_dims, 2, x_dims.size());
    const auto filter_dims = filter->dims();
    const auto filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());
    const auto ksize = phi::vectorize(filter_data_dims);
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, x_data_dims, strides, ksize);

    std::transform(
        dilations.begin(), dilations.end(), dilations.begin(), [](int64_t i) {
          return i - 1;
        });

    const auto src_tz = phi::vectorize(x->dims());
    const auto weights_tz = GetWeightsTz(filter, groups);
    const auto dst_tz = phi::vectorize(out->dims());
    const auto onednn_paddings = funcs::ToOneDNNPadding(paddings);

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    auto chosen_memory_format = funcs::OneDNNMemoryFormat::any;
    auto data_type = dnnl::memory::data_type::f32;
    const bool is_BFLOAT16 =
        dev_ctx.HasDnnAttr("mkldnn_data_type")
            ? PADDLE_GET_CONST(std::string,
                               dev_ctx.GetDnnAttr("mkldnn_data_type")) ==
                  "bfloat16"
            : false;
    if (is_BFLOAT16 || std::is_same<T_out, dtype::bfloat16>::value) {
      data_type = dnnl::memory::data_type::bf16;
    }

    dnnl::memory::desc src_md, weights_md;
    if (funcs::is_int8<T>()) {
      src_md = funcs::OneDNNMemDesc(
          src_tz, funcs::ToOneDNNDataType(x->dtype()), chosen_memory_format);
      weights_md = funcs::OneDNNMemDesc(
          weights_tz, dnnl::memory::data_type::s8, chosen_memory_format);
    } else {
      src_md = funcs::OneDNNMemDesc(src_tz, data_type, chosen_memory_format);
      weights_md = funcs::OneDNNMemDesc(
          weights_tz, data_type, funcs::OneDNNMemoryFormat::any);
    }

    const auto dst_md = funcs::OneDNNMemDesc(
        dst_tz, funcs::OneDNNGetDataType<T_out>(), chosen_memory_format);

    auto fwd_prop_kind = is_test_ ? dnnl::prop_kind::forward_inference
                                  : dnnl::prop_kind::forward_training;

    const std::string& fuse_activation =
        dev_ctx.HasDnnAttr("fuse_activation")
            ? PADDLE_GET_CONST(std::string,
                               dev_ctx.GetDnnAttr("fuse_activation"))
            : "";
    const bool force_fp32_output =
        dev_ctx.HasDnnAttr("force_fp32_output")
            ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
            : false;

    const dnnl::primitive_attr conv_attr = CreateConvAttrs(
        dev_ctx, filter, groups, force_fp32_output, false, fuse_activation);

    if (bias) {
      auto bias_tz = phi::vectorize(bias->dims());
      dnnl::memory::desc bias_md;
      if (funcs::is_int8<T>()) {
        bias_md = funcs::OneDNNMemDesc(bias_tz,
                                       dnnl::memory::data_type::s32,
                                       funcs::OneDNNMemoryFormat::x);
      } else {
        bias_md = funcs::OneDNNMemDesc(
            bias_tz, data_type, funcs::OneDNNMemoryFormat::x);
      }
      this->AcquireForwardPrimitiveDescriptor(
          fwd_prop_kind,
          dnnl::algorithm::deconvolution_direct,
          src_md,
          weights_md,
          bias_md,
          dst_md,
          strides,
          dilations,
          onednn_paddings[0],
          onednn_paddings[1]);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          fwd_prop_kind,
          dnnl::algorithm::deconvolution_direct,
          src_md,
          weights_md,
          dst_md,
          strides,
          dilations,
          onednn_paddings[0],
          onednn_paddings[1]);
    }
  }
  std::shared_ptr<std::tuple<float, std::vector<float>>> get_int8_bias_scales(
      const OneDNNContext& dev_ctx,
      const std::string& key,
      const DenseTensor* filter,
      int groups,
      const std::vector<float>& scale_weights_data) {
    // Get scales int8 bias key
    const std::string key_bs = key + "@bs";

    // Scales for int8 bias are to be cached to avoid
    // computing them each iteration
    groups = std::max(groups, 1);
    auto bias_scale_tuple =
        std::static_pointer_cast<std::tuple<float, std::vector<float>>>(
            dev_ctx.GetBlob(key_bs));
    if (bias_scale_tuple) return bias_scale_tuple;

    const auto& weights_tz = phi::vectorize(filter->dims());

    const auto& scale_in_data =
        dev_ctx.HasDnnAttr("Scale_in")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_in"))
            : 1.0f;

    bool is_multi_channel = scale_weights_data.size() > 1;
    int mask_reorder = is_multi_channel ? 1 << 0 : 1;

    int count = 1;
    if (is_multi_channel) {
      count *= weights_tz[0];
      if (groups > 1) {
        count *= weights_tz[1];
      }
    }

    bias_scale_tuple =
        std::make_shared<std::tuple<float, std::vector<float>>>(std::make_tuple(
            static_cast<float>(mask_reorder), std::vector<float>(count)));
    for (int i = 0; i < count; i++) {
      std::get<1>(*bias_scale_tuple)[i] = scale_in_data * scale_weights_data[i];
    }

    dev_ctx.SetBlob(key_bs, bias_scale_tuple);

    return bias_scale_tuple;
  }

  std::tuple<float, std::vector<float>, float> get_int8_scales(
      const OneDNNContext& dev_ctx,
      const DenseTensor* filter,
      int groups,
      bool force_fp32_output,
      bool fuse_residual_conn,
      const std::string& fuse_activation) const {
    const auto& weights_tz = phi::vectorize(filter->dims());
    groups = std::max(groups, 1);

    const auto& scale_weights_data =
        dev_ctx.HasDnnAttr("Scale_weights")
            ? PADDLE_GET_CONST(std::vector<float>,
                               dev_ctx.GetDnnAttr("Scale_weights"))
            : std::vector<float>{1.0f};
    const auto& scale_in_data =
        dev_ctx.HasDnnAttr("Scale_in")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_in"))
            : 1.0f;
    const auto& scale_in_eltwise_data =
        dev_ctx.HasDnnAttr("Scale_in_eltwise")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_in_eltwise"))
            : 1.0f;

    bool is_multi_channel = scale_weights_data.size() > 1;
    bool has_activation = !fuse_activation.empty();
    const auto& scale_out =
        dev_ctx.HasDnnAttr("Scale_out")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_out"))
            : 1.0f;
    float activation_scale =
        (!force_fp32_output && has_activation) ? scale_out : 1.0f;

    float scale_out_data =
        (force_fp32_output || has_activation) ? 1.0f : scale_out;
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

    return std::make_tuple(sum_scale, output_shift_scale, activation_scale);
  }

  dnnl::primitive_attr CreateConvAttrs(const OneDNNContext& dev_ctx,
                                       const DenseTensor* filter,
                                       int groups,
                                       bool force_fp32_output,
                                       bool fuse_residual_conn,
                                       const std::string& fuse_activation) {
    dnnl::primitive_attr conv_attr;
    dnnl::post_ops post_operations;

    float sum_scale = 1.0f;
    float activation_scale = 1.0f;
    std::vector<float> output_shift_scale;
    if (funcs::is_int8<T>()) {
      if (dev_ctx.HasDnnAttr("Sum_scale")) {
        sum_scale = PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Sum_scale"));
        activation_scale =
            dev_ctx.HasDnnAttr("Activation_scale")
                ? PADDLE_GET_CONST(float,
                                   dev_ctx.GetDnnAttr("Activation_scale"))
                : activation_scale;
        output_shift_scale =
            dev_ctx.HasDnnAttr("Output_shift_scale")
                ? PADDLE_GET_CONST(std::vector<float>,
                                   dev_ctx.GetDnnAttr("Output_shift_scale"))
                : output_shift_scale;
      } else {
        std::tie(sum_scale, output_shift_scale, activation_scale) =
            get_int8_scales(dev_ctx,
                            filter,
                            groups,
                            force_fp32_output,
                            fuse_residual_conn,
                            fuse_activation);
      }

      if (output_shift_scale.size() > 0) {
        int mask = output_shift_scale.size() > 1 ? 1 << 1 : 0;
        conv_attr.set_output_scales(mask, output_shift_scale);
      }
    }

    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }

    funcs::AppendActivation(dev_ctx, post_operations, activation_scale);

    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* x) {
    const T* input_data = x->data<T>();
    return funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward>::
        AcquireMemoryWithReorder(x->mem_desc(),
                                 this->fwd_pd_->src_desc(),
                                 funcs::to_void_cast<T>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const OneDNNContext& dev_ctx,
      const std::string& key,
      const phi::DenseTensor* filter,
      const int& groups,
      const std::vector<float>& scale_data = {1.0f}) {
    const K* filter_data = filter->data<K>();
    auto weights_tz = GetWeightsTz(filter, groups);
    int g = std::max(groups, 1);

    auto user_src_md =
        funcs::OneDNNMemDesc(weights_tz,
                             funcs::OneDNNGetDataType<K>(),
                             (g == 1) ? funcs::OneDNNMemoryFormat::iohw
                                      : funcs::OneDNNMemoryFormat::giohw);

    return this->template AcquireMemoryWithReorder<K>(
        dev_ctx,
        user_src_md,
        this->fwd_pd_->weights_desc(),
        funcs::to_void_cast<K>(filter_data),
        key,
        "@weights_mem_p",
        is_test_,
        scale_data);
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const OneDNNContext& dev_ctx,
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const std::string& key,
      const std::string& suffix,
      bool is_persistent = false,
      const std::vector<float>& scale_data = {1.0f},
      int mask = 0) {
    const auto target_key = key + suffix + "_target";
    const auto key_reorder_p = key + suffix + "reorder_p";
    const auto user_key = key + suffix + "_user";

    auto target_memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(target_key));

    if (target_memory_p == nullptr) {
      auto user_memory_p =
          std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
      if (user_md != target_md) {
        target_memory_p =
            std::make_shared<dnnl::memory>(target_md, this->engine_);
        dnnl::reorder::primitive_desc reorder_pdesc;
        if (funcs::is_int8<T>()) {
          dnnl::primitive_attr attr;
          attr.set_output_scales(mask, scale_data);
          reorder_pdesc = dnnl::reorder::primitive_desc(
              *user_memory_p, *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx.SetBlob(key_reorder_p, reorder_p);

        auto& astream = OneDNNContext::tls().get_stream();
        paddle::platform::RecordEvent record_reorder(
            "int_reorder",
            paddle::platform::TracerEventType::UserDefined,
            1,
            paddle::platform::EventRole::kUniqueOp);
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx.SetBlob(user_key, user_memory_p);
      dev_ctx.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = OneDNNContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      // TODO(jczaja): Here we detect if reorder is cached it means it is needed
      // need to change this to get rid of keys
      auto reorder_p = std::static_pointer_cast<dnnl::reorder>(
          dev_ctx.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        paddle::platform::RecordEvent record_reorder(
            "int_reorder",
            paddle::platform::TracerEventType::UserDefined,
            1,
            paddle::platform::EventRole::kUniqueOp);
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const OneDNNContext& dev_ctx,
      const std::string& key,
      const phi::DenseTensor* bias,
      const std::vector<float>& scale_data = {1.0f}) {
    const K* bias_data = bias->data<K>();
    auto user_bias_md = funcs::OneDNNMemDesc(phi::vectorize(bias->dims()),
                                             funcs::OneDNNGetDataType<K>(),
                                             funcs::OneDNNMemoryFormat::x);
    return this->AcquireMemoryWithReorder(dev_ctx,
                                          user_bias_md,
                                          this->fwd_pd_->bias_desc(),
                                          funcs::to_void_cast<K>(bias_data),
                                          key,
                                          "@bias_mem_p",
                                          is_test_,
                                          scale_data);
  }
};

static dnnl::memory::data_type GetDstType(bool is_int8,
                                          bool is_bfloat16,
                                          bool force_fp32_output,
                                          std::string fuse_activation) {
  auto dst_dt = dnnl::memory::data_type::f32;
  if (is_int8) {
    dst_dt = (fuse_activation == "relu" || fuse_activation == "relu6")
                 ? dnnl::memory::data_type::u8
                 : dnnl::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = dnnl::memory::data_type::f32;
    }
  } else {
    if (!force_fp32_output && is_bfloat16) {
      dst_dt = dnnl::memory::data_type::bf16;
    }
  }
  return dst_dt;
}

template <typename T, typename T_out>
void ComputeFP32(const OneDNNContext& dev_ctx,
                 const DenseTensor* x,
                 const DenseTensor* filter,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 const std::string& padding_algorithm,
                 int groups,
                 const std::vector<int>& dilations,
                 const std::string& data_format,
                 DenseTensor* out) {
  const auto* bias =
      dev_ctx.HasDnnInput("Bias") ? dev_ctx.GetDnnInput("Bias") : nullptr;

  ConvTransposeOneDNNHandlerT<T, float, T_out> handler(dev_ctx,
                                                       x,
                                                       filter,
                                                       bias,
                                                       strides,
                                                       paddings,
                                                       padding_algorithm,
                                                       groups,
                                                       dilations,
                                                       data_format,
                                                       out);

  auto src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
  // Caching Key for weights is needed
  std::string key =
      funcs::CreateKey(dev_ctx,
                       dev_ctx.GetInputsName("Input")[0],
                       dev_ctx.GetInputsName("Filter")[0],
                       (bias ? dev_ctx.GetInputsName("Bias")[0] : ""));
  key = funcs::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
  auto weights_memory_p =
      handler.AcquireWeightsMemoryWithReorder(dev_ctx, key, filter, groups);

  std::shared_ptr<dnnl::memory> dst_memory_p =
      handler.template AcquireDstMemory<T_out>(out);
  auto conv_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  if (bias) {
    auto bias_memory_p =
        handler.AcquireBiasMemoryWithReorder(dev_ctx, key, bias);
    args.insert({DNNL_ARG_BIAS, *bias_memory_p});
  }
  auto& astream = OneDNNContext::tls().get_stream();
  conv_p->execute(astream, args);
  astream.wait();
  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, typename T_out>
void ComputeINT8(const OneDNNContext& dev_ctx,
                 const DenseTensor* x,
                 const DenseTensor* filter,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 const std::string& padding_algorithm,
                 int groups,
                 const std::vector<int>& dilations,
                 const std::string& data_format,
                 DenseTensor* out) {
  const auto* bias =
      dev_ctx.HasDnnInput("Bias") ? dev_ctx.GetDnnInput("Bias") : nullptr;
  const std::string& fuse_activation =
      dev_ctx.HasDnnAttr("fuse_activation")
          ? PADDLE_GET_CONST(std::string, dev_ctx.GetDnnAttr("fuse_activation"))
          : "";
  bool unsigned_output =
      (fuse_activation == "relu" || fuse_activation == "relu6");
  bool need_s8_to_u8 = false;

  ConvTransposeOneDNNHandlerT<T, float, T_out> handler(dev_ctx,
                                                       x,
                                                       filter,
                                                       bias,
                                                       strides,
                                                       paddings,
                                                       padding_algorithm,
                                                       groups,
                                                       dilations,
                                                       data_format,
                                                       out);
  if (filter->dtype() == phi::DataType::INT8) {
    ConvTransposeOneDNNHandlerT<T, int8_t, T_out> handler(dev_ctx,
                                                          x,
                                                          filter,
                                                          bias,
                                                          strides,
                                                          paddings,
                                                          padding_algorithm,
                                                          groups,
                                                          dilations,
                                                          data_format,
                                                          out);
  }

  auto src_memory_p = handler.AcquireSrcMemoryWithReorder(x);

  const auto& scale_weights_data =
      dev_ctx.HasDnnAttr("Scale_weights")
          ? PADDLE_GET_CONST(std::vector<float>,
                             dev_ctx.GetDnnAttr("Scale_weights"))
          : std::vector<float>{1.0f};
  const bool is_multi_channel = scale_weights_data.size() > 1;
  int mask_reorder =
      is_multi_channel ? ((groups != 1) ? (1 << 1) + (1 << 0) : 1 << 0) : 0;

  std::string key =
      funcs::CreateKey(dev_ctx,
                       dev_ctx.GetInputsName("Input")[0],
                       dev_ctx.GetInputsName("Filter")[0],
                       (bias ? dev_ctx.GetInputsName("Bias")[0] : ""));
  key = funcs::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
  auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
      dev_ctx, key, filter, groups, scale_weights_data);

  auto dst_memory_p = handler.template AcquireDstMemory<T_out>(out);
  need_s8_to_u8 =
      (funcs::OneDNNGetDataType<T_out>() == dnnl::memory::data_type::s8) &&
      unsigned_output;

  auto conv_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  if (bias) {
    std::vector<float> bias_scales;
    auto p_scales_tuple =
        std::make_shared<std::tuple<float, std::vector<float>>>(
            std::make_tuple(static_cast<float>(mask_reorder), bias_scales));
    if (dev_ctx.HasDnnAttr("Bias_scales")) {
      bias_scales = PADDLE_GET_CONST(std::vector<float>,
                                     dev_ctx.GetDnnAttr("Bias_scales"));
      p_scales_tuple = std::make_shared<std::tuple<float, std::vector<float>>>(
          std::make_tuple(static_cast<float>(mask_reorder), bias_scales));
    } else {
      p_scales_tuple = handler.get_int8_bias_scales(
          dev_ctx, key, filter, groups, scale_weights_data);
    }
    auto bias_memory_p = handler.AcquireBiasMemoryWithReorder(
        dev_ctx, key, bias, std::get<1>(*p_scales_tuple));
    args.insert({DNNL_ARG_BIAS, *bias_memory_p});
  }

  auto& astream = OneDNNContext::tls().get_stream();
  conv_p->execute(astream, args);
  astream.wait();

  if (need_s8_to_u8) {
    dev_ctx.Alloc<uint8_t>(out);
  }

  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType(),
                    AllocationType::CPU,
                    phi::errors::PreconditionNotMet(
                        "Operator oneDNN Conv must use CPUPlace"));

  bool is_INT8 = funcs::is_int8<T>();
  const bool is_BFLOAT16 =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type")) ==
                "bfloat16"
          : false;
  const std::string& fuse_activation =
      dev_ctx.HasDnnAttr("fuse_activation")
          ? PADDLE_GET_CONST(std::string, dev_ctx.GetDnnAttr("fuse_activation"))
          : "";
  const bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;
  auto dst_dt =
      GetDstType(is_INT8, is_BFLOAT16, force_fp32_output, fuse_activation);

  if (!is_INT8) {
    if (dst_dt == dnnl::memory::data_type::f32) {
      ComputeFP32<T, float>(dev_ctx,
                            &x,
                            &filter,
                            strides,
                            paddings,
                            padding_algorithm,
                            groups,
                            dilations,
                            data_format,
                            out);
    } else if (dst_dt == dnnl::memory::data_type::bf16) {
      ComputeFP32<T, dtype::bfloat16>(dev_ctx,
                                      &x,
                                      &filter,
                                      strides,
                                      paddings,
                                      padding_algorithm,
                                      groups,
                                      dilations,
                                      data_format,
                                      out);
    }
  } else {
    if (dst_dt == dnnl::memory::data_type::f32) {
      ComputeINT8<T, float>(dev_ctx,
                            &x,
                            &filter,
                            strides,
                            paddings,
                            padding_algorithm,
                            groups,
                            dilations,
                            data_format,
                            out);
    } else if (dst_dt == dnnl::memory::data_type::u8) {
      ComputeINT8<T, uint8_t>(dev_ctx,
                              &x,
                              &filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out);
    } else if (dst_dt == dnnl::memory::data_type::s8) {
      ComputeINT8<T, int8_t>(dev_ctx,
                             &x,
                             &filter,
                             strides,
                             paddings,
                             padding_algorithm,
                             groups,
                             dilations,
                             data_format,
                             out);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose,
                   OneDNN,
                   ONEDNN,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {}
