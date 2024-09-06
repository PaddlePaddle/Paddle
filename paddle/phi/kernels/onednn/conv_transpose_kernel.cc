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

#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace phi {

struct DeconvolutionCache {
  dnnl::deconvolution_forward deconvolution_forward;
  dnnl::memory src_mem;
  dnnl::memory weights_mem;
  dnnl::memory bias_mem;
  dnnl::memory dst_mem;
};

inline dnnl::memory::dims GetWeightsTz(const phi::DenseTensor* filter,
                                       const int groups) {
  auto weights_tz = common::vectorize(filter->dims());
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
                              DenseTensor* out)
      : funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward>(
            dev_ctx.GetEngine(), dev_ctx.GetPlace()),
        is_test_(dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false) {
    PADDLE_ENFORCE_EQ(is_test_,
                      true,
                      common::errors::InvalidArgument(
                          "ConvTransposeOneDNN works only for inference. "
                          "The attribute \'is_test\' value should be set to "
                          "True, but got is_test=False."));

    PADDLE_ENFORCE_EQ(
        x->layout(),
        DataLayout::ONEDNN,
        common::errors::InvalidArgument(
            "Got wrong layout = %d for Input tensor.", x->layout()));

    PADDLE_ENFORCE_EQ(
        filter->layout(),
        DataLayout::ONEDNN,
        common::errors::InvalidArgument(
            "The filter tensor's layout should be %d, but got %d.",
            DataLayout::ONEDNN,
            filter->layout()));

    PADDLE_ENFORCE_EQ(
        x->dims().size(),
        4,
        common::errors::InvalidArgument("Input must be with 4 dimensions, "
                                        "i.e. NCHW. but got dimension =%d",
                                        x->dims().size()));
    PADDLE_ENFORCE_EQ(
        filter->dims().size(),
        4,
        common::errors::InvalidArgument("Filter must be with 4 dimensions, "
                                        "i.e. OIHW, but got dimension =%d",
                                        filter->dims().size()));

    if (bias) {
      PADDLE_ENFORCE_EQ(
          bias->layout(),
          DataLayout::ONEDNN,
          common::errors::InvalidArgument(
              "The bias tensor's laytout should be %d, but got %d.",
              DataLayout::ONEDNN,
              bias->layout()));

      PADDLE_ENFORCE_EQ(
          bias->dims().size(),
          1,
          common::errors::InvalidArgument("Bias must only have 1 dimension, "
                                          "i.e. X, but got dimension = %d .",
                                          bias->dims().size()));
    }

    dnnl::memory::dims strides(begin(strides_in), end(strides_in));
    dnnl::memory::dims paddings(begin(paddings_in), end(paddings_in));
    dnnl::memory::dims dilations(begin(dilations_in), end(dilations_in));

    PADDLE_ENFORCE_EQ(
        strides.size(),
        2,
        common::errors::Unimplemented(
            "Now we only support 2d oneDNN convolution transpose op"));

    const auto x_dims = x->dims();
    const auto x_data_dims = common::slice_ddim(x_dims, 2, x_dims.size());
    const auto filter_dims = filter->dims();
    const auto filter_data_dims =
        common::slice_ddim(filter_dims, 2, filter_dims.size());
    const auto ksize = common::vectorize(filter_data_dims);
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, x_data_dims, strides, ksize);

    std::transform(
        dilations.begin(), dilations.end(), dilations.begin(), [](int64_t i) {
          return i - 1;
        });

    const auto src_tz = common::vectorize(x->dims());
    const auto weights_tz = GetWeightsTz(filter, groups);
    const auto dst_tz = common::vectorize(out->dims());
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

    const auto src_md =
        funcs::OneDNNMemDesc(src_tz, data_type, chosen_memory_format);
    const auto weights_md =
        funcs::OneDNNMemDesc(weights_tz, data_type, chosen_memory_format);
    const auto dst_md = funcs::OneDNNMemDesc(
        dst_tz, funcs::OneDNNGetDataType<T_out>(), chosen_memory_format);

    auto fwd_prop_kind = is_test_ ? dnnl::prop_kind::forward_inference
                                  : dnnl::prop_kind::forward_training;

    if (bias) {
      std::vector<int64_t> bias_tz = common::vectorize(bias->dims());
      const auto bias_md = funcs::OneDNNMemDesc(
          bias_tz, data_type, funcs::OneDNNMemoryFormat::x);
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
      const int& groups) {
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
        is_test_);
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
          attr.set_scales_mask(DNNL_ARG_DST, mask);
          reorder_pdesc = dnnl::reorder::primitive_desc(
              *user_memory_p, *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx.SetBlob(key_reorder_p, reorder_p);

        auto& astream = OneDNNContext::tls().get_stream();

        std::unordered_map<int, dnnl::memory> reorder_args;
        reorder_args.insert({DNNL_ARG_SRC, *user_memory_p});
        reorder_args.insert({DNNL_ARG_DST, *target_memory_p});
        if (funcs::is_int8<T>()) {
          auto scale_md =
              dnnl::memory::desc({static_cast<int64_t>(scale_data.size())},
                                 dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::x);
          auto scale_data_mem = dnnl::memory(scale_md, this->engine_);
          scale_data_mem.set_data_handle(
              phi::funcs::to_void_cast(scale_data.data()));
          reorder_args.insert(
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale_data_mem});
        }
        reorder_p->execute(astream, reorder_args);
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
      const phi::DenseTensor* bias) {
    const K* bias_data = bias->data<K>();
    auto user_bias_md = funcs::OneDNNMemDesc(common::vectorize(bias->dims()),
                                             funcs::OneDNNGetDataType<K>(),
                                             funcs::OneDNNMemoryFormat::x);
    return this->AcquireMemoryWithReorder(dev_ctx,
                                          user_bias_md,
                                          this->fwd_pd_->bias_desc(),
                                          funcs::to_void_cast<K>(bias_data),
                                          key,
                                          "@bias_mem_p",
                                          is_test_);
  }
};

template <typename T>
void PrepareSrcMem(const std::shared_ptr<dnnl::deconvolution_forward>& fc_p
                       UNUSED,
                   const std::shared_ptr<dnnl::memory>& src_mem,
                   const phi::DenseTensor* x,
                   const dnnl::engine& engine) {
  auto x_md = x->mem_desc().reshape(src_mem->get_desc().get_dims());
  if (x_md != src_mem->get_desc()) {
    dnnl::memory x_mem(x_md, engine, phi::funcs::to_void_cast<T>(x->data<T>()));
    auto reorder_p = dnnl::reorder(x_mem, *src_mem);

    auto& astream = OneDNNContext::tls().get_stream();
    reorder_p.execute(astream, x_mem, *src_mem);
    astream.wait();
  } else {
    src_mem->set_data_handle(phi::funcs::to_void_cast<T>(x->data<T>()));
  }
}

template <typename T, typename T_out>
void Execute(const OneDNNContext& dev_ctx,
             const DenseTensor* x,
             const DenseTensor* filter,
             const DenseTensor* bias,
             const std::vector<int>& strides,
             const std::vector<int>& paddings,
             const std::string& padding_algorithm,
             int groups,
             const std::vector<int>& dilations,
             DenseTensor* out) {
  std::shared_ptr<dnnl::deconvolution_forward> conv_p;
  std::shared_ptr<dnnl::memory> src_memory_p;
  std::shared_ptr<dnnl::memory> weights_memory_p;
  std::shared_ptr<dnnl::memory> bias_memory_p;
  std::shared_ptr<dnnl::memory> dst_memory_p;
  std::unordered_map<int, dnnl::memory> args;

  std::string cache_key = funcs::CreateKey(dev_ctx,
                                           dev_ctx.GetInputsName("Input")[0],
                                           dev_ctx.GetInputsName("Filter")[0],
                                           common::vectorize(x->dims()),
                                           common::vectorize(filter->dims()));
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto deconvolution_cache =
      std::static_pointer_cast<DeconvolutionCache>(dev_ctx.GetBlob(cache_key));
  if (deconvolution_cache) {
    conv_p = std::make_shared<dnnl::deconvolution_forward>(
        deconvolution_cache->deconvolution_forward);

    src_memory_p = std::make_shared<dnnl::memory>(deconvolution_cache->src_mem);
    PrepareSrcMem<T>(conv_p, src_memory_p, x, onednn_engine);

    weights_memory_p =
        std::make_shared<dnnl::memory>(deconvolution_cache->weights_mem);

    dst_memory_p = std::make_shared<dnnl::memory>(deconvolution_cache->dst_mem);
    auto out_ptr =
        dev_ctx.template Alloc<T_out>(out, dst_memory_p->get_desc().get_size());

    dst_memory_p->set_data_handle(out_ptr);

    args.insert({DNNL_ARG_SRC, *src_memory_p});
    args.insert({DNNL_ARG_WEIGHTS, *weights_memory_p});
    args.insert({DNNL_ARG_DST, *dst_memory_p});

    if (bias) {
      bias_memory_p =
          std::make_shared<dnnl::memory>(deconvolution_cache->bias_mem);
      args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }
  } else {
    // Check if bias obey the rules
    if (bias) {
      PADDLE_ENFORCE_EQ(
          bias->layout(),
          DataLayout::ONEDNN,
          common::errors::InvalidArgument(
              "The Bias tensor's layout should be %d, but got %d.",
              DataLayout::ONEDNN,
              bias->layout()));

      PADDLE_ENFORCE_EQ(
          bias->dims().size(),
          1,
          common::errors::InvalidArgument("Bias must only have 1 dimension, "
                                          "i.e. X, but got dimension = %d .",
                                          bias->dims().size()));
    }
    // Caching Key for weights is needed
    std::string key =
        funcs::CreateKey(dev_ctx,
                         dev_ctx.GetInputsName("Input")[0],
                         dev_ctx.GetInputsName("Filter")[0],
                         (bias ? dev_ctx.GetInputsName("Bias")[0] : ""));

    ConvTransposeOneDNNHandlerT<T, float, T_out> handler(dev_ctx,
                                                         x,
                                                         filter,
                                                         bias,
                                                         strides,
                                                         paddings,
                                                         padding_algorithm,
                                                         groups,
                                                         dilations,
                                                         out);

    src_memory_p = handler.AcquireSrcMemoryWithReorder(x);

    key = funcs::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
    weights_memory_p =
        handler.AcquireWeightsMemoryWithReorder(dev_ctx, key, filter, groups);

    dst_memory_p = handler.template AcquireDstMemory<T_out>(out);

    conv_p = handler.AcquireForwardPrimitive();

    args.insert({DNNL_ARG_SRC, *src_memory_p});
    args.insert({DNNL_ARG_WEIGHTS, *weights_memory_p});
    args.insert({DNNL_ARG_DST, *dst_memory_p});

    if (bias) {
      bias_memory_p = handler.AcquireBiasMemoryWithReorder(dev_ctx, key, bias);
      args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }
    auto cache = std::make_shared<DeconvolutionCache>();
    cache->deconvolution_forward = *conv_p;
    cache->src_mem = *src_memory_p;
    cache->weights_mem = *weights_memory_p;
    cache->dst_mem = *dst_memory_p;
    if (bias) {
      cache->bias_mem = *bias_memory_p;
    }

    dev_ctx.SetBlob(cache_key, cache);
  }
  auto& astream = OneDNNContext::tls().get_stream();
  conv_p->execute(astream, args);
  astream.wait();
  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding UNUSED,
                           const IntArray& output_size UNUSED,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format UNUSED,
                           DenseTensor* out) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType(),
                    AllocationType::CPU,
                    common::errors::PreconditionNotMet(
                        "Operator oneDNN Conv must use CPUPlace"));

  const bool is_BFLOAT16 =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type")) ==
                "bfloat16"
          : false;
  const bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;
  const bool use_bfloat16 = (!force_fp32_output && is_BFLOAT16);

  if (use_bfloat16) {
    Execute<T, dtype::bfloat16>(dev_ctx,
                                &x,
                                &filter,
                                nullptr,
                                strides,
                                paddings,
                                padding_algorithm,
                                groups,
                                dilations,
                                out);
  } else {
    Execute<T, float>(dev_ctx,
                      &x,
                      &filter,
                      nullptr,
                      strides,
                      paddings,
                      padding_algorithm,
                      groups,
                      dilations,
                      out);
  }
}

template <typename T, typename Context>
void Conv2dTransposeBiasKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const paddle::optional<DenseTensor>& bias,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding UNUSED,
                               const IntArray& output_size UNUSED,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format UNUSED,
                               DenseTensor* out) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType(),
                    AllocationType::CPU,
                    common::errors::PreconditionNotMet(
                        "Operator oneDNN Conv must use CPUPlace"));

  const bool is_BFLOAT16 =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type")) ==
                "bfloat16"
          : false;
  const bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;
  const bool use_bfloat16 = (!force_fp32_output && is_BFLOAT16);

  if (use_bfloat16) {
    Execute<T, dtype::bfloat16>(dev_ctx,
                                &x,
                                &filter,
                                bias.get_ptr(),
                                strides,
                                paddings,
                                padding_algorithm,
                                groups,
                                dilations,
                                out);
  } else {
    Execute<T, float>(dev_ctx,
                      &x,
                      &filter,
                      bias.get_ptr(),
                      strides,
                      paddings,
                      padding_algorithm,
                      groups,
                      dilations,
                      out);
  }
}

KernelKey ConvTransposeGetKernelTypeForVar(
    const GetKernelTypeForVarContext* ctx) {
  const std::string& var_name = ctx->GetVarName();
  const DenseTensor& tensor = ctx->GetTensor();
  const KernelKey& expected_kernel_type = ctx->GetKernelKey();
  const AttributeMap& attrs = ctx->GetAttrs();
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if ((var_name == "Input") &&
      (expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    auto it = attrs.find("data_format");
    const std::string data_format = PADDLE_GET_CONST(std::string, it->second);
    auto dl = common::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != phi::DataLayout::kAnyLayout) {
      return phi::KernelKey(tensor.place(), dl, expected_kernel_type.dtype());
    }
  }
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose,
                   OneDNN,
                   ONEDNN,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::bfloat16) {
  kernel->get_kerneltype_forvar_fn_ = phi::ConvTransposeGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(conv2d_transpose_bias,
                   OneDNN,
                   ONEDNN,
                   phi::Conv2dTransposeBiasKernel,
                   float,
                   phi::dtype::bfloat16) {
  kernel->get_kerneltype_forvar_fn_ = phi::ConvTransposeGetKernelTypeForVar;
}
