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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/phi/kernels/onednn/conv_transpose_handler.h"

namespace phi {

#define PD_VISIT_FLOAT_AND_BF16_TYPES(TYPE, NAME, ...)                    \
  [&] {                                                                   \
    const auto& __dtype__ = TYPE;                                         \
    switch (__dtype__) {                                                  \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(NAME,                                          \
                           ::paddle::DataType::BFLOAT16,                  \
                           ::phi::dtype::bfloat16,                        \
                           __VA_ARGS__)                                   \
      default:                                                            \
        PD_THROW("function " #NAME " is not implemented for data type `", \
                 __dtype__,                                               \
                 "`");                                                    \
    }                                                                     \
  }()

template <typename T, typename T_out>
void ComputeFP32(const OneDNNContext& dev_ctx,
                 const DenseTensor* input,
                 const DenseTensor* filter,
                 const DenseTensor* bias,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 const std::string& padding_algorithm,
                 const std::vector<int>& dilations,
                 int groups,
                 const std::string& data_format,
                 bool is_test,
                 bool is_BFLOAT16,
                 const std::string& fuse_activation,
                 bool force_fp32_output,
                 DenseTensor* output) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  PD_VISIT_FLOAT_AND_BF16_TYPES(
      filter->dtype(), "ConvTransposeOneDNNHandlerT", ([&] {
        onednn::ConvTransposeOneDNNHandlerT<T, data_t, T_out> handler(
            dev_ctx,
            onednn_engine,
            dev_ctx.GetPlace(),
            input,
            filter,
            bias,
            strides,
            paddings,
            padding_algorithm,
            dilations,
            groups,
            data_format,
            is_test,
            is_BFLOAT16,
            fuse_activation,
            force_fp32_output,
            output);
        auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);
        // Caching Key for weights is needed
        std::string key =
            funcs::CreateKey(dev_ctx,
                             dev_ctx.GetInputsName("Input")[0],
                             dev_ctx.GetInputsName("Filter")[0],
                             (bias ? dev_ctx.GetInputsName("Bias")[0] : ""));
        key = funcs::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
        auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
            dev_ctx, key, filter, groups);

        std::shared_ptr<dnnl::memory> dst_memory_p =
            handler.template AcquireDstMemory<T_out>(output);
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
        output->set_mem_desc(dst_memory_p->get_desc());
      }));
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
    Execute<T, dtype::bfloat16>(dev_ctx, x, filter, groups, out);
  } else {
    Execute<T, float>(dev_ctx, x, filter, groups, out);
  }
}

template <typename T, typename T_out>
void Execute(const Context& dev_ctx,
             const DenseTensor& x,
             const DenseTensor& filter,
             int groups,
             DenseTensor* out) const {
  const auto* bias =
      dev_ctx.HasDnnInput("Bias") ? dev_ctx.GetDnnInput("Bias") : nullptr;
  auto* out = ctx.Output<phi::DenseTensor>("Output");
  ConvTransposeMKLDNNHandlerT<T, K, T_out> handler(
      dev_ctx.GetEngine(), x, filter, bias, out);
  auto src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
  // Caching Key for weights is needed
  std::string key = platform::CreateKey(dev_ctx,
                                        ctx.InputName("Input"),
                                        ctx.InputName("Filter"),
                                        (bias ? ctx.InputName("Bias") : ""));
  key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
  auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
      dev_ctx, key, filter, ctx.Attr<int>("groups"));

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
  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  conv_p->execute(astream, args);
  astream.wait();
  out->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose,
                   OneDNN,
                   ONEDNN,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::bfloat16) {}
