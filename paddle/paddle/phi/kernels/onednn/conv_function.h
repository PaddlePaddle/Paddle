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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/phi/kernels/onednn/conv_handler.h"

namespace phi {

static dnnl::memory::data_type GetDstType(
    bool is_int8,
    bool is_bfloat16,
    bool force_fp32_output,
    std::string fuse_activation,
    bool fuse_residual_conn,
    const phi::DenseTensor* residual_param) {
  auto dst_dt = dnnl::memory::data_type::f32;
  if (is_int8) {
    dst_dt = (fuse_activation == "relu" || fuse_activation == "relu6")
                 ? dnnl::memory::data_type::u8
                 : dnnl::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = dnnl::memory::data_type::f32;
    }
    if (fuse_residual_conn && residual_param) {
      auto residual_dt = funcs::ToOneDNNDataType(residual_param->dtype());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  } else {
    if (!force_fp32_output && is_bfloat16) {
      dst_dt = dnnl::memory::data_type::bf16;
      if (fuse_residual_conn && residual_param) {
        dst_dt = funcs::ToOneDNNDataType(residual_param->dtype());
      }
    }
  }
  return dst_dt;
}

#define PD_VISIT_FLOAT_AND_INT8_TYPES(TYPE, NAME, ...)                    \
  [&] {                                                                   \
    const auto& __dtype__ = TYPE;                                         \
    switch (__dtype__) {                                                  \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::INT8, int8_t, __VA_ARGS__)            \
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
                 const DenseTensor* residual_param,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 const std::string& padding_algorithm,
                 const std::vector<int>& dilations,
                 int groups,
                 const std::string& data_format,
                 bool is_test,
                 bool is_BFLOAT16,
                 const std::string& fuse_activation,
                 bool fuse_residual_conn,
                 bool force_fp32_output,
                 DenseTensor* output) {
  const auto& onednn_engine = dev_ctx.GetEngine();
  const bool is_conv3d = strides.size() == 3U;
  const std::string& unique_name =
      dev_ctx.GetInputsName("Input")[0] + dev_ctx.GetInputsName("Filter")[0];
  PD_VISIT_FLOAT_AND_INT8_TYPES(
      filter->dtype(), "ConvOneDNNHandlerT", ([&] {
        onednn::ConvOneDNNHandlerT<T, data_t, T_out> handler(dev_ctx,
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
                                                             fuse_residual_conn,
                                                             force_fp32_output,
                                                             output,
                                                             unique_name);
        auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);
        auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
            filter, groups, is_conv3d, is_test);
        std::shared_ptr<dnnl::memory> dst_memory_p;
        if (fuse_residual_conn) {
          dst_memory_p =
              handler.AcquireDstMemoryWithResidual(output, residual_param);
        } else {
          dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
        }

        auto conv_p = handler.AcquireForwardPrimitive();
        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, *src_memory_p},
            {DNNL_ARG_WEIGHTS, *weights_memory_p},
            {DNNL_ARG_DST, *dst_memory_p}};

        if (bias) {
          auto bias_memory_p =
              handler.AcquireBiasMemoryWithReorder(bias, is_test);
          args.insert({DNNL_ARG_BIAS, *bias_memory_p});
        }

        auto& astream = OneDNNContext::tls().get_stream();
        conv_p->execute(astream, args);
        astream.wait();
        output->set_mem_desc(dst_memory_p->get_desc());
      }));
}

template <typename T, typename T_out>
void ComputeINT8(const OneDNNContext& dev_ctx,
                 const DenseTensor* input,
                 const DenseTensor* filter,
                 const DenseTensor* bias,
                 const DenseTensor* residual_param,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 const std::string& padding_algorithm,
                 const std::vector<int>& dilations,
                 int groups,
                 const std::string& data_format,
                 bool is_test,
                 bool is_BFLOAT16,
                 const std::string& fuse_activation,
                 bool fuse_residual_conn,
                 bool force_fp32_output,
                 DenseTensor* output) {
  const auto& onednn_engine = dev_ctx.GetEngine();
  const bool is_conv3d = strides.size() == 3U;

  bool unsigned_output =
      (fuse_activation == "relu" || fuse_activation == "relu6");
  bool need_s8_to_u8 = false;

  PADDLE_ENFORCE_NE(
      is_conv3d,
      true,
      phi::errors::Unimplemented(
          "OneDNN int8 convolution does not support 3D inputs currently"));
  PADDLE_ENFORCE_EQ(
      fuse_residual_conn && force_fp32_output,
      false,
      phi::errors::Unimplemented(
          "residual fusion does not support force output with fp32"));
  const std::string& unique_name =
      dev_ctx.GetInputsName("Input")[0] + dev_ctx.GetInputsName("Filter")[0];
  PD_VISIT_FLOAT_AND_INT8_TYPES(
      filter->dtype(), "ConvOneDNNHandlerT", ([&] {
        onednn::ConvOneDNNHandlerT<T, data_t, T_out> handler(dev_ctx,
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
                                                             fuse_residual_conn,
                                                             force_fp32_output,
                                                             output,
                                                             unique_name);

        auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);

        const auto& scale_weights_data =
            dev_ctx.HasDnnAttr("Scale_weights")
                ? PADDLE_GET_CONST(std::vector<float>,
                                   dev_ctx.GetDnnAttr("Scale_weights"))
                : std::vector<float>{1.0f};
        const bool is_multi_channel = scale_weights_data.size() > 1;
        int mask_reorder = is_multi_channel
                               ? ((groups != 1) ? (1 << 1) + (1 << 0) : 1 << 0)
                               : 0;
        auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
            filter, groups, false, true, scale_weights_data, mask_reorder);

        std::shared_ptr<dnnl::memory> dst_memory_p;
        if (fuse_residual_conn) {
          PADDLE_ENFORCE_EQ(
              output->dims(),
              residual_param->dims(),
              phi::errors::InvalidArgument(
                  "Output and elementwise parameter need to have the "
                  "same dimension sizes, but got output's dimension = %d"
                  " and residual param's dimension =%d .",
                  output->dims().size(),
                  residual_param->dims().size()));
          dst_memory_p =
              handler.AcquireDstMemoryWithResidual(output, residual_param);
          need_s8_to_u8 = (funcs::OneDNNGetDataType<T_out>() ==
                           dnnl::memory::data_type::s8) &&
                          unsigned_output;
        } else {
          dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
        }

        auto conv_p = handler.AcquireForwardPrimitive();

        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, *src_memory_p},
            {DNNL_ARG_WEIGHTS, *weights_memory_p},
            {DNNL_ARG_DST, *dst_memory_p}};

        if (bias) {
          auto bias_memory_p = handler.AcquireBiasMemoryWithReorder(bias, true);
          args.insert({DNNL_ARG_BIAS, *bias_memory_p});
        }

        auto src_scales_memory = handler.AcquireScalesMemory(DNNL_ARG_SRC);
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *src_scales_memory});

        auto wei_scales_memory = handler.AcquireScalesMemory(DNNL_ARG_WEIGHTS);
        args.insert(
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *wei_scales_memory});

        if (!force_fp32_output) {
          auto dst_scales_memory = handler.AcquireScalesMemory(DNNL_ARG_DST);
          args.insert(
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, *dst_scales_memory});
        }

        auto& astream = OneDNNContext::tls().get_stream();
        conv_p->execute(astream, args);
        astream.wait();

        if (need_s8_to_u8) {
          dev_ctx.Alloc<uint8_t>(output);
        }

        output->set_mem_desc(dst_memory_p->get_desc());
      }));
}

template <typename T, typename Context>
void ConvOnednn(const Context& dev_ctx,
                const DenseTensor* input,
                const DenseTensor* filter,
                const DenseTensor* bias,
                const DenseTensor* residual_param,
                const std::vector<int>& strides,
                const std::vector<int>& paddings,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations,
                int groups,
                const std::string& data_format,
                bool is_test,
                bool is_bfloat16,
                const std::string& fuse_activation,
                bool fuse_residual_connection,
                bool force_fp32_output,
                DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      AllocationType::CPU,
      phi::errors::PreconditionNotMet("Operator DNNL Conv must use CPUPlace"));

  bool is_INT8 = funcs::is_int8<T>();

  auto dst_dt = GetDstType(is_INT8,
                           is_bfloat16,
                           force_fp32_output,
                           fuse_activation,
                           fuse_residual_connection,
                           residual_param);
  if (!is_INT8) {
    if (dst_dt == dnnl::memory::data_type::f32) {
      ComputeFP32<T, float>(dev_ctx,
                            input,
                            filter,
                            bias,
                            residual_param,
                            strides,
                            paddings,
                            padding_algorithm,
                            dilations,
                            groups,
                            data_format,
                            is_test,
                            is_bfloat16,
                            fuse_activation,
                            fuse_residual_connection,
                            force_fp32_output,
                            out);
    } else if (dst_dt == dnnl::memory::data_type::bf16) {
      ComputeFP32<T, dtype::bfloat16>(dev_ctx,
                                      input,
                                      filter,
                                      bias,
                                      residual_param,
                                      strides,
                                      paddings,
                                      padding_algorithm,
                                      dilations,
                                      groups,
                                      data_format,
                                      is_test,
                                      is_bfloat16,
                                      fuse_activation,
                                      fuse_residual_connection,
                                      force_fp32_output,
                                      out);
    }
  } else {
    if (dst_dt == dnnl::memory::data_type::f32) {
      ComputeINT8<T, float>(dev_ctx,
                            input,
                            filter,
                            bias,
                            residual_param,
                            strides,
                            paddings,
                            padding_algorithm,
                            dilations,
                            groups,
                            data_format,
                            is_test,
                            is_bfloat16,
                            fuse_activation,
                            fuse_residual_connection,
                            force_fp32_output,
                            out);
    } else if (dst_dt == dnnl::memory::data_type::u8) {
      ComputeINT8<T, uint8_t>(dev_ctx,
                              input,
                              filter,
                              bias,
                              residual_param,
                              strides,
                              paddings,
                              padding_algorithm,
                              dilations,
                              groups,
                              data_format,
                              is_test,
                              is_bfloat16,
                              fuse_activation,
                              fuse_residual_connection,
                              force_fp32_output,
                              out);
    } else if (dst_dt == dnnl::memory::data_type::s8) {
      ComputeINT8<T, int8_t>(dev_ctx,
                             input,
                             filter,
                             bias,
                             residual_param,
                             strides,
                             paddings,
                             padding_algorithm,
                             dilations,
                             groups,
                             data_format,
                             is_test,
                             is_bfloat16,
                             fuse_activation,
                             fuse_residual_connection,
                             force_fp32_output,
                             out);
    }
  }
}

}  // namespace phi
