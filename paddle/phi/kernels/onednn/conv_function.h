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
  std::cout << "GetDstType" << std::endl;
  auto dst_dt = dnnl::memory::data_type::f32;
  if (is_int8) {
    std::cout << "GetDstType1" << std::endl;
    dst_dt = (fuse_activation == "relu" || fuse_activation == "relu6")
                 ? dnnl::memory::data_type::u8
                 : dnnl::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = dnnl::memory::data_type::f32;
    }
    if (fuse_residual_conn && residual_param) {
      std::cout << "GetDstType2" << std::endl;
      auto residual_dt = funcs::ToOneDNNDataType(residual_param->dtype());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  } else {
    std::cout << "GetDstType3" << std::endl;
    if (!force_fp32_output && is_bfloat16) {
      dst_dt = dnnl::memory::data_type::bf16;
      if (fuse_residual_conn && residual_param) {
        std::cout << "GetDstType4" << std::endl;
        dst_dt = funcs::ToOneDNNDataType(residual_param->dtype());
      }
    }
  }
  std::cout << "GetDstType5" << std::endl;
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
  std::cout << "ComputeFP32 " << std::endl;
  const auto& onednn_engine = dev_ctx.GetEngine();
  const bool is_conv3d = strides.size() == 3U;
  const std::string& unique_name =
      dev_ctx.GetInputsName("Input")[0] + dev_ctx.GetInputsName("Filter")[0];
  std::cout << "ComputeFP32 2" << std::endl;
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
        std::cout << "ComputeFP32 3" << std::endl;
        auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);
        std::cout << "ComputeFP32 4" << std::endl;
        auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
            filter, groups, is_conv3d, is_test);
        std::cout << "ComputeFP32 5" << std::endl;
        std::shared_ptr<dnnl::memory> dst_memory_p;
        if (fuse_residual_conn) {
          std::cout << "ComputeFP32 6" << std::endl;
          dst_memory_p =
              handler.AcquireDstMemoryWithResidual(output, residual_param);
          std::cout << "ComputeFP32 7" << std::endl;
        } else {
          std::cout << "ComputeFP32 8" << std::endl;
          dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
          std::cout << "ComputeFP32 9" << std::endl;
        }
        std::cout << "ComputeFP32 10" << std::endl;
        auto conv_p = handler.AcquireForwardPrimitive();
        std::cout << "ComputeFP32 11" << std::endl;
        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, *src_memory_p},
            {DNNL_ARG_WEIGHTS, *weights_memory_p},
            {DNNL_ARG_DST, *dst_memory_p}};

        if (bias) {
          std::cout << "ComputeFP32 12" << std::endl;
          auto bias_memory_p =
              handler.AcquireBiasMemoryWithReorder(bias, is_test);
          std::cout << "ComputeFP32 13" << std::endl;
          args.insert({DNNL_ARG_BIAS, *bias_memory_p});
        }

        auto& astream = OneDNNContext::tls().get_stream();
        std::cout << "ComputeFP32 14" << std::endl;
        conv_p->execute(astream, args);
        std::cout << "ComputeFP32 15" << std::endl;
        astream.wait();
        std::cout << "ComputeFP32 16" << std::endl;
        output->set_mem_desc(dst_memory_p->get_desc());
        std::cout << "ComputeFP32 17" << std::endl;
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
  std::cout << "ComputeINT8 " << std::endl;
  const auto& onednn_engine = dev_ctx.GetEngine();
  std::cout << "ComputeINT8 2" << std::endl;
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
  std::cout << "ComputeINT8 3" << std::endl;
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
        std::cout << "ComputeINT8 4" << std::endl;
        auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);
        std::cout << "ComputeINT8 5" << std::endl;
        const auto& scale_weights_data =
            dev_ctx.HasDnnAttr("Scale_weights")
                ? PADDLE_GET_CONST(std::vector<float>,
                                   dev_ctx.GetDnnAttr("Scale_weights"))
                : std::vector<float>{1.0f};
        const bool is_multi_channel = scale_weights_data.size() > 1;
        int mask_reorder = is_multi_channel
                               ? ((groups != 1) ? (1 << 1) + (1 << 0) : 1 << 0)
                               : 0;
        std::cout << "ComputeINT8 6" << std::endl;
        auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
            filter, groups, false, true, scale_weights_data, mask_reorder);
        std::cout << "ComputeINT8 7" << std::endl;
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
          std::cout << "ComputeINT8 8" << std::endl;
          dst_memory_p =
              handler.AcquireDstMemoryWithResidual(output, residual_param);
          std::cout << "ComputeINT8 9" << std::endl;
          need_s8_to_u8 = (funcs::OneDNNGetDataType<T_out>() ==
                           dnnl::memory::data_type::s8) &&
                          unsigned_output;
          std::cout << "ComputeINT8 10" << std::endl;
        } else {
          std::cout << "ComputeINT8 11" << std::endl;
          dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
          std::cout << "ComputeINT8 12" << std::endl;
        }

        std::cout << "ComputeINT8 13" << std::endl;
        auto conv_p = handler.AcquireForwardPrimitive();
        std::cout << "ComputeINT8 14" << std::endl;
        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, *src_memory_p},
            {DNNL_ARG_WEIGHTS, *weights_memory_p},
            {DNNL_ARG_DST, *dst_memory_p}};

        if (bias) {
          std::cout << "ComputeINT8 15" << std::endl;
          auto bias_memory_p = handler.AcquireBiasMemoryWithReorder(bias, true);
          std::cout << "ComputeINT8 16" << std::endl;
          args.insert({DNNL_ARG_BIAS, *bias_memory_p});
        }
        std::cout << "ComputeINT8 17" << std::endl;
        auto src_scales_memory = handler.AcquireScalesMemory(DNNL_ARG_SRC);
        std::cout << "ComputeINT8 18" << std::endl;
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *src_scales_memory});
        std::cout << "ComputeINT8 19" << std::endl;
        auto wei_scales_memory = handler.AcquireScalesMemory(DNNL_ARG_WEIGHTS);
        std::cout << "ComputeINT8 20" << std::endl;
        args.insert(
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *wei_scales_memory});

        if (!force_fp32_output) {
          std::cout << "ComputeINT8 21" << std::endl;
          auto dst_scales_memory = handler.AcquireScalesMemory(DNNL_ARG_DST);
          std::cout << "ComputeINT8 22" << std::endl;
          args.insert(
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, *dst_scales_memory});
        }

        auto& astream = OneDNNContext::tls().get_stream();
        std::cout << "ComputeINT8 23" << std::endl;
        conv_p->execute(astream, args);
        std::cout << "ComputeINT8 24" << std::endl;
        astream.wait();
        std::cout << "ComputeINT8 25" << std::endl;

        if (need_s8_to_u8) {
          dev_ctx.Alloc<uint8_t>(output);
        }
        std::cout << "ComputeINT8 26" << std::endl;
        output->set_mem_desc(dst_memory_p->get_desc());
        std::cout << "ComputeINT8 27" << std::endl;
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
  std::cout << "ConvOnednn" << std::endl;
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
  std::cout << "ConvOnednn2" << std::endl;
  if (!is_INT8) {
    if (dst_dt == dnnl::memory::data_type::f32) {
      std::cout << "ConvOnednn3" << std::endl;
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
      std::cout << "ConvOnednn4" << std::endl;
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
      std::cout << "ConvOnednn5" << std::endl;
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
      std::cout << "ConvOnednn6" << std::endl;
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
      std::cout << "ConvOnednn7" << std::endl;
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
  std::cout << "ConvOnednn8" << std::endl;
}

}  // namespace phi
