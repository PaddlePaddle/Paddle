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

#include "paddle/phi/kernels/conv_grad_kernel.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/phi/kernels/onednn/conv_handler.h"

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

template <typename T, typename Context>
void ConvGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter,
                    const DenseTensor& out_grad,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    const std::string& padding_algorithm,
                    const std::vector<int>& dilations,
                    int groups,
                    const std::string& data_format,
                    DenseTensor* input_grad,
                    DenseTensor* filter_grad) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType(),
                    AllocationType::CPU,
                    phi::errors::PreconditionNotMet(
                        "Operator oneDNN ConvGrad must use CPUPlace"));
  const auto& onednn_engine = dev_ctx.GetEngine();

  const auto* bias =
      dev_ctx.HasDnnInput("Bias") ? dev_ctx.GetDnnInput("Bias") : nullptr;
  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;

  if (!input_grad && !filter_grad) return;

  const std::string& unique_name =
      dev_ctx.GetInputsName("Input")[0] + dev_ctx.GetInputsName("Filter")[0];

  PD_VISIT_FLOAT_AND_BF16_TYPES(
      filter.dtype(), "ConvOneDNNHandlerT", ([&] {
        // TODO(jczaja): Are all tensors really needed?
        onednn::ConvOneDNNHandlerT<T, data_t, T> handler(dev_ctx,
                                                         dev_ctx.GetPlace(),
                                                         &input,
                                                         &filter,
                                                         bias,
                                                         &out_grad,
                                                         strides,
                                                         paddings,
                                                         padding_algorithm,
                                                         dilations,
                                                         groups,
                                                         data_format,
                                                         is_test,
                                                         filter_grad,
                                                         input_grad,
                                                         unique_name);

        // create mkldnn memory from input tensors (data/weights)
        auto& astream = OneDNNContext::tls().get_stream();

        if (filter_grad) {
          auto src_memory_p =
              handler.AcquireSrcMemoryWithReorderFromWeightsPrimitive(&input);
          auto diff_dst_memory_p =
              handler.AcquireDiffDstMemoryWithReorderFromWeightsPrimitive(
                  &out_grad);

          // For convoluition with groups write filter grad into
          // oneDNN buffer and then we reorder it into filter_grad tensor
          int g = std::max(groups, 1);
          auto diff_weights_memory_p =
              g > 1 ? handler.AcquireDiffWeightsMemory()
                    : handler.AcquireDiffWeightsMemory(filter_grad);

          auto conv_bwd_weights_p = handler.AcquireBackwardWeightsPrimitive();

          conv_bwd_weights_p->execute(
              astream,
              {{DNNL_ARG_SRC, *src_memory_p},
               {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
               {DNNL_ARG_DIFF_WEIGHTS, *diff_weights_memory_p}});
          astream.wait();

          // For convolution with groups convert from blocked to NCHW
          // otherwise there will be problems in next operators working on
          // this data
          if (g > 1) {
            // in OneDNN groups in convolution are treated as separate
            // dimension which is not the case in paddlepaddle

            dnnl::memory::data_type in_type =
                funcs::ToOneDNNDataType(filter.dtype());
            // for 3d conv with groups (six dimensional data reorder to
            // goidhw) for 2d conv with groups (five dimensional data reorder
            // to goihw) auto weights_tz = phi::vectorize(filter->dims());

            auto weights_tz = diff_weights_memory_p->get_desc().dims();
            dnnl::memory::format_tag out_format =
                weights_tz.size() == 6 ? dnnl::memory::format_tag::goidhw
                                       : dnnl::memory::format_tag::goihw;
            funcs::ReorderOneDNNHandler handler(
                weights_tz, filter.dtype(), in_type, onednn_engine);
            auto reorder_dst_memory_p = handler.AcquireDstMemory(
                filter_grad, out_format, dev_ctx.GetPlace());

            auto reorder_p = handler.AcquireReorder(reorder_dst_memory_p,
                                                    diff_weights_memory_p);

            {
              paddle::platform::RecordEvent record_reorder(
                  "int_reorder",
                  paddle::platform::TracerEventType::UserDefined,
                  1,
                  paddle::platform::EventRole::kUniqueOp);
              reorder_p->execute(
                  astream, *diff_weights_memory_p, *reorder_dst_memory_p);
              astream.wait();
            }

            // So here we have a data in goihw , which can be interpreted as
            // OIHW (OIDHW for conv3d) because filter_grad shape is set for
            // OIHW (OIDHW for conv3d)
            dnnl::memory::format_tag target_format =
                weights_tz.size() == 6 ? dnnl::memory::format_tag::oidhw
                                       : dnnl::memory::format_tag::oihw;
            filter_grad->set_mem_desc(
                dnnl::memory::desc(phi::vectorize<int64_t>(filter_grad->dims()),
                                   in_type,
                                   target_format));
          } else {
            filter_grad->set_mem_desc(diff_weights_memory_p->get_desc());
          }
        }
        if (input_grad) {
          auto weights_memory_p =
              handler.AcquireWeightsMemoryWithReorderFromDataPrimitive(
                  &filter, groups, strides.size() == 3U);

          auto diff_dst_memory_p =
              handler.AcquireDiffDstMemoryWithReorderMemoryFromDataPrimitive(
                  &out_grad);
          auto diff_src_memory_p = handler.AcquireDiffSrcMemory(input_grad);

          auto conv_bwd_data_p = handler.AcquireBackwardPrimitive();

          conv_bwd_data_p->execute(astream,
                                   {{DNNL_ARG_WEIGHTS, *weights_memory_p},
                                    {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                    {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
          astream.wait();

          input_grad->set_mem_desc(diff_src_memory_p->get_desc());
        }
      }));
}

template <typename T, typename Context>
void DepthwiseConvGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& filter,
                             const DenseTensor& out_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format,
                             DenseTensor* input_grad,
                             DenseTensor* filter_grad) {
  ConvGradKernel<T, Context>(dev_ctx,
                             input,
                             filter,
                             out_grad,
                             strides,
                             paddings,
                             padding_algorithm,
                             dilations,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad);
}

template <typename T, typename Context>
void Conv3DGradKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      DenseTensor* input_grad,
                      DenseTensor* filter_grad) {
  ConvGradKernel<T, Context>(dev_ctx,
                             input,
                             filter,
                             out_grad,
                             strides,
                             paddings,
                             padding_algorithm,
                             dilations,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d_grad,
                   OneDNN,
                   ONEDNN,
                   phi::ConvGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad,
                   OneDNN,
                   ONEDNN,
                   phi::DepthwiseConvGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d_grad, OneDNN, ONEDNN, phi::Conv3DGradKernel, float) {}
