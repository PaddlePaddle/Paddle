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

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/impl/conv_cudnn_impl.h"

#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace phi {

template <typename T, typename Context>
void CudnnConvFwdV8(const DenseTensor* input_tensor,
                    const DenseTensor* filter_channel_tensor,
                    const Context& ctx,
                    const std::vector<int>& strides,
                    const std::vector<int>& padding_common,
                    const std::vector<int>& dilations,
                    cudnnDataType_t dtype,
                    cudnnTensorFormat_t layout_format,
                    bool exhaustive_search,
                    bool deterministic,
                    int groups,
                    DenseTensor* output_tensor) {
  static CudnnFrontendPlanCache plan_cache("conv");
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      paddle::platform::errors::Unimplemented(
          "Group concolution using CUDNNv8 API unsupported for now"));

  T* input_data = const_cast<T*>(input_tensor->data<T>());
  T* filter_data = const_cast<T*>(filter_channel_tensor->data<T>());
  T* output_data = output_tensor->data<T>();
  cudnnHandle_t handle = const_cast<cudnnHandle_t>(ctx.cudnn_handle());
  auto workspace_handle = ctx.cudnn_workspace_handle();

  float alpha = 1.0f;
  float beta = 0.0f;

  using helper = CudnnFrontendConvHelper;
  auto op_graph = helper::BuildConvOperationGraph<
      CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR>(
      input_tensor,
      output_tensor,
      filter_channel_tensor,
      layout_format,
      strides,
      padding_common,
      dilations,
      dtype,
      handle,
      alpha,
      beta);

  if (plan_cache.FindPlan(op_graph)) {
    auto cached_plan = plan_cache.GetPlan(op_graph, handle);
    auto workspace_size = cached_plan.getWorkspaceSize();
    VLOG(4) << "Cached execution plan found." << cached_plan.getTag()
            << "; Require workspace: " << workspace_size;
    workspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          void* data_ptrs[] = {input_data, output_data, filter_data};
          int64_t uids[] = {'x', 'y', 'w'};
          auto variant_pack = cudnn_frontend::VariantPackBuilder()
                                  .setWorkspacePointer(workspace_ptr)
                                  .setDataPointers(3, data_ptrs)
                                  .setUids(3, uids)
                                  .build();
          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendExecute(
              handle, cached_plan.get_raw_desc(), variant_pack.get_raw_desc()));
        },
        workspace_size);
    return;
  }

  auto plans = helper::FindExecutionPlans(&op_graph,
                                          exhaustive_search,
                                          deterministic,
                                          input_data,
                                          output_data,
                                          filter_data,
                                          handle,
                                          &workspace_handle);

  for (auto& plan : plans) {
    try {
      int64_t workspace_size = plan.getWorkspaceSize();
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            void* data_ptrs[] = {input_data, output_data, filter_data};
            int64_t uids[] = {'x', 'y', 'w'};
            auto variant_pack = cudnn_frontend::VariantPackBuilder()
                                    .setWorkspacePointer(workspace_ptr)
                                    .setDataPointers(3, data_ptrs)
                                    .setUids(3, uids)
                                    .build();
            PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendExecute(
                handle, plan.get_raw_desc(), variant_pack.get_raw_desc()));
          },
          workspace_size);
      if (!exhaustive_search || plan_cache.IsStable(op_graph, plan.getTag())) {
        plan_cache.InsertPlan(op_graph, plan);
      }
      return;
    } catch (cudnn_frontend::cudnnException& e) {
    } catch (phi::enforce::EnforceNotMet& e) {
    }
  }
  PADDLE_ENFORCE_EQ(
      true,
      false,
      phi::errors::InvalidArgument("[CUDNN Frontend API] No valid plan could "
                                   "be found to execute conv."));
}

template <typename T, typename Context>
void ConvCudnnKernelImplV8(const Context& ctx,
                           const DenseTensor& input,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings_t,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations_t,
                           const std::string& data_format,
                           bool use_addto,
                           int workspace_size_MB,
                           bool exhaustive_search_t,
                           DenseTensor* output) {
  ctx.template Alloc<T>(output);
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  bool exhaustive_search = FLAGS_cudnn_exhaustive_search || exhaustive_search_t;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    phi::errors::InvalidArgument(
                        "Cann't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
  auto dtype = paddle::platform::CudnnDataType<T>::type;

  // Tensor Core introduced from Volta GPUs supports more faster conv op
  // with FP16 in NHWC data format.
  const bool compute_in_nhwc = dtype == CUDNN_DATA_HALF && IsVoltaOrLater(ctx);
  // We will only do data format conversion from NHWC to NCHW.
  // cudnn will convert NCHW to NHWC automatically on Tensor Core.
  auto compute_format = compute_in_nhwc && channel_last
                            ? paddle::platform::DataLayout::kNHWC
                            : paddle::platform::DataLayout::kNCHW;
  VLOG(3) << "Compute ConvOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == paddle::platform::DataLayout::kNHWC ? "NHWC"
                                                                    : "NCHW");

  // ------------ transformed tensor -----------
  DenseTensor transformed_input_channel(input.type());
  DenseTensor transformed_output(output->type());
  DenseTensor transformed_filter_channel(filter.type());
  if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);
    TransToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);

    ResizeToChannelFirst<Context, T>(ctx, output, &transformed_output);

  } else {
    transformed_input_channel.ShareDataWith(input);
    transformed_output.ShareDataWith(*output);
  }
  if (compute_format == paddle::platform::DataLayout::kNHWC) {
    VLOG(3) << "Transform filter tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
  } else {
    transformed_filter_channel.ShareDataWith(filter);
  }
  // update padding and dilation
  auto in_dims = transformed_input_channel.dims();
  auto filter_dims = transformed_filter_channel.dims();
  DDim in_data_dims;
  DDim filter_data_dims;

  if (compute_format == paddle::platform::DataLayout::kNCHW) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
    filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    filter_data_dims = slice_ddim(filter_dims, 1, filter_dims.size() - 1);
  }

  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);

  DenseTensor transformed_input;
  std::vector<int> padding_common(data_dim, 0);
  if (!is_sys_pad) {
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];

    if (compute_format == paddle::platform::DataLayout::kNCHW) {
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];
    } else {
      new_input_shape_vec[data_dim + 1] =
          transformed_input_channel.dims()[data_dim + 1];
    }

    std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);
    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
      padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
      if (compute_format == paddle::platform::DataLayout::kNCHW) {
        new_input_shape_vec[i + 2] =
            transformed_input_channel.dims()[i + 2] + padding_diff[i];
      } else {
        new_input_shape_vec[i + 1] =
            transformed_input_channel.dims()[i + 1] + padding_diff[i];
      }
      if (compute_format == paddle::platform::DataLayout::kNCHW) {
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      } else {
        input_pad[2 * i + 2] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 2 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
    }
    DDim new_input_shape(make_ddim(new_input_shape_vec));
    transformed_input.Resize(new_input_shape);
    ctx.template Alloc<T>(&transformed_input);

    const int rank = transformed_input_channel.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(ctx,
                                          input_pad,
                                          transformed_input_channel,
                                          pad_value,
                                          &transformed_input);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(ctx,
                                          input_pad,
                                          transformed_input_channel,
                                          pad_value,
                                          &transformed_input);
      } break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
    }

  } else {
    transformed_input.ShareDataWith(transformed_input_channel);
    if (paddings.size() == data_dim) {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[i];
      }
    } else {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[2 * i];
      }
    }
  }

  paddle::platform::DataLayout layout =
      compute_format == paddle::platform::DataLayout::kNHWC
          ? paddle::platform::DataLayout::kNHWC
          : paddle::platform::DataLayout::kNCHW;
  if (transformed_input.dims().size() == 5) {
    layout = compute_format == paddle::platform::DataLayout::kNHWC
                 ? paddle::platform::DataLayout::kNDHWC
                 : paddle::platform::DataLayout::kNCDHW;
  }
  auto layout_format = paddle::platform::GetCudnnTensorFormat(layout);

  CudnnConvFwdV8<T, Context>(&transformed_input,
                             &transformed_filter_channel,
                             ctx,
                             strides,
                             padding_common,
                             dilations,
                             dtype,
                             layout_format,
                             exhaustive_search,
                             deterministic,
                             groups,
                             &transformed_output);

  if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
    TransToChannelLast<Context, T>(ctx, &transformed_output, output);
  }
}

}  // namespace phi
