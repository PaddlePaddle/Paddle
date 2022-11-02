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

#include "paddle/phi/kernels/conv_grad_kernel.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
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

template <typename T>
void CudnnConvBwdDataV8(DenseTensor* dy_tensor,
                        DenseTensor* w_tensor,
                        cudnnHandle_t handle,
                        DnnWorkspaceHandle* workspace_handle,
                        const std::vector<int>& strides,
                        const std::vector<int>& padding_common,
                        const std::vector<int>& dilations,
                        cudnnDataType_t dtype,
                        paddle::platform::DataLayout compute_format,
                        cudnnTensorFormat_t layout_format,
                        bool use_addto,
                        bool exhaustive_search,
                        bool deterministic,
                        DenseTensor* dx_tensor) {
  static CudnnFrontendPlanCache plan_cache_bwd_data("conv_backward_data");
  T* dy_tensor_data = dy_tensor->data<T>();
  T* w_tensor_data = w_tensor->data<T>();
  T* dx_tensor_data = dx_tensor->data<T>();

  float alpha = 1.0f;
  float beta = use_addto ? 1.0f : 0.0f;

  using helper = CudnnFrontendConvHelper;
  auto op_graph = helper::BuildConvOperationGraph<
      CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR>(
      dx_tensor,
      dy_tensor,
      w_tensor,
      layout_format,
      strides,
      padding_common,
      dilations,
      dtype,
      handle,
      alpha,
      beta);

  if (plan_cache_bwd_data.FindPlan(op_graph, use_addto)) {
    auto cached_plan = plan_cache_bwd_data.GetPlan(op_graph, handle, use_addto);
    auto workspace_size = cached_plan.getWorkspaceSize();
    VLOG(4) << "Cached execution plan found." << cached_plan.getTag()
            << "; Require workspace: " << workspace_size;
    workspace_handle->RunFunc(
        [&](void* workspace_ptr) {
          void* data_ptrs[] = {dx_tensor_data, dy_tensor_data, w_tensor_data};
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
                                          dx_tensor_data,
                                          dy_tensor_data,
                                          w_tensor_data,
                                          handle,
                                          workspace_handle);

  for (auto& plan : plans) {
    try {
      int64_t workspace_size = plan.getWorkspaceSize();
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            void* data_ptrs[] = {dx_tensor_data, dy_tensor_data, w_tensor_data};
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
      if (!exhaustive_search ||
          plan_cache_bwd_data.IsStable(op_graph, plan.getTag(), use_addto)) {
        plan_cache_bwd_data.InsertPlan(op_graph, plan, use_addto);
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
                                   "be found to execute conv backward data."));
}

template <typename T>
void CudnnConvBwdFilterV8(DenseTensor* x_tensor,
                          DenseTensor* dy_tensor,
                          cudnnHandle_t handle,
                          DnnWorkspaceHandle* workspace_handle,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding_common,
                          const std::vector<int>& dilations,
                          cudnnDataType_t dtype,
                          paddle::platform::DataLayout compute_format,
                          cudnnTensorFormat_t layout_format,
                          bool use_addto,
                          bool exhaustive_search,
                          bool deterministic,
                          DenseTensor* dw_tensor) {
  static CudnnFrontendPlanCache plan_cache_bwd_filter("conv_backward_filter");
  T* x_tensor_data = x_tensor->data<T>();
  T* dy_tensor_data = dy_tensor->data<T>();
  T* dw_tensor_data = dw_tensor->data<T>();

  float alpha = 1.0f;
  float beta = 0.0f;

  using helper = CudnnFrontendConvHelper;
  auto op_graph = helper::BuildConvOperationGraph<
      CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR>(
      x_tensor,
      dy_tensor,
      dw_tensor,
      layout_format,
      strides,
      padding_common,
      dilations,
      dtype,
      handle,
      alpha,
      beta);

  if (plan_cache_bwd_filter.FindPlan(op_graph)) {
    auto cached_plan = plan_cache_bwd_filter.GetPlan(op_graph, handle);
    auto workspace_size = cached_plan.getWorkspaceSize();
    VLOG(4) << "Cached execution plan found." << cached_plan.getTag()
            << "; Require workspace: " << workspace_size;
    workspace_handle->RunFunc(
        [&](void* workspace_ptr) {
          void* data_ptrs[] = {x_tensor_data, dy_tensor_data, dw_tensor_data};
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
                                          x_tensor_data,
                                          dy_tensor_data,
                                          dw_tensor_data,
                                          handle,
                                          workspace_handle);

  for (auto& plan : plans) {
    try {
      int64_t workspace_size = plan.getWorkspaceSize();
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            void* data_ptrs[] = {x_tensor_data, dy_tensor_data, dw_tensor_data};
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
      if (!exhaustive_search ||
          plan_cache_bwd_filter.IsStable(op_graph, plan.getTag())) {
        plan_cache_bwd_filter.InsertPlan(op_graph, plan);
      }
      return;
    } catch (cudnn_frontend::cudnnException& e) {
    } catch (phi::enforce::EnforceNotMet& e) {
    }
  }

  PADDLE_ENFORCE_EQ(true,
                    false,
                    phi::errors::InvalidArgument(
                        "[CUDNN Frontend API] No valid plan could "
                        "be found to execute conv backward filter."));
}

template <typename T, typename Context>
void ConvCudnnGradKernelImplV8(const Context& ctx,
                               const DenseTensor& input,
                               const DenseTensor& filter,
                               const DenseTensor& output_grad,
                               const std::vector<int>& strides_t,
                               const std::vector<int>& paddings_t,
                               const std::string& padding_algorithm,
                               const std::vector<int>& dilations_t,
                               int groups,
                               const std::string& data_format,
                               DenseTensor* input_grad,
                               DenseTensor* filter_grad) {
  if (input_grad) {
    ctx.template Alloc<T>(input_grad);
  }
  if (filter_grad) {
    ctx.template Alloc<T>(filter_grad);
  }

  bool has_use_addto = ctx.HasDnnAttr("use_addto");
  VLOG(4) << "GPUContext contains `use_addto`: " << has_use_addto;
  bool use_addto = has_use_addto
                       ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("use_addto"))
                       : false;

  std::vector<int> dilations = dilations_t;
  std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;

  bool has_exhaustive_search = ctx.HasDnnAttr("exhaustive_search");
  VLOG(4) << "GPUContext contains `exhaustive_search`: "
          << has_exhaustive_search;
  bool exhaustive_search_attr =
      has_exhaustive_search
          ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("exhaustive_search"))
          : false;
  bool exhaustive_search =
      FLAGS_cudnn_exhaustive_search || exhaustive_search_attr;
  bool deterministic = FLAGS_cudnn_deterministic;
  auto exhaustive_deterministic = exhaustive_search && deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_deterministic,
                    false,
                    phi::errors::InvalidArgument(
                        "Cann't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  auto dtype = paddle::platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
  // HIP MIOPEN ONLY SUPPORT NCHW format
  auto compute_format = paddle::platform::DataLayout::kNCHW;
#else
  const bool compute_in_nhwc = dtype == CUDNN_DATA_HALF && IsVoltaOrLater(ctx);
  auto compute_format = compute_in_nhwc && channel_last
                            ? paddle::platform::DataLayout::kNHWC
                            : paddle::platform::DataLayout::kNCHW;
#endif
  VLOG(3) << "Compute ConvGradOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == paddle::platform::DataLayout::kNHWC ? "NHWC"
                                                                    : "NCHW");

  // transform Tensor
  DenseTensor transformed_input_channel(input.type());
  DenseTensor transformed_output_grad_channel(output_grad.type());
  DenseTensor transformed_input_grad_channel(input.type());
  DenseTensor transformed_filter_channel(filter.type());
  DenseTensor transformed_filter_grad_channel(filter.type());

  if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
    VLOG(3) << "Transform input, output_grad, input_grad and tensor from "
               "NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);
    TransToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);

    ResizeToChannelFirst<Context, T>(
        ctx, &output_grad, &transformed_output_grad_channel);
    TransToChannelFirst<Context, T>(
        ctx, &output_grad, &transformed_output_grad_channel);

    if (input_grad) {
      ResizeToChannelFirst<Context, T>(
          ctx, input_grad, &transformed_input_grad_channel);
      // NOTE(zhiqiu): If inplace_addto strategy is enabled, we need to copy
      // the data of input_grad to transformed_input_grad_channel.
      if (use_addto) {
        TransToChannelFirst<Context, T>(
            ctx, input_grad, &transformed_input_grad_channel);
      }
    }
  } else {
    transformed_input_channel.ShareDataWith(input);
    transformed_output_grad_channel.ShareDataWith(output_grad);
    if (input_grad) {
      transformed_input_grad_channel.ShareDataWith(*input_grad);
    }
  }

  if (compute_format == paddle::platform::DataLayout::kNHWC) {
    VLOG(3) << "Transform filter and filter_grad tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);

    if (filter_grad) {
      ResizeToChannelLast<Context, T>(
          ctx, filter_grad, &transformed_filter_grad_channel);
    }
  } else {
    transformed_filter_channel.ShareDataWith(filter);
    if (filter_grad) {
      transformed_filter_grad_channel.ShareDataWith(*filter_grad);
    }
  }

  //  update paddings
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

  // cuDNN only supports padding the same amount on every dimension.
  // So we create a new padded input tensor.
  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);
  Tensor transformed_input(input.type());
  Tensor transformed_input_grad(input.type());
  std::vector<int> padding_common(data_dim, 0);
  std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);

  if (!is_sys_pad) {
    // get pad
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];
    if (compute_format == paddle::platform::DataLayout::kNCHW) {
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];
    } else {
      new_input_shape_vec[data_dim + 1] =
          transformed_input_channel.dims()[data_dim + 1];
    }

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

    transformed_input_grad.Resize(new_input_shape);

    if (input_grad) {
      ctx.template Alloc<T>(&transformed_input_grad);
    }
    // pad for input
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
    if (input_grad) {
      transformed_input_grad.ShareDataWith(transformed_input_grad_channel);
    }
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

  auto handle = ctx.cudnn_handle();
  paddle::platform::DataLayout layout =
      compute_format == paddle::platform::DataLayout::kNHWC
          ? paddle::platform::DataLayout::kNHWC
          : paddle::platform::DataLayout::kNCHW;

  if (transformed_input.dims().size() == 5) {
    layout = compute_format == paddle::platform::DataLayout::kNHWC
                 ? paddle::platform::DataLayout::kNDHWC
                 : paddle::platform::DataLayout::kNCDHW;
  }
  auto layout_tensor = paddle::platform::GetCudnnTensorFormat(layout);
  auto workspace_handle = ctx.cudnn_workspace_handle();

  if (input_grad) {
    CudnnConvBwdDataV8<T>(&transformed_output_grad_channel,
                          &transformed_filter_channel,
                          handle,
                          &workspace_handle,
                          strides,
                          padding_common,
                          dilations,
                          dtype,
                          compute_format,
                          layout_tensor,
                          use_addto,
                          exhaustive_search,
                          deterministic,
                          &transformed_input_grad);
  }

  if (filter_grad) {
    CudnnConvBwdFilterV8<T>(&transformed_input,
                            &transformed_output_grad_channel,
                            handle,
                            &workspace_handle,
                            strides,
                            padding_common,
                            dilations,
                            dtype,
                            compute_format,
                            layout_tensor,
                            use_addto,
                            exhaustive_search,
                            deterministic,
                            &transformed_filter_grad_channel);
  }

  if (input_grad) {
    if (!is_sys_pad) {
      std::vector<int> starts(transformed_input_channel.dims().size(), 0);
      std::vector<int> axes(transformed_input_channel.dims().size(), 0);

      for (size_t i = 0; i < transformed_input_channel.dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }

      ctx.template Alloc<T>(&transformed_input_grad_channel);
      if (transformed_input_channel.dims().size() == 4) {
        RemovePaddingSlice<Context, T, 4>(ctx,
                                          &transformed_input_grad,
                                          &transformed_input_grad_channel,
                                          starts,
                                          axes);
      } else {
        RemovePaddingSlice<Context, T, 5>(ctx,
                                          &transformed_input_grad,
                                          &transformed_input_grad_channel,
                                          starts,
                                          axes);
      }
    }

    if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
      TransToChannelLast<Context, T>(
          ctx, &transformed_input_grad_channel, input_grad);
    }
  }

  if (filter_grad) {
    if (compute_format == paddle::platform::DataLayout::kNHWC) {
      TransToChannelFirst<Context, T>(
          ctx, &transformed_filter_grad_channel, filter_grad);
    }
  }
}

}  // namespace phi
