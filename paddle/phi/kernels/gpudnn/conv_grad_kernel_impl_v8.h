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
#include "paddle/phi/kernels/autotune/cache.h"
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
  auto& plan_cache_bwd_data =
      phi::autotune::AutoTuneCache::Instance().GetConvV8(
          phi::autotune::AlgorithmType::kConvBackwardDataV8);
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
  PADDLE_THROW(
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
  auto& plan_cache_bwd_filter =
      phi::autotune::AutoTuneCache::Instance().GetConvV8(
          phi::autotune::AlgorithmType::kConvBackwardFilterV8);
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

  PADDLE_THROW(phi::errors::InvalidArgument(
      "[CUDNN Frontend API] No valid plan could "
      "be found to execute conv backward filter."));
}

template <typename T, typename Context>
void ConvCudnnGradKernelImplV8(DenseTensor* transformed_input,
                               DenseTensor* transformed_filter_channel,
                               DenseTensor* transformed_output_grad_channel,
                               DenseTensor* input_grad,
                               DenseTensor* filter_grad,
                               const Context& ctx,
                               const std::vector<int>& strides,
                               const std::vector<int>& padding_common,
                               const std::vector<int>& dilations,
                               cudnnDataType_t dtype,
                               paddle::platform::DataLayout compute_format,
                               cudnnTensorFormat_t layout_format,
                               bool use_addto,
                               bool exhaustive_search,
                               bool deterministic,
                               int groups,
                               DenseTensor* transformed_input_grad,
                               DenseTensor* transformed_filter_grad_channel) {
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      paddle::platform::errors::Unimplemented(
          "Group concolution using CUDNNv8 API is unsupported for now"));

  cudnnHandle_t handle = const_cast<cudnnHandle_t>(ctx.cudnn_handle());
  auto workspace_handle = ctx.cudnn_workspace_handle();

  if (input_grad) {
    CudnnConvBwdDataV8<T>(transformed_output_grad_channel,
                          transformed_filter_channel,
                          handle,
                          &workspace_handle,
                          strides,
                          padding_common,
                          dilations,
                          dtype,
                          compute_format,
                          layout_format,
                          use_addto,
                          exhaustive_search,
                          deterministic,
                          transformed_input_grad);
  }

  if (filter_grad) {
    CudnnConvBwdFilterV8<T>(transformed_input,
                            transformed_output_grad_channel,
                            handle,
                            &workspace_handle,
                            strides,
                            padding_common,
                            dilations,
                            dtype,
                            compute_format,
                            layout_format,
                            use_addto,
                            exhaustive_search,
                            deterministic,
                            transformed_filter_grad_channel);
  }
}

}  // namespace phi
