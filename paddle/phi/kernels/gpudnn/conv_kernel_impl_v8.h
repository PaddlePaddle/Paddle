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
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace phi {

template <typename T, typename Context>
void ConvCudnnKernelImplV8(DenseTensor* input_tensor,
                           DenseTensor* filter_channel_tensor,
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
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kConvForwardV8);

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      paddle::platform::errors::Unimplemented(
          "Group concolution using CUDNNv8 API unsupported for now"));

  T* input_data = input_tensor->data<T>();
  T* filter_data = filter_channel_tensor->data<T>();
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
  PADDLE_THROW(
      phi::errors::InvalidArgument("[CUDNN Frontend API] No valid plan could "
                                   "be found to execute conv."));
}

}  // namespace phi
