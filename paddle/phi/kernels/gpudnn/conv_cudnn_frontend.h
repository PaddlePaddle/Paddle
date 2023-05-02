/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

namespace phi {

class CudnnFrontendConvHelper {
 public:
  static bool IsNonDeterministic(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<
        CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
  }
  static bool AllowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
  }

  static uint8_t GetAlignment(const phi::DenseTensor* tensor) {
    // alignment are in bytes
    uint8_t alignment = 1;
    uint64_t address = reinterpret_cast<uint64_t>(tensor->data());
    while (address % alignment == 0 && alignment < 16) alignment *= 2;
    return alignment;
  }

  static std::vector<int64_t> GetInt64Array(const std::vector<int>& in_array) {
    std::vector<int64_t> out_array(in_array.size());
    for (int i = 0; i < in_array.size(); i++) {
      out_array[i] = static_cast<int64_t>(in_array[i]);
    }
    return out_array;
  }

  static std::vector<int64_t> GenerateStrides(
      const std::vector<int64_t>& dim, cudnnTensorFormat_t filter_format) {
    // ref:
    // https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/helpers.cpp
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the
    // cpu ref.
    size_t nb_dims = dim.size();
    std::vector<int64_t> stride(nb_dims);
    if (filter_format == CUDNN_TENSOR_NCHW) {
      stride[nb_dims - 1] = 1;
      for (int64_t d = nb_dims - 2; d >= 0; d--) {
        stride[d] = stride[d + 1] * dim[d + 1];
      }
    } else {
      // Here we assume that the format is CUDNN_TENSOR_NHWC
      stride[1] = 1;
      stride[nb_dims - 1] = stride[1] * dim[1];
      for (int64_t d = nb_dims - 2; d >= 2; d--) {
        stride[d] = stride[d + 1] * dim[d + 1];
      }
      stride[0] = stride[2] * dim[2];
    }
    return stride;
  }

  static cudnn_frontend::Tensor GetTensorDescriptor(
      const phi::DenseTensor* tensor,
      int64_t id,
      cudnnTensorFormat_t layout_format) {
    auto transformed_dims = phi::vectorize<int64_t>(tensor->dims());
    if (layout_format == CUDNN_TENSOR_NHWC) {
      transformed_dims =
          phi::backends::gpu::TransformDimOrder(transformed_dims);
    }
    std::vector<int64_t> strides =
        GenerateStrides(transformed_dims, layout_format);
    return cudnn_frontend::TensorBuilder()
        .setDim(transformed_dims.size(), transformed_dims.data())
        .setStrides(strides.size(), strides.data())
        .setId(id)
        .setAlignment(GetAlignment(tensor))
        .setDataType(phi::backends::gpu::ToCudnnDataType(tensor->dtype()))
        .build();
  }

  static cudnn_frontend::ConvDesc_v8 GetConvDescriptor(
      cudnnDataType_t dataType,
      const std::vector<int>& padding,
      const std::vector<int>& stride,
      const std::vector<int>& dilation) {
    uint64_t conv_dim = stride.size();
    cudnnDataType_t compute_type =
        (dataType == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    std::vector<int64_t> padding_int64 = GetInt64Array(padding);
    std::vector<int64_t> stride_int64 = GetInt64Array(stride);
    std::vector<int64_t> dilation_int64 = GetInt64Array(dilation);
    return cudnn_frontend::ConvDescBuilder()
        .setDataType(compute_type)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setNDims(conv_dim)
        .setStrides(conv_dim, stride_int64.data())
        .setPrePadding(conv_dim, padding_int64.data())
        .setPostPadding(conv_dim, padding_int64.data())
        .setDilation(conv_dim, dilation_int64.data())
        .build();
  }

  template <cudnnBackendDescriptorType_t op_mode>
  static cudnn_frontend::OperationGraph BuildConvOperationGraph(
      const phi::DenseTensor* x_tensor,
      const phi::DenseTensor* y_tensor,
      const phi::DenseTensor* w_tensor,
      cudnnTensorFormat_t layout_format,
      const std::vector<int>& strides,
      const std::vector<int>& padding_common,
      const std::vector<int>& dilations,
      cudnnDataType_t dtype,
      cudnnHandle_t handle,
      float alpha,
      float beta) {
    auto op = cudnn_frontend::OperationBuilder(op_mode)
                  .setxDesc(GetTensorDescriptor(x_tensor, 'x', layout_format))
                  .setyDesc(GetTensorDescriptor(y_tensor, 'y', layout_format))
                  .setwDesc(GetTensorDescriptor(w_tensor, 'w', layout_format))
                  .setcDesc(GetConvDescriptor(
                      dtype, padding_common, strides, dilations))
                  .setAlpha(alpha)
                  .setBeta(beta)
                  .build();
    std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
    return cudnn_frontend::OperationGraphBuilder()
        .setHandle(handle)
        .setOperationGraph(1, ops.data())
        .build();
  }

  static cudnn_frontend::executionPlans_t FindExecutionPlans(
      cudnn_frontend::OperationGraph* op_graph_pointer,
      bool exhaustive_search,
      bool deterministic,
      void* x_data,
      void* y_data,
      void* w_data,
      cudnnHandle_t handle,
      phi::DnnWorkspaceHandle* workspace_handle) {
    auto heurgen_method = [=](cudnn_frontend::OperationGraph& op_graph_)
        -> cudnn_frontend::EngineConfigList {
      auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                            .setOperationGraph(op_graph_)
                            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                            .build();
      VLOG(4) << "Heuristic has " << heuristics.getEngineConfigCount()
              << " configurations ";

      auto& engine_configs =
          heuristics.getEngineConfig(heuristics.getEngineConfigCount());
      cudnn_frontend::EngineConfigList filtered_configs;
      cudnn_frontend::filter(engine_configs,
                             filtered_configs,
                             deterministic ? IsNonDeterministic : AllowAll);
      return filtered_configs;
    };

    auto fallback_method = [=](cudnn_frontend::OperationGraph& op_graph_)
        -> cudnn_frontend::EngineConfigList {
      auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                          .setOperationGraph(op_graph_)
                          .build();
      auto& fallback_list = fallback.getFallbackList();
      cudnn_frontend::EngineConfigList filtered_configs;
      cudnn_frontend::filter(fallback_list,
                             filtered_configs,
                             deterministic ? IsNonDeterministic : AllowAll);
      return filtered_configs;
    };

    std::array<cudnn_frontend::GeneratorSource const, 2> sources = {
        heurgen_method, fallback_method};
    cudnn_frontend::EngineConfigGenerator generator(sources.size(),
                                                    sources.data());

    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    auto predicate_function =
        [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
      return plan.getWorkspaceSize() > workspace_size_limit;
    };

    auto plans =
        generator.cudnnGetPlan(handle, *op_graph_pointer, predicate_function);

    bool use_autotune = phi::autotune::AutoTuneStatus::Instance().UseAutoTune();

    if (!deterministic && (exhaustive_search || use_autotune)) {
      size_t workspace_size_max = 0;
      std::for_each(
          plans.begin(), plans.end(), [&](cudnn_frontend::ExecutionPlan& opt) {
            if (opt.getWorkspaceSize() > workspace_size_max) {
              workspace_size_max = opt.getWorkspaceSize();
            }
          });
      VLOG(6) << "[cudnn_frontend] Max workspace size: " << workspace_size_max;
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            void* data_ptrs[] = {x_data, y_data, w_data};
            int64_t uids[] = {'x', 'y', 'w'};
            auto variant_pack = cudnn_frontend::VariantPackBuilder()
                                    .setWorkspacePointer(workspace_ptr)
                                    .setDataPointers(3, data_ptrs)
                                    .setUids(3, uids)
                                    .build();
            plans =
                generator
                    .cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::
                                       CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                        handle,
                        *op_graph_pointer,
                        variant_pack,
                        predicate_function);
          },
          workspace_size_max);
    }

    std::for_each(
        plans.begin(), plans.end(), [](cudnn_frontend::ExecutionPlan& opt) {
          VLOG(6) << "Plan tag: " << opt.getTag() << " finished in "
                  << opt.getExecutionTime() << " ms,"
                  << " workspace: " << opt.getWorkspaceSize() << " bytes";
        });

    return plans;
  }
};  // class CudnnFrontendConvHelper

template <typename T>
void CudnnConvBwdDataV8(const DenseTensor* dy_tensor,
                        const DenseTensor* w_tensor,
                        cudnnHandle_t handle,
                        DnnWorkspaceHandle* workspace_handle,
                        const std::vector<int>& strides,
                        const std::vector<int>& padding_common,
                        const std::vector<int>& dilations,
                        cudnnDataType_t dtype,
                        cudnnTensorFormat_t layout_format,
                        bool use_addto,
                        bool exhaustive_search,
                        bool deterministic,
                        DenseTensor* dx_tensor) {
  auto& plan_cache_bwd_data =
      phi::autotune::AutoTuneCache::Instance().GetConvV8(
          phi::autotune::AlgorithmType::kConvBackwardDataV8);
  T* dy_tensor_data = const_cast<T*>(dy_tensor->data<T>());
  T* w_tensor_data = const_cast<T*>(w_tensor->data<T>());
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
    auto engine_config =
        plan_cache_bwd_data.GetConfig(op_graph, handle, use_addto);
    auto cached_plan = cudnn_frontend::ExecutionPlanBuilder()
                           .setHandle(handle)
                           .setEngineConfig(engine_config, op_graph.getTag())
                           .build();
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
void CudnnConvBwdFilterV8(const DenseTensor* x_tensor,
                          const DenseTensor* dy_tensor,
                          cudnnHandle_t handle,
                          DnnWorkspaceHandle* workspace_handle,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding_common,
                          const std::vector<int>& dilations,
                          cudnnDataType_t dtype,
                          cudnnTensorFormat_t layout_format,
                          bool use_addto,
                          bool exhaustive_search,
                          bool deterministic,
                          DenseTensor* dw_tensor) {
  auto& plan_cache_bwd_filter =
      phi::autotune::AutoTuneCache::Instance().GetConvV8(
          phi::autotune::AlgorithmType::kConvBackwardFilterV8);
  T* x_tensor_data = const_cast<T*>(x_tensor->data<T>());
  T* dy_tensor_data = const_cast<T*>(dy_tensor->data<T>());
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
    auto engine_config = plan_cache_bwd_filter.GetConfig(op_graph, handle);
    auto cached_plan = cudnn_frontend::ExecutionPlanBuilder()
                           .setHandle(handle)
                           .setEngineConfig(engine_config, op_graph.getTag())
                           .build();
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
      VLOG(4) << "Plan " << plan.describe()
              << "failed to execute. Trying next plan.";
    } catch (phi::enforce::EnforceNotMet& e) {
      VLOG(4) << "Plan " << plan.describe()
              << "failed to execute. Trying next plan.";
    }
  }

  PADDLE_THROW(phi::errors::InvalidArgument(
      "[CUDNN Frontend API] No valid plan could "
      "be found to execute conv backward filter."));
}

}  // namespace phi
