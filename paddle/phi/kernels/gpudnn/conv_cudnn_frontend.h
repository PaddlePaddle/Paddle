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
#include "paddle/phi/kernels/gpudnn/conv_gpudnn_base.h"

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
    auto transformed_dims = common::vectorize<int64_t>(tensor->dims());
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

  static inline cudnn_frontend::Tensor GetGeneralTensorDescriptor(
      std::vector<int64_t> dims,
      cudnnTensorFormat_t layout,
      int64_t id,
      int64_t alignment,
      cudnnDataType_t dtype,
      bool is_virtual = false,
      int64_t group_count = 0) {
    std::vector<int64_t> strides = GenerateStrides(dims, layout);
    if (group_count > 0) {
      int64_t c_per_group = dims[1];
      int64_t c_stride = strides[1];
      dims.insert(dims.begin() + 1, group_count);
      strides.insert(strides.begin() + 1, c_stride * c_per_group);
    }
    cudnn_frontend::TensorBuilder builder;
    builder.setDim(dims.size(), dims.data())
        .setStride(strides.size(), strides.data())
        .setId(id)
        .setAlignment(alignment)
        .setDataType(dtype);
    if (is_virtual) {
      builder.setVirtual();
    }
    return builder.build();
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
      std::vector<void*>* data_ptrs,
      std::vector<int64_t>* uids,
      cudnnHandle_t handle,
      phi::DnnWorkspaceHandle* workspace_handle) {
    auto heurgen_method = [=](cudnn_frontend::OperationGraph& op_graph_)
        -> cudnn_frontend::EngineConfigList {
      cudnn_frontend::EngineConfigList filtered_configs;
      auto statuses = cudnn_frontend::get_heuristics_list<2>(
          {"heuristics_instant", "heuristics_fallback"},
          op_graph_,
          deterministic ? IsNonDeterministic : AllowAll,
          filtered_configs,
          true);
      VLOG(6) << "Filter config list has " << filtered_configs.size()
              << " configurations ";
      return filtered_configs;
    };

    std::array<cudnn_frontend::GeneratorSource const, 1> sources = {
        heurgen_method};
    cudnn_frontend::EngineConfigGenerator generator(sources.size(),
                                                    sources.data());

    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    auto predicate_function =
        [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
      return plan.getWorkspaceSize() > workspace_size_limit;
    };
    VLOG(6) << "[cudnn_frontend] Max workspace size: " << workspace_size_limit;
    cudnn_frontend::executionPlans_t plans;
    bool use_autotune = phi::autotune::AutoTuneStatus::Instance().UseAutoTune();

    if (!deterministic && (exhaustive_search || use_autotune)) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            auto variant_pack =
                cudnn_frontend::VariantPackBuilder()
                    .setWorkspacePointer(workspace_ptr)
                    .setDataPointers(data_ptrs->size(), data_ptrs->data())
                    .setUids(uids->size(), uids->data())
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
          workspace_size_limit);
    } else {
      plans =
          generator.cudnnGetPlan(handle, *op_graph_pointer, predicate_function);
    }

    std::for_each(
        plans.begin(), plans.end(), [](cudnn_frontend::ExecutionPlan& opt) {
          VLOG(6) << "Plan tag: " << opt.getTag() << " finished in "
                  << opt.getExecutionTime() << " ms,"
                  << " workspace: " << opt.getWorkspaceSize() << " bytes";
        });

    return plans;
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
    std::vector<void*> data_ptrs({x_data, y_data, w_data});
    std::vector<int64_t> uids({'x', 'y', 'w'});
    return FindExecutionPlans(op_graph_pointer,
                              exhaustive_search,
                              deterministic,
                              &data_ptrs,
                              &uids,
                              handle,
                              workspace_handle);
  }

  static void ExecutePlan(cudnnHandle_t handle_,
                          phi::DnnWorkspaceHandle* workspace_handle,
                          std::vector<void*>* data_ptrs,
                          std::vector<int64_t>* uids,
                          cudnnBackendDescriptor_t plan_desc,
                          int64_t workspace_size) {
    workspace_handle->RunFunc(
        [&](void* workspace_ptr) {
          auto variant_pack =
              cudnn_frontend::VariantPackBuilder()
                  .setWorkspacePointer(workspace_ptr)
                  .setDataPointers(data_ptrs->size(), data_ptrs->data())
                  .setUids(uids->size(), uids->data())
                  .build();
          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendExecute(
              handle_, plan_desc, variant_pack.get_raw_desc()));
        },
        workspace_size);
  }

  static void ExecutePlan(cudnnHandle_t handle_,
                          phi::DnnWorkspaceHandle* workspace_handle,
                          void* x_data,
                          void* y_data,
                          void* w_data,
                          cudnnBackendDescriptor_t plan_desc,
                          int64_t workspace_size) {
    std::vector<void*> data_ptrs({x_data, y_data, w_data});
    std::vector<int64_t> uids({'x', 'y', 'w'});
    ExecutePlan(handle_,
                workspace_handle,
                &data_ptrs,
                &uids,
                plan_desc,
                workspace_size);
  }

  static void ExecutePlansAndCache(
      cudnnHandle_t handle_,
      phi::DnnWorkspaceHandle* workspace_handle,
      std::vector<void*>* data_ptrs,
      std::vector<int64_t>* uids,
      cudnn_frontend::executionPlans_t* plans,
      bool exhaustive_search,
      const cudnn_frontend::feature_vector_t& feature_vector,
      phi::autotune::CudnnFrontendPlanCache* plan_cache) {
    for (auto& plan : *plans) {
      try {
        ExecutePlan(handle_,
                    workspace_handle,
                    data_ptrs,
                    uids,
                    plan.get_raw_desc(),
                    plan.getWorkspaceSize());
        if (!exhaustive_search ||
            plan_cache->IsStable(feature_vector, plan.getTag(), handle_)) {
          plan_cache->InsertPlan(feature_vector, plan, handle_);
        }
        return;
      } catch (cudnn_frontend::cudnnException& e) {
        VLOG(4) << "Plan " << plan.describe()
                << "failed to execute. Trying next plan.";
      } catch (common::enforce::EnforceNotMet& e) {
        VLOG(4) << "Plan " << plan.describe()
                << "failed to execute. Trying next plan.";
      }
    }
    PADDLE_THROW(phi::errors::InvalidArgument(
        "[CUDNN Frontend API] No valid plan could "
        "be found to execute. Try setting FLAGS_conv_workspace_size_limit "
        "higher."));
  }

  static void ExecutePlansAndCache(
      cudnnHandle_t handle_,
      phi::DnnWorkspaceHandle* workspace_handle,
      void* x_data,
      void* y_data,
      void* w_data,
      cudnn_frontend::executionPlans_t* plans,
      bool exhaustive_search,
      const cudnn_frontend::OperationGraph& op_graph,
      phi::autotune::CudnnFrontendPlanCache* plan_cache) {
    std::vector<void*> data_ptrs({x_data, y_data, w_data});
    std::vector<int64_t> uids({'x', 'y', 'w'});
    ExecutePlansAndCache(handle_,
                         workspace_handle,
                         &data_ptrs,
                         &uids,
                         plans,
                         exhaustive_search,
                         op_graph.getFeatureVector(),
                         plan_cache);
  }

  static void QueryCacheAndExecute(
      cudnnHandle_t handle,
      phi::DnnWorkspaceHandle* workspace_handle,
      cudnn_frontend::OperationGraph* op_graph_pointer,
      std::vector<void*>* data_ptrs,
      std::vector<int64_t>* uids,
      bool exhaustive_search,
      bool deterministic,
      const cudnn_frontend::feature_vector_t& feature_vector,
      phi::autotune::CudnnFrontendPlanCache* plan_cache) {
    if (plan_cache->FindPlan(feature_vector, handle)) {
      const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
      int64_t workspace_size = 0;
      plan_cache->GetPlanAndWorkspaceSize(
          feature_vector, &cached_plan, &workspace_size, handle);
      ExecutePlan(handle,
                  workspace_handle,
                  data_ptrs,
                  uids,
                  cached_plan->get_raw_desc(),
                  workspace_size);
      return;
    }

    auto plans = FindExecutionPlans(op_graph_pointer,
                                    exhaustive_search,
                                    deterministic,
                                    data_ptrs,
                                    uids,
                                    handle,
                                    workspace_handle);

    ExecutePlansAndCache(handle,
                         workspace_handle,
                         data_ptrs,
                         uids,
                         &plans,
                         exhaustive_search,
                         feature_vector,
                         plan_cache);
  }

  static cudnn_frontend::Operation MakePointwiseOp(
      cudnnPointwiseMode_t mode,
      cudnnDataType_t dtype,
      cudnn_frontend::Tensor const& x_desc,
      cudnn_frontend::Tensor const& b_desc,
      cudnn_frontend::Tensor const& y_desc,
      float alpha1 = 1.0,
      float alpha2 = 1.0) {
    auto op_desc = cudnn_frontend::PointWiseDescBuilder()
                       .setMode(mode)
                       .setComputeType(dtype)
                       .build();
    auto op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                  .setxDesc(x_desc)
                  .setbDesc(b_desc)
                  .setyDesc(y_desc)
                  .setpwDesc(op_desc)
                  .setAlpha(alpha1)
                  .setAlpha2(alpha2)
                  .build();
    VLOG(6) << op.describe();
    return op;
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

  if (plan_cache_bwd_data.FindPlan(op_graph, handle)) {
    const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
    int64_t workspace_size = 0;
    plan_cache_bwd_data.GetPlanAndWorkspaceSize(
        op_graph, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        workspace_handle,
                        dx_tensor_data,
                        dy_tensor_data,
                        w_tensor_data,
                        cached_plan->get_raw_desc(),
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

  helper::ExecutePlansAndCache(handle,
                               workspace_handle,
                               dx_tensor_data,
                               dy_tensor_data,
                               w_tensor_data,
                               &plans,
                               exhaustive_search,
                               op_graph,
                               &plan_cache_bwd_data);
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

  if (plan_cache_bwd_filter.FindPlan(op_graph, handle)) {
    const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
    int64_t workspace_size = 0;
    plan_cache_bwd_filter.GetPlanAndWorkspaceSize(
        op_graph, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        workspace_handle,
                        x_tensor_data,
                        dy_tensor_data,
                        dw_tensor_data,
                        cached_plan->get_raw_desc(),
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

  helper::ExecutePlansAndCache(handle,
                               workspace_handle,
                               x_tensor_data,
                               dy_tensor_data,
                               dw_tensor_data,
                               &plans,
                               exhaustive_search,
                               op_graph,
                               &plan_cache_bwd_filter);
}

}  // namespace phi
