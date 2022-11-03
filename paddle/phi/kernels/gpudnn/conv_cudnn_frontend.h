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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/device/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_v7.h"

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

  static void GenerateStrides(const int64_t* dimA,
                              int64_t* strideA,
                              int nbDims,
                              cudnnTensorFormat_t filter_format) {
    // ref:
    // https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/helpers.cpp
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the
    // cpu ref.
    if (filter_format == CUDNN_TENSOR_NCHW) {
      strideA[nbDims - 1] = 1;
      for (int64_t d = nbDims - 2; d >= 0; d--) {
        strideA[d] = strideA[d + 1] * dimA[d + 1];
      }
    } else {
      // Here we assume that the format is CUDNN_TENSOR_NHWC
      strideA[1] = 1;
      strideA[nbDims - 1] = strideA[1] * dimA[1];
      for (int64_t d = nbDims - 2; d >= 2; d--) {
        strideA[d] = strideA[d + 1] * dimA[d + 1];
      }
      strideA[0] = strideA[2] * dimA[2];
    }
  }

  static cudnn_frontend::Tensor GetTensorDescriptor(
      const phi::DenseTensor* tensor,
      int64_t id,
      cudnnTensorFormat_t layout_format) {
    auto dims = phi::vectorize<int>(tensor->dims());
    std::vector<int64_t> transformed_dims;
    if (layout_format == CUDNN_TENSOR_NHWC) {
      transformed_dims =
          GetInt64Array(paddle::platform::TransformDimOrder(dims));
    } else {
      transformed_dims = GetInt64Array(dims);
    }
    std::vector<int64_t> strides(dims.size());
    GenerateStrides(transformed_dims.data(),
                    strides.data(),
                    transformed_dims.size(),
                    layout_format);
    return cudnn_frontend::TensorBuilder()
        .setDim(transformed_dims.size(), transformed_dims.data())
        .setStrides(strides.size(), strides.data())
        .setId(id)
        .setAlignment(GetAlignment(tensor))
        .setDataType(paddle::platform::ToCudnnDataType(
            paddle::framework::TransToProtoVarType(tensor->dtype())))
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

}  // namespace phi
