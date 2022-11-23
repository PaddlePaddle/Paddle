/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/conv_base_helper.h"

namespace paddle {
namespace operators {

using ConvArgs = ConvArgsBase<miopenHandle_t, miopenDataType_t>;

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const phi::GPUContext& context,
                               const phi::DenseTensor* input,
                               phi::DenseTensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place = *context.eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    int start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);

  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out, new_out_dims);
  out_t.device(place) = in_t.slice(offsets, extents);
}

template <typename PerfT>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<miopenConvFwdAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvFwdAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    int find_count;
    miopenConvAlgoPerf_t find_result;
    auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::miopenFindConvolutionForwardAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.x->data<T>(),
              args.wdesc.desc(),
              args.w->data<T>(),
              args.cdesc.desc(),
              args.odesc.desc(),
              const_cast<T*>(args.o->data<T>()),
              kNUM_CUDNN_FWD_ALGS,
              &find_count,
              &find_result,
              cudnn_workspace_ptr,
              workspace_size,
              false));
    };

    workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
    algo = find_result.fwd_algo;
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenConvolutionForwardGetWorkSpaceSize(
            args.handle,
            args.wdesc.desc(),
            args.idesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<miopenConvBwdDataAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    int find_count;
    miopenConvAlgoPerf_t find_result;
    auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::miopenFindConvolutionBackwardDataAlgorithm(
              args.handle,
              args.odesc.desc(),
              args.o->data<T>(),
              args.wdesc.desc(),
              args.w->data<T>(),
              args.cdesc.desc(),
              args.idesc.desc(),
              const_cast<T*>(args.x->data<T>()),
              kNUM_CUDNN_BWD_DATA_ALGS,
              &find_count,
              &find_result,
              cudnn_workspace_ptr,
              workspace_size,
              false));
    };

    workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
    algo = find_result.bwd_data_algo;
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
            args.handle,
            args.odesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdWeightsAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    int find_count;
    miopenConvAlgoPerf_t find_result;
    auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::miopenFindConvolutionBackwardWeightsAlgorithm(
              args.handle,
              args.odesc.desc(),
              args.o->data<T>(),
              args.idesc.desc(),
              args.x->data<T>(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              const_cast<T*>(args.w->data<T>()),
              kNUM_CUDNN_BWD_FILTER_ALGS,
              &find_count,
              &find_result,
              cudnn_workspace_ptr,
              workspace_size,
              false));
    };

    workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
    algo = find_result.bwd_weights_algo;
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            args.handle,
            args.odesc.desc(),
            args.idesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            &workspace_size));
    return workspace_size;
  }
};

}  // namespace operators
}  // namespace paddle
