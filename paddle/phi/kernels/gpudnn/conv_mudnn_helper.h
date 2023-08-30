// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <mudnn.h>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/conv_kernel.h"
#include "paddle/phi/kernels/gpudnn/conv_gpudnn_base.h"

namespace phi {

using ConvArgs = ConvArgsBase<dynload::Handle*, dynload::Tensor::Type>;

template <typename PerfT>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<dynload::Convolution::Algorithm> {
  using algo_t = dynload::Convolution::Algorithm;

  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    auto mudnn_find_func = [&](void* mudnn_workspace_ptr) {
      args.cdesc.desc()->GetRecommendForwardAlgorithm(*args.handle,
                                                      algo,
                                                      *args.odesc.desc(),
                                                      *args.idesc.desc(),
                                                      *args.wdesc.desc());
    };

    workspace_handle.RunFuncSync(mudnn_find_func, workspace_size);
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) { return 0; }
};

template <>
struct SearchAlgorithm<dynload::Convolution::AlgorithmBwdData> {
  using algo_t = dynload::Convolution::AlgorithmBwdData;

  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    auto mudnn_find_func = [&](void* mudnn_workspace_ptr) {
      args.cdesc.desc()->GetRecommendBackwardDataAlgorithm(*args.handle,
                                                           algo,
                                                           *args.idesc.desc(),
                                                           *args.odesc.desc(),
                                                           *args.wdesc.desc());
    };

    workspace_handle.RunFuncSync(mudnn_find_func, workspace_size);
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) { return 0; }
};

template <>
struct SearchAlgorithm<dynload::Convolution::AlgorithmBwdFilter> {
  using algo_t = dynload::Convolution::AlgorithmBwdFilter;

  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    auto mudnn_find_func = [&](void* mudnn_workspace_ptr) {
      args.cdesc.desc()->GetRecommendBackwardFilterAlgorithm(
          *args.handle,
          algo,
          *args.wdesc.desc(),
          *args.idesc.desc(),
          *args.odesc.desc());
    };

    workspace_handle.RunFuncSync(mudnn_find_func, workspace_size);
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) { return 0; }
};

using backends::gpu::InternalMemAlloc;

}  // namespace phi
