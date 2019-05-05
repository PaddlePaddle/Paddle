/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/operator_kernel_configs.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/platform/cudnn_desc.h"

namespace paddle {
namespace operators {

using framework::AlgorithmsCache;

struct ConvArgs {
  cudnnHandle_t handle;
  platform::TensorDescriptor idesc, odesc;
  platform::FilterDescriptor wdesc;
  platform::ConvolutionDescriptor cdesc;
  const framework::Tensor *x, *w, *o;

  // strides
  std::vector<int> s;
  // paddings
  std::vector<int> p;
  // dilations
  std::vector<int> d;

  ConvArgs(const framework::Tensor* x, const framework::Tensor* w,
           const framework::Tensor* o, const std::vector<int> s,
           const std::vector<int> p, const std::vector<int> d)
      : x(x), w(w), o(o), s(s), p(p), d(d) {}
};

template <typename perf_t>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  template <typename T>
  static algo_t find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, int algo_cache_id,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool exhaustive = (exhaustive_search) & (dtype != CUDNN_DATA_HALF);

    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;

    algo_t algo;
    if (!exhaustive) {
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          args.handle, args.idesc.desc(), args.wdesc.desc(), args.cdesc.desc(),
          args.odesc.desc(), CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &algo));
      VLOG(3) << "choose algo " << algo;
    } else {
      AlgorithmsCache<algo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<algo_t>>(algo_cache_id);
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0, [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(
                  platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                      args.handle, args.idesc.desc(), args.x->data<T>(),
                      args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                      args.odesc.desc(), const_cast<T*>(args.o->data<T>()),
                      kNUM_CUDNN_FWD_ALGS, &returned_algo_count,
                      perf_stat.data(), cudnn_workspace_ptr,
                      workspace_size_limit));
            };
            workspace_handle.RunFunc(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "FwdAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
        args.handle, args.idesc.desc(), args.wdesc.desc(), args.cdesc.desc(),
        args.odesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  template <typename T>
  static algo_t find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, int algo_cache_id,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool exhaustive = (exhaustive_search) & (dtype != CUDNN_DATA_HALF);

    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;

    algo_t algo;
    if (!exhaustive && !deterministic) {
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
          args.handle, args.wdesc.desc(), args.idesc.desc(), args.cdesc.desc(),
          args.odesc.desc(), CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &algo));
    } else if (deterministic) {
      return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else {
      AlgorithmsCache<algo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<algo_t>>(algo_cache_id);
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0, [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(
                  platform::dynload::
                      cudnnFindConvolutionBackwardDataAlgorithmEx(
                          args.handle, args.wdesc.desc(), args.w->data<T>(),
                          args.odesc.desc(), args.o->data<T>(),
                          args.cdesc.desc(), args.idesc.desc(),
                          const_cast<T*>(args.x->data<T>()),
                          kNUM_CUDNN_BWD_DATA_ALGS, &returned_algo_count,
                          perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
            };
            workspace_handle.RunFunc(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "BwdDataAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }

            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    CUDNN_ENFORCE(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle, args.wdesc.desc(), args.idesc.desc(),
            args.cdesc.desc(), args.odesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  template <typename T>
  static algo_t find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, int algo_cache_id,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool exhaustive = (exhaustive_search) & (dtype != CUDNN_DATA_HALF);

    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;

    algo_t algo;
    if (!exhaustive && !deterministic) {
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(),
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
    } else if (deterministic) {
      return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else {
      AlgorithmsCache<algo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<algo_t>>(algo_cache_id);
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0, [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;
            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(
                  platform::dynload::
                      cudnnFindConvolutionBackwardFilterAlgorithmEx(
                          args.handle, args.idesc.desc(), args.x->data<T>(),
                          args.odesc.desc(), args.o->data<T>(),
                          args.cdesc.desc(), args.wdesc.desc(),
                          const_cast<T*>(args.w->data<T>()),
                          kNUM_CUDNN_BWD_FILTER_ALGS, &returned_algo_count,
                          perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
            };
            workspace_handle.RunFunc(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "BwdFilterAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    CUDNN_ENFORCE(
        platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

}  // namespace operators
}  // namespace paddle
