/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace paddle {
namespace operators {

using ConvArgs = ConvArgsBase<cudnnHandle_t, cudnnDataType_t>;

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const phi::GPUContext& context,
                               const Tensor* input,
                               Tensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place = *context.eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
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

  phi::funcs::EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_t, in_t, offsets, extents);
}

static inline double ToMegaBytes(size_t bytes) {
  return static_cast<double>(bytes) / (1 << 20);
}

static inline bool UseFixedWorkspace() {
  return FLAGS_conv_workspace_size_limit >= 0;
}

static size_t CalcWorkspaceLimitInBytes(bool use_fixed_workspace) {
  if (!use_fixed_workspace) {
    int device_id = platform::GetCurrentDeviceId();
    int64_t allocated =
        memory::DeviceMemoryStatCurrentValue("Allocated", device_id);
    int64_t reserved =
        memory::DeviceMemoryStatCurrentValue("Reserved", device_id);
    int64_t availble = platform::GpuAvailableMemToAlloc();
    VLOG(3) << "[memory] allocated=" << ToMegaBytes(allocated)
            << " MB, reserved=" << ToMegaBytes(reserved)
            << " MB, available_to_alloc=" << ToMegaBytes(availble) << " MB.";
    return std::max(availble, reserved - allocated);
  } else {
    return FLAGS_conv_workspace_size_limit * 1024 * 1024;
  }
}

template <typename PerfT>
std::string GetPerfResultString(std::string prefix,
                                const std::vector<PerfT>& perf_results,
                                int actual_algo_count,
                                size_t workspace_limit) {
  std::ostringstream out;
  out << prefix << " (workspace limit=" << ToMegaBytes(workspace_limit)
      << " MB):\n";
  for (int i = 0; i < actual_algo_count; ++i) {
    const auto& result = perf_results[i];
    auto math_type_str = (result.mathType == CUDNN_TENSOR_OP_MATH) ? "T" : "F";
    out << "  algo=" << result.algo << ": tensor_core=" << math_type_str
        << ", time=" << result.time
        << " ms, memory=" << ToMegaBytes(result.memory)
        << " MB, status=" << result.status << "\n";
  }
  return out.str();
}

// Choose an algorithm which has the minimize time cost and less memory.
// NOTE: perf_results is ordered by time.
template <typename PerfT, typename AlgoT>
void ChooseAlgoByWorkspace(const std::vector<PerfT>& perf_results,
                           size_t workspace_limit,
                           SearchResult<AlgoT>* search_result) {
  int best_algo_idx = -1;
  for (size_t i = 0; i < perf_results.size(); ++i) {
    auto result = perf_results[i];
    if (result.status == CUDNN_STATUS_SUCCESS &&
        result.memory < workspace_limit) {
      if (best_algo_idx == -1) {
        // The algorithm which has minimize time cost and need a workspace_size
        // fitting the workspace_limit constraint.
        best_algo_idx = i;
        // Each perf_results[i].time is set to be -1 in heuristic search.
        if (perf_results[best_algo_idx].time < 0) {
          break;
        }
      } else {
        float best_algo_time = perf_results[best_algo_idx].time;
        if ((result.time - best_algo_time) / best_algo_time < 0.01) {
          best_algo_idx = (result.memory < perf_results[best_algo_idx].memory)
                              ? i
                              : best_algo_idx;
          break;
        }
      }
    }
  }
  if (best_algo_idx != -1) {
    search_result->algo = perf_results[best_algo_idx].algo;
    search_result->time = perf_results[best_algo_idx].time;
    search_result->workspace_size = perf_results[best_algo_idx].memory;
  } else {
    VLOG(3) << "Can not find an algorithm that requires memory < "
            << ToMegaBytes(workspace_limit) << " MB";
  }
}

template <typename PerfT>
struct SearchAlgorithmBase {};

// cuDNN convolution forward algorithm searcher, consisted of three searching
// modes, namely: deterministic, heuristic and exhaustive_search mode.
// As well as one workspace size acquirsition function with respect to
// the chosen alogrithm.
template <>
struct SearchAlgorithmBase<cudnnConvolutionFwdAlgoPerf_t> {
  using PerfT = cudnnConvolutionFwdAlgoPerf_t;
  using AlgoT = cudnnConvolutionFwdAlgo_t;
  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvForward;

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionFwdAlgo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const ConvArgs& args) {
    auto workspace_size = GetWorkspaceSize(args, static_cast<AlgoT>(1));
    return SearchResult<AlgoT>(static_cast<AlgoT>(1), -1.0, workspace_size);
  }

  // Heuristic search mode, calling the cudnnGetXxxAlgorithm.
  static SearchResult<AlgoT> FindAlgoHeuristic(const ConvArgs& args,
                                               const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int actual_perf_count;
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_FWD_ALGS);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            kNUM_CUDNN_FWD_ALGS,
            &actual_perf_count,
            perf_results.data()));
    result.algo = perf_results[best_algo_idx].algo;
    result.workspace_size = perf_results[best_algo_idx].memory;

    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      VLOG(4) << GetPerfResultString<PerfT>("[Heuristic] FwdAlgo Perf result",
                                            perf_results,
                                            actual_perf_count,
                                            workspace_size_limit);
      // cudnnGetConvolutionForwardAlgorithm is removed in CUDNN-8
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(3) << "Fallback to non-v7 method to find conv algorithm "
                 "becasue the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.wdesc.desc(),
              args.cdesc.desc(),
              args.odesc.desc(),
              CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit,
              &(result.algo)));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardAlgorithm(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &(result.algo)));
#endif
    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const ConvArgs& args, const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    size_t max_workspace_size = GetMaxWorkspaceSize(args, workspace_size_limit);
    VLOG(4) << "max_workspace_size=" << ToMegaBytes(max_workspace_size)
            << " MB";

    int returned_algo_count;
    std::vector<PerfT> perf_results(kNUM_CUDNN_FWD_ALGS);
    auto cudnn_find_func = [&](void* workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
              args.handle,
              args.idesc.desc(),
              args.x->data<T>(),
              args.wdesc.desc(),
              args.w->data<T>(),
              args.cdesc.desc(),
              args.odesc.desc(),
              const_cast<T*>(args.o->data<T>()),
              kNUM_CUDNN_FWD_ALGS,
              &returned_algo_count,
              perf_results.data(),
              workspace_ptr,
              max_workspace_size));
    };

    auto workspace_handle = ctx.cudnn_workspace_handle();
    workspace_handle.RunFuncSync(
        cudnn_find_func, max_workspace_size, UseFixedWorkspace());

    VLOG(4) << GetPerfResultString<PerfT>(
        "[Exhaustive Search] FwdAlgo Perf result",
        perf_results,
        returned_algo_count,
        workspace_size_limit);
    ChooseAlgoByWorkspace<PerfT, AlgoT>(
        perf_results, workspace_size_limit, &result);

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_FWD_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status =
            platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
                args.handle,
                args.idesc.desc(),
                args.wdesc.desc(),
                args.cdesc.desc(),
                args.odesc.desc(),
                static_cast<cudnnConvolutionFwdAlgo_t>(algo),
                &workspace_size);
        if (status == CUDNN_STATUS_SUCCESS &&
            workspace_size <= workspace_size_limit) {
          max_workspace_size = std::max(workspace_size, max_workspace_size);
        }
      }
      return max_workspace_size;
    } else {
      return workspace_size_limit;
    }
  }
};

// cuDNN convolution backward data-algorithm searcher, consisting of three
// searching modes, namely: deterministic, heuristic, and exhaustive_search
// mode. Specially, there are 2 pattens of exhaustive search mode, one for
// HALF precision only, one for the rest.
// As well as one workspace size acquirsition function with
// respect to the chosen alogrithm.
template <>
struct SearchAlgorithmBase<cudnnConvolutionBwdDataAlgoPerf_t> {
  using PerfT = cudnnConvolutionBwdDataAlgoPerf_t;
  using AlgoT = cudnnConvolutionBwdDataAlgo_t;
  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvBackwardData;

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionBwdDataAlgo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const ConvArgs& args) {
    auto workspace_size =
        GetWorkspaceSize(args, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
    return SearchResult<AlgoT>(
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, -1.0, workspace_size);
  }

  static SearchResult<AlgoT> FindAlgoHeuristic(const ConvArgs& args,
                                               const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int actual_perf_count;
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_DATA_ALGS);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            kNUM_CUDNN_BWD_DATA_ALGS,
            &actual_perf_count,
            perf_results.data()));
    result.algo = perf_results[best_algo_idx].algo;

#if CUDNN_VERSION < 7500
    int stride_dim = args.x->dims().size() - 2;
    bool blacklist = std::any_of(args.s.begin(),
                                 args.s.begin() + stride_dim,
                                 [=](int n) { return n != 1; });
    if (blacklist && (perf_results[best_algo_idx].algo ==
                          CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                      perf_results[best_algo_idx].algo ==
                          CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
      result.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    }
#endif
    result.workspace_size = GetWorkspaceSize(args, result.algo);
    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionBackwardDataAlgorithm is removed in CUDNN-8
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                 "the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
              args.handle,
              args.wdesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.idesc.desc(),
              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit,
              &(result.algo)));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &(result.algo)));
#endif
    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const ConvArgs& args, const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    size_t max_workspace_size = GetMaxWorkspaceSize(args, workspace_size_limit);
    VLOG(3) << "max_workspace_size=" << ToMegaBytes(max_workspace_size)
            << " MB";

    int returned_algo_count;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_DATA_ALGS);
    auto cudnn_find_func = [&](void* workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnFindConvolutionBackwardDataAlgorithmEx(
              args.handle,
              args.wdesc.desc(),
              args.w->data<T>(),
              args.odesc.desc(),
              args.o->data<T>(),
              args.cdesc.desc(),
              args.idesc.desc(),
              const_cast<T*>(args.x->data<T>()),
              kNUM_CUDNN_BWD_DATA_ALGS,
              &returned_algo_count,
              perf_results.data(),
              workspace_ptr,
              max_workspace_size));
    };

    auto workspace_handle = ctx.cudnn_workspace_handle();
    workspace_handle.RunFuncSync(
        cudnn_find_func, max_workspace_size, UseFixedWorkspace());

    VLOG(4) << GetPerfResultString<PerfT>(
        "[Exhaustive Search] BwdDataAlgo Perf result",
        perf_results,
        returned_algo_count,
        workspace_size_limit);
    ChooseAlgoByWorkspace<PerfT, AlgoT>(
        perf_results, workspace_size_limit, &result);

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_BWD_DATA_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status =
            platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
                args.handle,
                args.wdesc.desc(),
                args.odesc.desc(),
                args.cdesc.desc(),
                args.idesc.desc(),
                static_cast<cudnnConvolutionBwdDataAlgo_t>(algo),
                &workspace_size);
        if (status == CUDNN_STATUS_SUCCESS &&
            workspace_size <= workspace_size_limit) {
          max_workspace_size = std::max(workspace_size, max_workspace_size);
        }
      }
      return max_workspace_size;
    } else {
      return workspace_size_limit;
    }
  }
};

// cuDNN convution backward filter-algorithm searcher, consisted of three
// algorithm searching modes, namely: deterministic, heuristic, and
// exhaustive_search mode. As well as one workspace size acquirsition function
// with respect to the chosen alogrithm.
template <>
struct SearchAlgorithmBase<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using PerfT = cudnnConvolutionBwdFilterAlgoPerf_t;
  using AlgoT = cudnnConvolutionBwdFilterAlgo_t;
  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvBackwardFilter;

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionBwdFilterAlgo_t algo) {
    platform::CUDAGraphCaptureModeGuard guard;
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const ConvArgs& args) {
    auto workspace_size =
        GetWorkspaceSize(args, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
    return SearchResult<AlgoT>(
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, -1.0, workspace_size);
  }

  static SearchResult<AlgoT> FindAlgoHeuristic(const ConvArgs& args,
                                               const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int actual_perf_count;
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_FILTER_ALGS);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            kNUM_CUDNN_BWD_FILTER_ALGS,
            &actual_perf_count,
            perf_results.data()));
    result.algo = perf_results[best_algo_idx].algo;
    result.workspace_size = perf_results[best_algo_idx].memory;

    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionBackwardFilterAlgorithm is removed in CUDNN-8
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                 "the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit,
              &(result.algo)));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &(result.algo)));
#endif

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const ConvArgs& args, const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    int returned_algo_count = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_FILTER_ALGS);
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    auto workspace_handle = ctx.cudnn_workspace_handle();
    if (platform::CudnnDataType<T>::type != CUDNN_DATA_HALF) {
      size_t max_workspace_size =
          GetMaxWorkspaceSize(args, workspace_size_limit);
      VLOG(3) << "max_workspace_size=" << ToMegaBytes(max_workspace_size)
              << " MB";

      auto cudnn_find_func = [&](void* workspace_ptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cudnnFindConvolutionBackwardFilterAlgorithmEx(
                args.handle,
                args.idesc.desc(),
                args.x->data<T>(),
                args.odesc.desc(),
                args.o->data<T>(),
                args.cdesc.desc(),
                args.wdesc.desc(),
                const_cast<T*>(args.w->data<T>()),
                kNUM_CUDNN_BWD_FILTER_ALGS,
                &returned_algo_count,
                perf_results.data(),
                workspace_ptr,
                max_workspace_size));
      };
      workspace_handle.RunFuncSync(
          cudnn_find_func, max_workspace_size, UseFixedWorkspace());

      VLOG(4) << GetPerfResultString<PerfT>(
          "[Exhaustive Search] BwdFilterAlgo Perf result",
          perf_results,
          returned_algo_count,
          workspace_size_limit);
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
    } else {
      int max_algos = GetAlgorithmMaxCount(args.handle);
      std::vector<PerfT> perf_results(max_algos);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnFindConvolutionBackwardFilterAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              perf_results.size(),
              &returned_algo_count,
              perf_results.data()));
      perf_results.resize(returned_algo_count);

      VLOG(4) << GetPerfResultString<PerfT>(
          "[Exhaustive Search] BwdFilterAlgo Perf result",
          perf_results,
          perf_results.size(),
          workspace_size_limit);
      ChooseAlgo(perf_results, workspace_size_limit, &result);
    }

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  static int GetAlgorithmMaxCount(cudnnHandle_t handle) {
#if CUDNN_VERSION_MIN(7, 0, 1)
    int max_algos = 0;
    auto status =
        platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
            handle, &max_algos);
    if (status == gpuSuccess) {
      VLOG(5) << "[BackwardFilter] max_algos: predefined="
              << kNUM_CUDNN_BWD_FILTER_ALGS << ", actual=" << max_algos;
      return max_algos;
    }
#endif
    return kNUM_CUDNN_BWD_FILTER_ALGS;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_BWD_FILTER_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status =
            platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
                args.handle,
                args.idesc.desc(),
                args.odesc.desc(),
                args.cdesc.desc(),
                args.wdesc.desc(),
                static_cast<cudnnConvolutionBwdFilterAlgo_t>(algo),
                &workspace_size);
        if (status == CUDNN_STATUS_SUCCESS &&
            workspace_size <= workspace_size_limit) {
          max_workspace_size = std::max(workspace_size, max_workspace_size);
        }
      }
      return max_workspace_size;
    } else {
      return workspace_size_limit;
    }
  }

  static void ChooseAlgo(const std::vector<PerfT>& perf_results,
                         size_t workspace_limit,
                         SearchResult<AlgoT>* algo_result) {
    for (size_t i = 0; i != perf_results.size(); ++i) {
      const auto& result = perf_results[i];
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (result.memory <= workspace_limit)) {
        if ((result.mathType == CUDNN_TENSOR_OP_MATH) &&
            (i != perf_results.size() - 1)) {
          const auto& next_result = perf_results[i + 1];
          if (next_result.status == CUDNN_STATUS_SUCCESS &&
              next_result.algo == result.algo &&
              next_result.memory == result.memory &&
              next_result.mathType != CUDNN_TENSOR_OP_MATH &&
              next_result.time < 1.01 * result.time) {
            // Skip over this result- it's not really a Tensor Core algo.
            // Because it is only 1% performance difference.
            // Prefer to choose the next equivalent non-Tensor Core algo.
            continue;
          }
        }
        algo_result->algo = result.algo;
        algo_result->time = result.time;
        auto math_type_str = "0";
        if (result.mathType == CUDNN_TENSOR_OP_MATH) {
          math_type_str = "1";
        }
        VLOG(3) << "    choose algo: " << result.algo
                << ", TC: " << math_type_str << ", time: " << result.time
                << " ms, wksp = " << result.memory
                << ", status = " << result.status;
        break;
      }
    }
  }
};

template <typename PerfT>
struct SearchAlgorithm : public SearchAlgorithmBase<PerfT> {
  using AlgoT = typename SearchAlgorithmBase<PerfT>::AlgoT;

  template <typename T>
  static SearchResult<AlgoT> Find(const ConvArgs& args,
                                  bool exhaustive_search,
                                  bool deterministic,
                                  const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    auto dtype = platform::CudnnDataType<T>::type;
    SetConvMathType(ctx, dtype, args.cdesc);

    if (deterministic) {
      result = SearchAlgorithmBase<PerfT>::FindAlgoDeterministic(args);
    } else {
      // 1. Once turning on exhaustive FLAGS, always get exhaustive_search.
      // 2. Once turning on auto-tune, runn heuristic search(default) before
      //    auto-tune process, run exhaustive_search during mentioned process.
      // 3. After auto-tune process, run cached algorithm if cached, run
      //    default mode for the rest.
      auto key = args.Convert2ConvCacheKey<T>();
      auto& cache = phi::autotune::AutoTuneCache::Instance().GetConv(
          SearchAlgorithmBase<PerfT>::kAlgoType);
      if (cache.Find(key)) {
        auto t = cache.Get(key);
        result.algo = static_cast<AlgoT>(t.algo);
        result.workspace_size = t.workspace_size;
      } else {
        bool use_autotune =
            phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
        if (exhaustive_search || use_autotune) {
          result =
              SearchAlgorithmBase<PerfT>::template FindAlgoExhaustiveSearch<T>(
                  args, ctx);
        } else {
          result = SearchAlgorithmBase<PerfT>::FindAlgoHeuristic(args, ctx);
        }
        phi::autotune::DnnNode node(static_cast<int64_t>(result.algo),
                                    result.workspace_size);
        cache.Set(key, node);
      }
    }
    VLOG(3) << "[cuDNN Convoltion] exhaustive_search=" << exhaustive_search
            << ", deterministic=" << deterministic
            << ", choose algo=" << result.algo
            << ", workspace=" << ToMegaBytes(result.workspace_size) << " MB";
    return result;
  }

  static void SetConvMathType(const phi::GPUContext& ctx,
                              cudnnDataType_t dtype,
                              const platform::ConvolutionDescriptor& cdesc) {
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    if (ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "Enable Tensor Core for FLOAT16";
#if CUDA_VERSION >= 11000
#if CUDNN_VERSION_MIN(8, 1, 0)
    } else if (ctx.GetComputeCapability() >= 80 &&
               dtype == CUDNN_DATA_BFLOAT16) {
      VLOG(5) << "Enable Tensor Core for BFLOAT16";
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_TENSOR_OP_MATH));
#endif  // CUDNN_VERSION_MIN(8, 1, 0)
    } else if (dtype == CUDNN_DATA_FLOAT && !cdesc.allow_tf32_) {
      VLOG(5) << "Disable TensorFloat (Tensor Core) for FLOAT";
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_FMA_MATH));
#endif  // CUDA_VERSION >= 11000
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_DEFAULT_MATH));
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle
