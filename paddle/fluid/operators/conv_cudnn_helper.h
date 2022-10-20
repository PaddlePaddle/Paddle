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

#include "paddle/fluid/operators/conv_base_helper.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace paddle {
namespace operators {

using ConvArgs = ConvArgsBase<cudnnHandle_t, cudnnDataType_t>;

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const phi::GPUContext& context,
                               const phi::DenseTensor* input,
                               phi::DenseTensor* out,
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
      phi::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(*input);
  auto out_t = phi::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
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
    const auto& result = perf_results[i];
    if (result.status == CUDNN_STATUS_SUCCESS &&
        result.memory <= workspace_limit) {
      if (best_algo_idx == -1) {
        // The algorithm which has minimize time cost and need a workspace_size
        // fitting the workspace_limit constraint.
        best_algo_idx = i;
        // Each perf_results[i].time is set to be -1 in heuristic search.
        if (perf_results[best_algo_idx].time < 0) {
          break;
        }
      } else {
        // Compared to the next suboptimal algorithm, if the best one only has
        // 1% performance difference, we'd like to pick the one which need less
        // memory.
        if (result.time < 1.01 * perf_results[best_algo_idx].time) {
          best_algo_idx = (result.memory < perf_results[best_algo_idx].memory)
                              ? i
                              : best_algo_idx;
          break;
        }
      }
    }
  }
  if (best_algo_idx != -1) {
    const auto& result = perf_results[best_algo_idx];
    search_result->algo = result.algo;
    search_result->time = result.time;
    search_result->use_tensor_op_math = result.mathType == CUDNN_TENSOR_OP_MATH;
    search_result->workspace_size = result.memory;
    auto math_type_str = search_result->use_tensor_op_math ? "T" : "F";
    VLOG(3) << "Choose algo=" << result.algo
            << ", tensor_core=" << math_type_str << ", time=" << result.time
            << " ms, memory=" << ToMegaBytes(result.memory)
            << " MB, status=" << result.status;
  } else {
    VLOG(3) << "Can not find an algorithm that requires memory < "
            << ToMegaBytes(workspace_limit) << " MB";
  }
}

static void SetConvolutionMathType(const phi::GPUContext& ctx,
                                   const platform::ConvolutionDescriptor& cdesc,
                                   cudnnDataType_t dtype,
                                   bool use_tensor_op_math) {
  int compute_capability = ctx.GetComputeCapability();
  cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;

  if (use_tensor_op_math) {
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    if (dtype == CUDNN_DATA_HALF && compute_capability >= 70) {
      VLOG(5) << "Enable Tensor Core for FLOAT16";
      math_type = CUDNN_TENSOR_OP_MATH;
#if CUDA_VERSION >= 11000 && CUDNN_VERSION_MIN(8, 1, 0)
    } else if (dtype == CUDNN_DATA_BFLOAT16 && compute_capability >= 80) {
      VLOG(5) << "Enable Tensor Core for BFLOAT16";
      math_type = CUDNN_TENSOR_OP_MATH;
#endif  // CUDA_VERSION >= 11000 && CUDNN_VERSION_MIN(8, 1, 0)
    }
#endif  // CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
  }

#if CUDA_VERSION >= 11000
  if (dtype == CUDNN_DATA_FLOAT && compute_capability >= 80 &&
      !cdesc.allow_tf32_) {
    VLOG(5) << "Disable TensorFloat (Tensor Core) for FLOAT";
    math_type = CUDNN_FMA_MATH;
  }
#endif  // CUDA_VERSION >= 11000

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnSetConvolutionMathType(cdesc.desc(), math_type));
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

  constexpr static int kNumAlgorithms = kNUM_CUDNN_FWD_ALGS;
  constexpr static AlgoT kDeterministicAlgo = static_cast<AlgoT>(1);
  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvForward;

  static const std::string GetPerfName() { return "ConvForward"; }

  static size_t GetWorkspaceSize(const phi::GPUContext& ctx,
                                 const ConvArgs& args,
                                 const SearchResult<AlgoT>& search_result) {
    size_t workspace_size = 0;
    SetConvolutionMathType(
        ctx, args.cdesc, args.cudnn_dtype, search_result.use_tensor_op_math);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            search_result.algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static int GetConvolutionAlgorithmV7(const ConvArgs& args,
                                       std::vector<PerfT>* perf_results) {
    int actual_perf_count = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            kNUM_CUDNN_FWD_ALGS,
            &actual_perf_count,
            perf_results->data()));
    return actual_perf_count;
  }

#if CUDNN_VERSION < 8000
  static int GetConvolutionAlgorithm(const ConvArgs& args,
                                     int64_t workspace_size_limit) {
    int algo = -1;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionForwardAlgorithm(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &algo));
    return algo;
  }
#endif

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const phi::GPUContext& ctx, const ConvArgs& args) {
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
          phi::dynload::cudnnFindConvolutionForwardAlgorithmEx(
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

    result.workspace_size = GetWorkspaceSize(ctx, args, result);
    return result;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_FWD_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status = phi::dynload::cudnnGetConvolutionForwardWorkspaceSize(
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

  constexpr static int kNumAlgorithms = kNUM_CUDNN_BWD_DATA_ALGS;
  constexpr static AlgoT kDeterministicAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvBackwardData;

  static const std::string GetPerfName() { return "ConvBackwardData"; }

  static size_t GetWorkspaceSize(const phi::GPUContext& ctx,
                                 const ConvArgs& args,
                                 const SearchResult<AlgoT>& search_result) {
    size_t workspace_size = 0;
    SetConvolutionMathType(
        ctx, args.cdesc, args.cudnn_dtype, search_result.use_tensor_op_math);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            search_result.algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static int GetConvolutionAlgorithmV7(const ConvArgs& args,
                                       std::vector<PerfT>* perf_results) {
    int actual_perf_count = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            kNUM_CUDNN_BWD_DATA_ALGS,
            &actual_perf_count,
            perf_results->data()));
    return actual_perf_count;
  }

#if CUDNN_VERSION < 8000
  static int GetConvolutionAlgorithm(const ConvArgs& args,
                                     int64_t workspace_size_limit) {
    int algo = -1;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &algo));
    return algo;
  }
#endif

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const phi::GPUContext& ctx, const ConvArgs& args) {
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
          phi::dynload::cudnnFindConvolutionBackwardDataAlgorithmEx(
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

    result.workspace_size = GetWorkspaceSize(ctx, args, result);
    return result;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_BWD_DATA_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status =
            phi::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
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

  constexpr static int kNumAlgorithms = kNUM_CUDNN_BWD_FILTER_ALGS;
  constexpr static AlgoT kDeterministicAlgo =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvBackwardFilter;

  static const std::string GetPerfName() { return "ConvBackwardFilter"; }

  static size_t GetWorkspaceSize(const phi::GPUContext& ctx,
                                 const ConvArgs& args,
                                 const SearchResult<AlgoT>& search_result) {
    platform::CUDAGraphCaptureModeGuard guard;
    size_t workspace_size = 0;
    SetConvolutionMathType(
        ctx, args.cdesc, args.cudnn_dtype, search_result.use_tensor_op_math);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            search_result.algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static int GetConvolutionAlgorithmV7(const ConvArgs& args,
                                       std::vector<PerfT>* perf_results) {
    int actual_perf_count = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            kNUM_CUDNN_BWD_FILTER_ALGS,
            &actual_perf_count,
            perf_results->data()));
    return actual_perf_count;
  }

#if CUDNN_VERSION < 8000
  static int GetConvolutionAlgorithm(const ConvArgs& args,
                                     int64_t workspace_size_limit) {
    int algo = -1;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &algo));
    return algo;
  }
#endif

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const phi::GPUContext& ctx, const ConvArgs& args) {
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
            phi::dynload::cudnnFindConvolutionBackwardFilterAlgorithmEx(
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
          phi::dynload::cudnnFindConvolutionBackwardFilterAlgorithm(
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
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
    }

    result.workspace_size = GetWorkspaceSize(ctx, args, result);
    return result;
  }

  static int GetAlgorithmMaxCount(cudnnHandle_t handle) {
#if CUDNN_VERSION_MIN(7, 0, 1)
    int max_algos = 0;
    auto status =
        phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
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
            phi::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
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
};

template <typename PerfT>
struct SearchAlgorithm : public SearchAlgorithmBase<PerfT> {
  using AlgoT = typename SearchAlgorithmBase<PerfT>::AlgoT;

  template <typename T>
  static SearchResult<AlgoT> Find(const phi::GPUContext& ctx,
                                  const ConvArgs& args,
                                  bool exhaustive_search,
                                  bool deterministic,
                                  bool enable_autotune = true) {
    SearchResult<AlgoT> result;
    bool use_autotune = false;
    auto dtype = platform::CudnnDataType<T>::type;
    SetConvolutionMathType(ctx, args.cdesc, dtype, true);

    if (deterministic) {
      result = FindAlgoDeterministic(ctx, args);
    } else {
      // 1. Once turning on exhaustive FLAGS, always get exhaustive_search.
      // 2. Once turning on auto-tune, run heuristic (default) before
      //    auto-tune process, run exhaustive_search during mentioned process.
      //    Auto tune is only enabled between specified range.
      // 3. After auto-tune process, run cached algorithm if cached, run
      //    default mode for the rest.
      auto key = args.ConvertToConvCacheKey<T>();
      auto& cache = phi::autotune::AutoTuneCache::Instance().GetConv(
          SearchAlgorithmBase<PerfT>::kAlgoType);
      bool find_in_cache = cache.Find(key);
      if (find_in_cache) {
        auto t = cache.Get(key);
        result.algo = static_cast<AlgoT>(t.algo);
        result.workspace_size = t.workspace_size;
        result.use_tensor_op_math = t.use_tensor_op_math;
        result.exhaustive_search = t.exhaustive_search;
      }
      if (!result.exhaustive_search) {
        // In conv2d_tranpose, enable_autotune is set to false because some
        // algorithm picked by exhaustive search method produce wrong result.
        use_autotune = enable_autotune &&
                       phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
        if (exhaustive_search || use_autotune) {
          // Once autotune is enabled, the autotuned result can rewrite the
          // previous result in cache found by heuristic method.
          result =
              SearchAlgorithmBase<PerfT>::template FindAlgoExhaustiveSearch<T>(
                  ctx, args);
          cache.Set(key,
                    phi::autotune::ConvAutoTuneResult(
                        static_cast<int64_t>(result.algo),
                        result.workspace_size,
                        result.use_tensor_op_math,
                        true));
        } else if (!find_in_cache) {
          result = FindAlgoHeuristic(ctx, args);
          cache.Set(key,
                    phi::autotune::ConvAutoTuneResult(
                        static_cast<int64_t>(result.algo),
                        result.workspace_size,
                        result.use_tensor_op_math,
                        false));
        }
      }
    }
    VLOG(3) << "[cuDNN " << SearchAlgorithmBase<PerfT>::GetPerfName()
            << "] exhaustive_search=" << exhaustive_search
            << ", use_autotune=" << use_autotune
            << ", deterministic=" << deterministic
            << ", choose algo=" << result.algo
            << ", tensor_core=" << result.use_tensor_op_math
            << ", workspace=" << ToMegaBytes(result.workspace_size) << " MB";
    return result;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const phi::GPUContext& ctx,
                                                   const ConvArgs& args) {
    SearchResult<AlgoT> result(SearchAlgorithmBase<PerfT>::kDeterministicAlgo);
    result.workspace_size =
        SearchAlgorithmBase<PerfT>::GetWorkspaceSize(ctx, args, result);
    return result;
  }

  // Heuristic search mode, calling the cudnnGetXxxAlgorithm.
  static SearchResult<AlgoT> FindAlgoHeuristic(const phi::GPUContext& ctx,
                                               const ConvArgs& args) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(SearchAlgorithmBase<PerfT>::kNumAlgorithms);
    int actual_perf_count =
        SearchAlgorithmBase<PerfT>::GetConvolutionAlgorithmV7(args,
                                                              &perf_results);
    result.algo = perf_results[best_algo_idx].algo;
    result.use_tensor_op_math =
        perf_results[best_algo_idx].mathType == CUDNN_TENSOR_OP_MATH;

#if CUDNN_VERSION < 7500
    if (std::is_same<PerfT, cudnnConvolutionBwdDataAlgoPerf_t>::value) {
      int stride_dim = args.x->dims().size() - 2;
      bool blacklist = std::any_of(args.s.begin(),
                                   args.s.begin() + stride_dim,
                                   [=](int n) { return n != 1; });
      if (blacklist && (perf_results[best_algo_idx].algo ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                        perf_results[best_algo_idx].algo ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
        result.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        // result.use_tensor_op_math =
      }
    }
#endif

    result.workspace_size =
        SearchAlgorithmBase<PerfT>::GetWorkspaceSize(ctx, args, result);
    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      VLOG(4) << GetPerfResultString<PerfT>("Heuristic search result",
                                            perf_results,
                                            actual_perf_count,
                                            workspace_size_limit);
      // cudnnGetConvolutionXxxAlgorithm is removed in CUDNN-8.
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(3) << "Fallback to non-v7 method to find conv algorithm "
                 "becasue the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      result.algo = SearchAlgorithmBase<PerfT>::GetConvolutionAlgorithm(
          args, workspace_size_limit);
#endif
    }
#else
    result.algo = SearchAlgorithmBase<PerfT>::GetConvolutionAlgorithm(
        args, workspace_size_limit);
#endif
    result.workspace_size =
        SearchAlgorithmBase<PerfT>::GetWorkspaceSize(ctx, args, result);

    return result;
  }
};

template <typename T>
struct ConvRunner {
  static void RunForward(
      const phi::GPUContext& ctx,
      const ConvArgs& args,
      const SearchResult<cudnnConvolutionFwdAlgo_t>& search_result,
      const T* input_ptr,
      const T* filter_ptr,
      T* output_ptr,
      int groups,
      int group_offset_in,
      int group_offset_filter,
      int group_offset_out,
      size_t workspace_size,
      phi::DnnWorkspaceHandle* workspace_handle,
      bool use_addto = false) {
    SetConvolutionMathType(
        ctx, args.cdesc, args.cudnn_dtype, search_result.use_tensor_op_math);

    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

    auto cudnn_handle = ctx.cudnn_handle();
    for (int i = 0; i < groups; i++) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnConvolutionForward(
                cudnn_handle,
                &alpha,
                args.idesc.desc(),
                input_ptr + i * group_offset_in,
                args.wdesc.desc(),
                filter_ptr + i * group_offset_filter,
                args.cdesc.desc(),
                search_result.algo,
                workspace_ptr,
                workspace_size,
                &beta,
                args.odesc.desc(),
                output_ptr + i * group_offset_out));
          },
          workspace_size);
    }
  }

  static void RunBackwardData(
      const phi::GPUContext& ctx,
      const ConvArgs& args,
      const SearchResult<cudnnConvolutionBwdDataAlgo_t>& search_result,
      const T* output_grad_ptr,
      const T* filter_ptr,
      T* input_grad_ptr,
      int groups,
      int group_offset_in,
      int group_offset_filter,
      int group_offset_out,
      size_t workspace_size,
      phi::DnnWorkspaceHandle* workspace_handle,
      bool use_addto = false) {
    SetConvolutionMathType(
        ctx, args.cdesc, args.cudnn_dtype, search_result.use_tensor_op_math);

    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

    auto cudnn_handle = ctx.cudnn_handle();
    for (int i = 0; i < groups; i++) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::cudnnConvolutionBackwardData(
                    cudnn_handle,
                    &alpha,
                    args.wdesc.desc(),
                    filter_ptr + i * group_offset_filter,
                    args.odesc.desc(),
                    output_grad_ptr + i * group_offset_out,
                    args.cdesc.desc(),
                    search_result.algo,
                    workspace_ptr,
                    workspace_size,
                    &beta,
                    args.idesc.desc(),
                    input_grad_ptr + i * group_offset_in));
          },
          workspace_size);
    }
  }

  static void RunBackwardFilter(
      const phi::GPUContext& ctx,
      const ConvArgs& args,
      const SearchResult<cudnnConvolutionBwdFilterAlgo_t>& search_result,
      const T* output_grad_ptr,
      const T* input_ptr,
      T* filter_grad_ptr,
      int groups,
      int group_offset_in,
      int group_offset_filter,
      int group_offset_out,
      size_t workspace_size,
      phi::DnnWorkspaceHandle* workspace_handle,
      bool use_addto = false) {
    SetConvolutionMathType(
        ctx, args.cdesc, args.cudnn_dtype, search_result.use_tensor_op_math);

    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

    auto cudnn_handle = ctx.cudnn_handle();
    for (int i = 0; i < groups; i++) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::cudnnConvolutionBackwardFilter(
                    cudnn_handle,
                    &alpha,
                    args.idesc.desc(),
                    input_ptr + i * group_offset_in,
                    args.odesc.desc(),
                    output_grad_ptr + i * group_offset_out,
                    args.cdesc.desc(),
                    search_result.algo,
                    workspace_ptr,
                    workspace_size,
                    &beta,
                    args.wdesc.desc(),
                    filter_grad_ptr + i * group_offset_filter));
          },
          workspace_size);
    }
  }
};

}  // namespace operators
}  // namespace paddle
