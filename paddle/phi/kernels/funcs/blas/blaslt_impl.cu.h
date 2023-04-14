/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060

#include "glog/logging.h"

#include <cuda_runtime_api.h>  // NOLINT
#include "cuda.h"              // NOLINT
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#endif

namespace phi {
namespace funcs {

#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060)
// Set this enum according to
// https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t
enum MatmulFusedType {
  kMatmul = CUBLASLT_EPILOGUE_DEFAULT,  // No special postprocessing.
  kMatmulBias = CUBLASLT_EPILOGUE_BIAS,
  kMatmulRelu = CUBLASLT_EPILOGUE_RELU,
  kMatmulBiasRelu =
      CUBLASLT_EPILOGUE_RELU_BIAS,  // Apply bias and then ReLU transform.
  kMatmulBiasGelu =
      CUBLASLT_EPILOGUE_GELU_BIAS,  // Apply Bias and then GELU transform.
  kMatmulBiasReluWithReservedData = CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
  kMatmulBiasGeluWithReservedData = CUBLASLT_EPILOGUE_GELU_AUX_BIAS
};

struct MatmulPlanner {
 public:
  const void* bias{nullptr};
  void* aux_data{nullptr};

  MatmulPlanner() {}
  MatmulPlanner(const std::vector<int64_t>& x_dims,
                const std::vector<int64_t>& y_dims,
                const bool trans_x,
                const bool trans_y,
                phi::DataType dtype,
                MatmulFusedType impl_type,
                const void* bias_data = nullptr,
                void* reserve_data = nullptr)
      : bias(bias_data), aux_data(reserve_data) {
    type = impl_type;
    key = phi::autotune::GenKey(x_dims,
                                y_dims,
                                static_cast<int64_t>(trans_x),
                                static_cast<int64_t>(trans_y),
                                static_cast<int64_t>(dtype));
  }

  MatmulFusedType ImplType() const { return type; }
  size_t GetKey() const { return key; }
  size_t GenSubKey(int idx) const { return phi::autotune::GenKey(key, idx); }

 private:
  MatmulFusedType type;
  size_t key;
};

template <typename T>
cublasComputeType_t GetCudaComputeType() {
  if (std::is_same<T, double>::value) {
    return CUBLAS_COMPUTE_64F;
  } else {
    return CUBLAS_COMPUTE_32F;
  }
}

struct MatmulDescriptor {
 public:
  cublasLtMatmulDesc_t op_desc{nullptr};
  cublasLtMatrixLayout_t x_desc{nullptr};
  cublasLtMatrixLayout_t y_desc{nullptr};
  cublasLtMatrixLayout_t out_desc{nullptr};
  cublasLtMatmulAlgo_t* algo{nullptr};
  bool is_cached{false};

  MatmulDescriptor() {}
  MatmulDescriptor(const MatmulDescriptor& obj) {
    algo = obj.algo;
    x_desc = obj.x_desc;
    y_desc = obj.y_desc;
    op_desc = obj.op_desc;
    out_desc = obj.out_desc;
    is_cached = obj.is_cached;
  }

  ~MatmulDescriptor() PADDLE_MAY_THROW {
    if (!is_cached) {
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatmulDescDestroy(op_desc));
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(y_desc));
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(x_desc));
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(out_desc));
      delete algo;

      op_desc = nullptr;
      x_desc = nullptr;
      y_desc = nullptr;
      out_desc = nullptr;
      algo = nullptr;
    }
  }

  // x_desc, y_desc, op_desc are allocated in heap memory.
  template <typename T>
  void Create(const int M,
              const int N,
              const int K,
              const bool trans_x,
              const bool trans_y,
              phi::funcs::MatmulPlanner* planner,
              const int batch_size = 1,
              int64_t stride_x = 0,
              int64_t stride_y = 0,
              int64_t stride_out = 0) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;

    cudaDataType_t mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t scale_type = phi::backends::gpu::ToCudaDataType<MT>();
    cublasComputeType_t compute_type = GetCudaComputeType<T>();

    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
    cublasOperation_t cublas_trans_x = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_trans_y = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                CUBLASLT_MATMUL_DESC_TRANSB,
                                                &cublas_trans_x,
                                                sizeof(cublas_trans_x)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                &cublas_trans_y,
                                                sizeof(cublas_trans_y)));

    // Create matrix descriptors
    CreateMatrixLayout(&x_desc, mat_type, M, K, trans_x);
    CreateMatrixLayout(&y_desc, mat_type, K, N, trans_y);
    CreateMatrixLayout(&out_desc, mat_type, M, N, false);

    // Config batch size and stride.
    if (batch_size > 1) {
      SetBatchAndStride(x_desc, batch_size, stride_x);
      SetBatchAndStride(y_desc, batch_size, stride_y);
      SetBatchAndStride(out_desc, batch_size, stride_out);
    }
    SetFusedEpilogueOpDescriptor(planner, N);
  }

  cublasLtMatmulAlgo_t* SetAlgo() {
    // while entering this function, the desc shall be cached.
    is_cached = true;
    algo = new cublasLtMatmulAlgo_t;
    return algo;
  }

  template <typename T>
  void SetFusedEpiloguePtr(phi::funcs::MatmulPlanner* planner) {
    if (planner->bias != nullptr) {
      const T* bias_data = static_cast<const T*>(planner->bias);
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_data,
          sizeof(bias_data)));

      if (planner->aux_data != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            &(planner->aux_data),
            sizeof(planner->aux_data)));
      }
    }
  }

  std::string GetDescResultString(std::string prefix,
                                  bool has_algo = true) const {
    std::ostringstream out;
    out << prefix << " \n";
#define GET_DESC_DATA_INFO(src)                      \
  do {                                               \
    out << #src << "= [";                            \
    int num = sizeof((*src)) / sizeof(src->data[0]); \
    for (int i = 0; i < num; ++i) {                  \
      out << src->data[i] << ", ";                   \
    }                                                \
    out << "]\n";                                    \
  } while (0);

    if (has_algo) {
      GET_DESC_DATA_INFO(&algo);
    }
    GET_DESC_DATA_INFO(x_desc);
    GET_DESC_DATA_INFO(y_desc);
    GET_DESC_DATA_INFO(out_desc);
    GET_DESC_DATA_INFO(op_desc);
    return out.str();
  }

 private:
  void CreateMatrixLayout(cublasLtMatrixLayout_t* desc,
                          cudaDataType type,
                          uint64_t rows,
                          uint64_t cols,
                          bool trans) {
    if (trans) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(desc, type, rows, cols, rows));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(desc, type, cols, rows, cols));
    }
  }

  void SetBatchAndStride(cublasLtMatrixLayout_t desc,
                         int batch_size,
                         int64_t stride) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride,
        sizeof(stride)));
  }

  void SetFusedEpilogueOpDescriptor(phi::funcs::MatmulPlanner* planner,
                                    int64_t lead_dim) {
    if (planner->bias) {
      auto fuse_type = static_cast<cublasLtEpilogue_t>(planner->ImplType());
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &fuse_type,
                                                  sizeof(fuse_type)));
      if (planner->aux_data) {
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
            &lead_dim,
            sizeof(lead_dim)));
      }
    }
  }
};

template <typename T>
struct DescriptorSetter {
  MatmulDescriptor desc;
  size_t sub_key{std::numeric_limits<size_t>::min()};

  DescriptorSetter(phi::funcs::MatmulPlanner* planner,
                   const int M,
                   const int N,
                   const int K,
                   const bool trans_x,
                   const bool trans_y,
                   const int batch_size = 1,
                   int64_t stride_x = 0,
                   int64_t stride_y = 0,
                   int64_t stride_out = 0) {
    if (planner != nullptr) {
      sub_key = planner->GenSubKey(static_cast<size_t>(planner->ImplType()));
    }

    auto& mamtul_cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
    if (mamtul_cache.FindSubKey(sub_key)) {
      desc = *(
          reinterpret_cast<MatmulDescriptor*>(mamtul_cache.GetSubKey(sub_key)));
      desc.SetFusedEpiloguePtr<T>(planner);
      VLOG(6) << desc.GetDescResultString("[Heap MatmulDescriptor] ");
    } else {
      desc.Create<T>(M,
                     N,
                     K,
                     trans_x,
                     trans_y,
                     planner,
                     batch_size,
                     stride_x,
                     stride_y,
                     stride_out);
      if (planner != nullptr) {
        desc.SetFusedEpiloguePtr<T>(planner);
      }
      VLOG(6) << desc.GetDescResultString("[Stack MatmulDescriptor] ", false);
    }
  }
};

template <typename T>
struct MatmulWithCublasLt {
 public:
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  static void Run(const phi::GPUContext& ctx,
                  const T* x_data,
                  const T* y_data,
                  T* out_data,
                  const int M,
                  const int N,
                  const int K,
                  const bool trans_x,
                  const bool trans_y,
                  phi::funcs::MatmulPlanner* planner = nullptr) {
    auto setter = DescriptorSetter<T>(planner, M, N, K, trans_x, trans_y);
    RunImpl(
        ctx, &setter.desc, setter.sub_key, x_data, y_data, out_data, planner);
  }

  static void RunWithBatch(const phi::GPUContext& ctx,
                           const T* x_data,
                           const T* y_data,
                           T* out_data,
                           const int M,
                           const int N,
                           const int K,
                           bool trans_x,
                           bool trans_y,
                           int batch_size,
                           int64_t stride_x,
                           int64_t stride_y,
                           int64_t stride_out,
                           phi::funcs::MatmulPlanner* planner = nullptr) {
    auto setter = DescriptorSetter<T>(planner,
                                      M,
                                      N,
                                      K,
                                      trans_x,
                                      trans_y,
                                      batch_size,
                                      stride_x,
                                      stride_y,
                                      stride_out);
    RunImpl(
        ctx, &setter.desc, setter.sub_key, x_data, y_data, out_data, planner);
  }

  static void RunWithBatch(const phi::GPUContext& ctx,
                           const T** x_data,
                           const T** y_data,
                           T** out_data,
                           const int M,
                           const int N,
                           const int K,
                           bool trans_x,
                           bool trans_y,
                           int batch_size,
                           phi::funcs::MatmulPlanner* planner = nullptr) {
    for (int i = 0; i < batch_size; ++i) {
      Run(ctx,
          x_data[i],
          y_data[i],
          out_data[i],
          M,
          N,
          K,
          trans_x,
          trans_y,
          planner);
    }
  }

 private:
  static phi::Allocator::AllocationPtr GetWorkspace(const phi::GPUContext& ctx,
                                                    size_t workspace_size) {
    return phi::memory_utils::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }

  static void RunImpl(const phi::GPUContext& ctx,
                      MatmulDescriptor* desc,
                      const size_t sub_key,
                      const T* x_ptr,
                      const T* y_ptr,
                      T* out_ptr,
                      phi::funcs::MatmulPlanner* planner) {
    MT alpha = static_cast<MT>(1);
    MT beta = static_cast<MT>(0);

    cublasLtHandle_t cublaslt_handle = ctx.cublaslt_handle();
    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    phi::Allocator::AllocationPtr workspace = GetWorkspace(ctx, workspace_size);

    if (planner != nullptr) {
      if (phi::autotune::AutoTuneStatus::Instance().UseAutoTune() &&
          (!desc->is_cached)) {
        SearchBestAlgo(ctx,
                       cublaslt_handle,
                       desc,
                       static_cast<void*>(&alpha),
                       static_cast<void*>(&beta),
                       y_ptr,
                       x_ptr,
                       out_ptr,
                       workspace->ptr(),
                       workspace_size);
        MatmulDescriptor* best_desc = new MatmulDescriptor(*desc);
        VLOG(6) << best_desc->GetDescResultString(
            "[Searched MatmulDescriptor] ");

        auto& cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
        cache.SetSubKey(sub_key, reinterpret_cast<void*>(best_desc));
      }
    }

    VLOG(6) << desc->GetDescResultString("[Impl MatmulDescriptor] ");
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmul(cublaslt_handle,
                                desc->op_desc,
                                static_cast<void*>(&alpha),
                                y_ptr,
                                desc->y_desc,
                                x_ptr,
                                desc->x_desc,
                                static_cast<void*>(&beta),
                                out_ptr,
                                desc->out_desc,
                                out_ptr,
                                desc->out_desc,
                                desc->algo,
                                workspace->ptr(),
                                workspace_size,
                                ctx.stream()));
  }

  static void SearchBestAlgo(const phi::GPUContext& ctx,
                             const cublasLtHandle_t& lt_handle,
                             MatmulDescriptor* desc,
                             const void* alpha,
                             const void* beta,
                             const void* y_data,
                             const void* x_data,
                             void* out_data,
                             void* workspace_ptr,
                             size_t workspace_size) {
    cublasLtMatmulAlgo_t* best_algo = desc->SetAlgo();
    const auto& stream = ctx.stream();
    int returned_results = 0;
    constexpr int requested_algo_count = 10;
    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)));
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
        requested_algo_count);
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                                desc->op_desc,
                                                desc->y_desc,
                                                desc->x_desc,
                                                desc->out_desc,
                                                desc->out_desc,
                                                preference,
                                                requested_algo_count,
                                                heuristic_results.data(),
                                                &returned_results));
    PADDLE_ENFORCE_GT(returned_results,
                      0,
                      phi::errors::Unavailable("No GEMM algorithm avaliable."));
    phi::GpuTimer timer;
    int best_algo_idx = -1;
    constexpr int repeats = 6;
    float min_time_cost = std::numeric_limits<float>::max();
    for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
      ctx.Wait();
      float cur_time = 0.f;
      for (int i = 0; i < repeats; ++i) {
        timer.Start(stream);
        PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cublasLtMatmul(lt_handle,
                                    desc->op_desc,
                                    alpha,
                                    y_data,
                                    desc->y_desc,
                                    x_data,
                                    desc->x_desc,
                                    beta,
                                    out_data,
                                    desc->out_desc,
                                    out_data,
                                    desc->out_desc,
                                    &(heuristic_results[algo_idx].algo),
                                    workspace_ptr,
                                    workspace_size,
                                    stream));
        timer.Stop(stream);
        auto time = timer.ElapsedTime();
        if (i > 0) {
          cur_time += time;
        }
      }
      float time_cnt = (cur_time / (repeats - 1));
      VLOG(4) << "Time cost in MatmulWithCublaslt algo[" << algo_idx << "]"
              << "is : " << time_cnt << " s";

      if (cur_time < min_time_cost) {
        best_algo_idx = algo_idx;
        min_time_cost = cur_time;
      }
    }
    VLOG(4) << "Best_algo_idx in MatmulWithCublaslt is : " << best_algo_idx;
    *best_algo = heuristic_results[best_algo_idx].algo;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceDestroy(preference));
  }
};
#else
// A void structure just for successfully complile.
struct MatmulPlanner {};
#endif  // (PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060

}  // namespace funcs
}  // namespace phi
