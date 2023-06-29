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
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

PHI_DECLARE_int64(cublaslt_exhaustive_search_times);
#endif

namespace phi {
namespace funcs {

#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060)

// Set this enum according to
// https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t
// While kMatmul, kMatmulGrad, kMatmulGradWithoutBias share the same
// enum value, but if all elements for MatmulPlanner->GetKey() is same,
// no matter forward or backward, they could share the same descriptor
// cache, in that the descriptor is for description of matmul operation.
enum MatmulFusedType {
  kMatmul = 0,
  kMatmulGrad = 1,
  kMatmulGradWithoutBias = 2,
  kMatmulBias = 3,
  kMatmulRelu = 4,
  kMatmulBiasRelu = 5,
  kMatmulBiasGelu = 6,
  kMatmulBiasReluWithReservedData = 7,
  kMatmulBiasGeluWithReservedData = 8,
  kMatmulReluGrad = 9,
  kMatmulGeluGrad = 10,
  kMatmulBiasGradToA = 11,
  kMatmulBiasGradToB = 12
};

static cublasLtEpilogue_t ConvertFusedType(MatmulFusedType fused_type) {
  static std::map<MatmulFusedType, cublasLtEpilogue_t> fused_type_map = {
      {MatmulFusedType::kMatmul, CUBLASLT_EPILOGUE_DEFAULT},
      {MatmulFusedType::kMatmulGrad, CUBLASLT_EPILOGUE_DEFAULT},
      {MatmulFusedType::kMatmulGradWithoutBias, CUBLASLT_EPILOGUE_DEFAULT},
      {MatmulFusedType::kMatmulBias, CUBLASLT_EPILOGUE_BIAS},
      {MatmulFusedType::kMatmulRelu, CUBLASLT_EPILOGUE_RELU},
      {MatmulFusedType::kMatmulBiasRelu, CUBLASLT_EPILOGUE_RELU_BIAS},
      {MatmulFusedType::kMatmulBiasGelu, CUBLASLT_EPILOGUE_GELU_BIAS},
      {MatmulFusedType::kMatmulBiasReluWithReservedData,
       CUBLASLT_EPILOGUE_RELU_AUX_BIAS},
      {MatmulFusedType::kMatmulBiasGeluWithReservedData,
       CUBLASLT_EPILOGUE_GELU_AUX_BIAS},
      {MatmulFusedType::kMatmulReluGrad, CUBLASLT_EPILOGUE_DRELU},
      {MatmulFusedType::kMatmulGeluGrad, CUBLASLT_EPILOGUE_DGELU},
      {MatmulFusedType::kMatmulBiasGradToA, CUBLASLT_EPILOGUE_BGRADA},
      {MatmulFusedType::kMatmulBiasGradToB, CUBLASLT_EPILOGUE_BGRADB}};

  return fused_type_map[fused_type];
}

enum FusedGEMMGradInType { kDX = 0, kDY = 1, kDZ = 2 };

template <bool TransX, bool TransY>
struct FusedGEMMGradTrait;

template <>
struct FusedGEMMGradTrait<false, false> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<true, false> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradATrans = false;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<false, true> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = false;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<true, true> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradATrans = true;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = true;
};

// To tell any matmul or fused matmul operation from each other.
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
                MatmulFusedType fused_type,
                const void* bias_data = nullptr,
                void* reserve_data = nullptr,  // Commonly for ReLu bit-mask.
                bool use_addto = false,
                bool no_exchange = true)
      : bias(bias_data), aux_data(reserve_data), fused_type_(fused_type) {
    use_addto_ = use_addto;
    key_ = phi::autotune::GenKey(x_dims,
                                 y_dims,
                                 static_cast<int>(trans_x),
                                 static_cast<int>(trans_y),
                                 static_cast<int>(dtype),
                                 static_cast<int>(fused_type_),
                                 static_cast<int>(use_addto_),
                                 static_cast<int>(no_exchange));
  }

  bool UseAddTo() const { return use_addto_; }
  size_t GetKey() const { return key_; }
  MatmulFusedType GetFusedType() const { return fused_type_; }

  size_t GenSubKey() const { return key_; }

 private:
  MatmulFusedType fused_type_;
  bool use_addto_;
  size_t key_;
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
  template <typename T, typename DXT, typename DYT, bool TransX, bool TransY>
  void Create(const int64_t M,
              const int64_t N,
              const int64_t K,
              const bool trans_x,
              const bool trans_y,
              phi::funcs::MatmulPlanner* planner,
              const int batch_size = 1,
              const int64_t stride_x = 0,
              const int64_t stride_y = 0,
              const int64_t stride_out = 0,
              bool grad_for_dx = true) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    cudaDataType_t mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t scale_type = phi::backends::gpu::ToCudaDataType<MT>();
    cublasComputeType_t compute_type = GetCudaComputeType<T>();

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
    SetFusedEpilogueOpDescriptor(planner, trans_x, trans_y, N);

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
    }
    if (planner->aux_data != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
          &(planner->aux_data),
          sizeof(planner->aux_data)));
    }
  }

  std::string GetDescResultString(std::string prefix,
                                  bool has_algo = true) const {
    std::ostringstream out;
    out << prefix << " \n";
#define GET_DESC_DATA_STRING(src)                    \
  do {                                               \
    out << "  " << #src << " = [";                   \
    int num = sizeof((*src)) / sizeof(src->data[0]); \
    for (int i = 0; i < num; ++i) {                  \
      if (i == 0) {                                  \
        out << src->data[i];                         \
      } else {                                       \
        out << ", " << src->data[i];                 \
      }                                              \
    }                                                \
    out << "]\n";                                    \
  } while (0);

    if (has_algo) {
      GET_DESC_DATA_STRING(algo);
    }
    GET_DESC_DATA_STRING(x_desc);
    GET_DESC_DATA_STRING(y_desc);
    GET_DESC_DATA_STRING(out_desc);
    GET_DESC_DATA_STRING(op_desc);
#undef GET_DESC_DATA_STRING
    return out.str();
  }

  void ExchangeXYDesc(bool no_exchange) {}

 protected:
  void SetFusedEpilogueOpDescriptor(phi::funcs::MatmulPlanner* planner,
                                    const bool trans_x,
                                    const bool trans_y,
                                    int64_t lead_dim) {
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
    MatmulFusedType fused_type = planner->GetFusedType();
    if (fused_type != MatmulFusedType::kMatmul) {
      cublasLtEpilogue_t cublaslt_fused_type = ConvertFusedType(fused_type);
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &cublaslt_fused_type,
                                                  sizeof(fused_type)));
    }
    if (planner->aux_data) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
          &lead_dim,
          sizeof(lead_dim)));
    }
  }

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
};

struct MatmulGradDescriptor : MatmulDescriptor {
 public:
  MatmulGradDescriptor() {}

  template <typename T, typename DXT, typename DYT, bool TransX, bool TransY>
  void Create(const int64_t M,
              const int64_t N,
              const int64_t K,
              const bool trans_x,
              const bool trans_y,
              phi::funcs::MatmulPlanner* planner,
              const int batch_size = 1,
              int64_t stride_x = 0,
              int64_t stride_y = 0,
              int64_t stride_out = 0,
              bool grad_for_dx = true) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    cudaDataType_t mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t scale_type = phi::backends::gpu::ToCudaDataType<MT>();
    cublasComputeType_t compute_type = GetCudaComputeType<T>();

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
    this->SetFusedEpilogueOpDescriptor(
        planner, trans_x, trans_y, TransX ? M : K);

    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    this->CreateMatrixLayout(&x_desc, mat_type, N, M, true);
    if (grad_for_dx) {
      this->CreateMatrixLayout(&y_desc, mat_type, K, N, TransY);
      this->CreateMatrixLayout(
          &out_desc, phi::backends::gpu::ToCudaDataType<DXT>(), M, K, TransX);
    } else {
      this->CreateMatrixLayout(&y_desc, mat_type, M, K, TransX);
      this->CreateMatrixLayout(
          &out_desc, phi::backends::gpu::ToCudaDataType<DYT>(), K, N, TransY);
    }
  }

  void ExchangeXYDesc(bool no_exchange) {
    if (no_exchange) {
      return;
    }
    auto* temp = y_desc;
    y_desc = x_desc;
    x_desc = temp;
  }
};

template <typename T, typename OutT = T, class MatmulDescT = MatmulDescriptor>
struct CublasLtBase {
 public:
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  static phi::Allocator::AllocationPtr GetWorkspace(const phi::GPUContext& ctx,
                                                    size_t workspace_size) {
    return phi::memory_utils::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }

  static void RunImpl(const phi::GPUContext& ctx,
                      MatmulDescT* desc,
                      const size_t sub_key,
                      const T* x_ptr,
                      const T* y_ptr,
                      OutT* out_ptr,
                      phi::funcs::MatmulPlanner* planner) {
    MT alpha = static_cast<MT>(1);
    MT beta = planner->UseAddTo() ? static_cast<MT>(1) : static_cast<MT>(0);
    cublasLtHandle_t cublaslt_handle = ctx.cublaslt_handle();

    // NOTE(limingshu): As workspace_size varies from different DL framework,
    // I wonder is there any smarter idea for workspace setting, currently I
    // just followed the settings from the NVIDIA colleague`s setting.
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
        MatmulDescT* best_desc = new MatmulDescT(*desc);
        VLOG(6) << best_desc->GetDescResultString(
            "[Searched CublasltDescriptor] ");

        auto& cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
        cache.SetSubKey(sub_key, reinterpret_cast<void*>(best_desc));
      }
    }

    VLOG(7) << desc->GetDescResultString("[Impl CublasltDescriptor] ");
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
                             MatmulDescT* desc,
                             const void* alpha,
                             const void* beta,
                             const void* y_data,
                             const void* x_data,
                             void* out_data,
                             void* workspace_ptr,
                             size_t workspace_size) {
    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)));

    int returned_results = 0;
    constexpr int requested_algo_count = 10;
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
    int best_algo_idx = -1;
    if (returned_results == 1 || FLAGS_cublaslt_exhaustive_search_times <= 0) {
      best_algo_idx = 0;
    } else {
      float min_time_cost = std::numeric_limits<float>::max();
      for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
        float cur_time_cost =
            RunAndMeasureAlgo(ctx,
                              lt_handle,
                              desc,
                              alpha,
                              beta,
                              y_data,
                              x_data,
                              out_data,
                              workspace_ptr,
                              workspace_size,
                              &(heuristic_results[algo_idx].algo));
        VLOG(6) << "[MatmulWithCublaslt] algo[" << algo_idx
                << "] time: " << cur_time_cost << " s";

        if ((best_algo_idx == 0 && (1.05 * cur_time_cost < min_time_cost)) ||
            (cur_time_cost < min_time_cost)) {
          best_algo_idx = algo_idx;
          min_time_cost = cur_time_cost;
        }
      }
    }
    VLOG(6) << "[MatmulWithCublaslt] best_algo_idx: " << best_algo_idx;

    cublasLtMatmulAlgo_t* best_algo = desc->SetAlgo();
    *best_algo = heuristic_results[best_algo_idx].algo;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceDestroy(preference));
  }

  static float RunAndMeasureAlgo(const phi::GPUContext& ctx,
                                 const cublasLtHandle_t& lt_handle,
                                 MatmulDescT* desc,
                                 const void* alpha,
                                 const void* beta,
                                 const void* y_data,
                                 const void* x_data,
                                 void* out_data,
                                 void* workspace_ptr,
                                 size_t workspace_size,
                                 cublasLtMatmulAlgo_t* algo) {
    int repeats = FLAGS_cublaslt_exhaustive_search_times;
    if (repeats <= 0) {
      return std::numeric_limits<float>::max();
    }

    phi::GpuTimer timer;
    float time_cost = 0.f;
    const auto& stream = ctx.stream();

    for (int i = 0; i < repeats; ++i) {
      timer.Start(stream);
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(lt_handle,
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
                                                         algo,
                                                         workspace_ptr,
                                                         workspace_size,
                                                         stream));
      timer.Stop(stream);
      ctx.Wait();
      auto time = timer.ElapsedTime();
      if (i > 0) {
        // Exclude the warmup runtime.
        time_cost += time;
      }
    }
    return (time_cost / (repeats - 1));
  }
};

// To judge if desc is cached or not.
template <class DescT,
          typename T,
          typename DXT = T,
          typename DYT = T,
          bool TransX = false,
          bool TransY = false>
struct DescriptorSetter {
 public:
  DescT desc;
  size_t sub_key{std::numeric_limits<size_t>::min()};

  DescriptorSetter(phi::funcs::MatmulPlanner* planner,
                   const int64_t M,
                   const int64_t N,
                   const int64_t K,
                   const bool trans_x,
                   const bool trans_y,
                   const int batch_size = 1,
                   int64_t stride_x = 0,
                   int64_t stride_y = 0,
                   int64_t stride_out = 0,
                   const bool no_exchange = true,
                   bool grad_for_dx = true) {
    if (planner != nullptr) {
      sub_key = planner->GenSubKey();
    }

    auto& mamtul_cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
    if (mamtul_cache.FindSubKey(sub_key)) {
      desc = *(reinterpret_cast<DescT*>(mamtul_cache.GetSubKey(sub_key)));
      desc.template SetFusedEpiloguePtr<DYT>(planner);
      VLOG(7) << desc.GetDescResultString("[Heap CublasltDescriptor] ");
    } else {
      desc.template Create<T, DXT, DYT, TransX, TransY>(M,
                                                        N,
                                                        K,
                                                        trans_x,
                                                        trans_y,
                                                        planner,
                                                        batch_size,
                                                        stride_x,
                                                        stride_y,
                                                        stride_out,
                                                        grad_for_dx);
      desc.ExchangeXYDesc(no_exchange);
      if (planner != nullptr) {
        desc.template SetFusedEpiloguePtr<DYT>(planner);
      }
      VLOG(7) << desc.GetDescResultString("[Stack CublasltDescriptor] ", false);
    }
  }
};

// For matmul with kernels autotune
template <typename T>
struct MatmulWithCublasLt : public CublasLtBase<T> {
 public:
  static void Run(const phi::GPUContext& ctx,
                  const T* x_data,
                  const T* y_data,
                  T* out_data,
                  const int64_t M,
                  const int64_t N,
                  const int64_t K,
                  const bool trans_x,
                  const bool trans_y,
                  phi::funcs::MatmulPlanner* planner = nullptr) {
    auto setter = DescriptorSetter<MatmulDescriptor, T>(
        planner, M, N, K, trans_x, trans_y);
    CublasLtBase<T>::RunImpl(
        ctx, &setter.desc, setter.sub_key, x_data, y_data, out_data, planner);
  }

  static void RunWithBatch(const phi::GPUContext& ctx,
                           const T* x_data,
                           const T* y_data,
                           T* out_data,
                           const int64_t M,
                           const int64_t N,
                           const int64_t K,
                           bool trans_x,
                           bool trans_y,
                           int batch_size,
                           int64_t stride_x,
                           int64_t stride_y,
                           int64_t stride_out,
                           phi::funcs::MatmulPlanner* planner = nullptr) {
    auto setter = DescriptorSetter<MatmulDescriptor, T>(planner,
                                                        M,
                                                        N,
                                                        K,
                                                        trans_x,
                                                        trans_y,
                                                        batch_size,
                                                        stride_x,
                                                        stride_y,
                                                        stride_out);
    CublasLtBase<T>::RunImpl(
        ctx, &setter.desc, setter.sub_key, x_data, y_data, out_data, planner);
  }

  static void RunWithBatch(const phi::GPUContext& ctx,
                           const T** x_data,
                           const T** y_data,
                           T** out_data,
                           const int64_t M,
                           const int64_t N,
                           const int64_t K,
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
};

// As for just Linear fused ephilogue below: out = matmul(x, y) + bias.
template <typename T>
struct LinearWithCublasLt : public CublasLtBase<T> {
  static void Run(const phi::GPUContext& ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* y,
                  phi::DenseTensor* out,
                  const void* bias_data,
                  void* reserve_data,
                  const int64_t M,
                  const int64_t N,
                  const int64_t K,
                  const bool trans_x,
                  const bool trans_y,
                  const MatmulFusedType fused_type) {
    auto planner = phi::funcs::MatmulPlanner(vectorize(x->dims()),
                                             vectorize(y->dims()),
                                             trans_x,
                                             trans_y,
                                             phi::CppTypeToDataType<T>::Type(),
                                             fused_type,
                                             bias_data,
                                             reserve_data);
    auto setter = DescriptorSetter<MatmulDescriptor, T>(
        &planner, M, N, K, trans_x, trans_y);
    CublasLtBase<T>::RunImpl(ctx,
                             &setter.desc,
                             setter.sub_key,
                             x->data<T>(),
                             y->data<T>(),
                             out->data<T>(),
                             &planner);
  }
};

template <typename T, typename DXT, typename DYT, bool TransX, bool TransY>
struct LinearGradWithCublasLt : public CublasLtBase<T> {
  static void Run(
      const phi::GPUContext& ctx,
      const phi::DenseTensor* x,
      const phi::DenseTensor* y,
      phi::DenseTensor* out,
      const void* bias_data,
      void* reserve_data,
      const int64_t M,
      const int64_t N,
      const int64_t K,
      const MatmulFusedType fused_type,
      const bool trans_x,
      const bool trans_y,
      const bool use_addto,
      const bool no_exchange,  // exchange x_desc and y_desc for grad.
      bool grad_for_dx = true) {
    auto planner = phi::funcs::MatmulPlanner(vectorize(x->dims()),
                                             vectorize(y->dims()),
                                             trans_x,
                                             trans_y,
                                             phi::CppTypeToDataType<T>::Type(),
                                             fused_type,
                                             bias_data,
                                             reserve_data,
                                             use_addto,
                                             no_exchange);
    auto setter =
        DescriptorSetter<MatmulGradDescriptor, T, DXT, DYT, TransX, TransY>(
            &planner,
            M,
            N,
            K,
            trans_x,
            trans_y,
            /*batch_size=*/1,
            /*stride_x=*/0,
            /*stride_y=*/0,
            /*stride_out=*/0,
            /*exchange_x_y_desc=*/no_exchange,
            /*grad_for_dx=*/grad_for_dx);

    // To setting data type for different kinda out_data.
    if (grad_for_dx) {
      CublasLtBase<T, DXT, MatmulGradDescriptor>::RunImpl(
          ctx,
          &setter.desc,
          setter.sub_key,
          no_exchange ? x->data<T>() : y->data<T>(),
          no_exchange ? y->data<T>() : x->data<T>(),
          out->data<DXT>(),
          &planner);
    } else {
      CublasLtBase<T, DYT, MatmulGradDescriptor>::RunImpl(
          ctx,
          &setter.desc,
          setter.sub_key,
          no_exchange ? x->data<T>() : y->data<T>(),
          no_exchange ? y->data<T>() : x->data<T>(),
          out->data<DYT>(),
          &planner);
    }
  }
};
#else
// A void structure just for successfully compile.
struct MatmulPlanner {};
#endif  // (PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060

}  // namespace funcs
}  // namespace phi
