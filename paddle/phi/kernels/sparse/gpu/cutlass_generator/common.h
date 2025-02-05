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

#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
#include "cutlass/arch/mma.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "examples/common/helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
namespace phi {
namespace sparse {
size_t constexpr max_splitk_slices = 256;
size_t constexpr max_in_channels = 256;
size_t constexpr max_out_channels = 256;
static size_t workspace_size =
    sizeof(float) * max_splitk_slices * max_in_channels * max_out_channels;

#define TYPEDEF_KERNEL_POINTER(kernel, in_type, out_type) \
  typedef void (*kernel)(out_type const alpha,            \
                         out_type const beta,             \
                         const GPUContext& dev_ctx,       \
                         const in_type* const a,          \
                         const in_type* const b,          \
                         const out_type* const c,         \
                         out_type* const d,               \
                         const int m,                     \
                         const int n,                     \
                         const int k,                     \
                         const int32_t* a_indices,        \
                         const int32_t* b_indices,        \
                         const int32_t* c_d_indices,      \
                         void* const workspace_ptr);
#define GATHER_GEMM_SCATTER_CHECK(status)                      \
  {                                                            \
    cutlass::Status error = status;                            \
    if (error != cutlass::Status::kSuccess) {                  \
      throw std::runtime_error(cutlassGetStatusString(error)); \
    }                                                          \
  }
#define DEFINE_LAUNCH_KERNEL(in_type, out_type)                               \
  template <typename Config>                                                  \
  void launchKernel(out_type const alpha,                                     \
                    out_type const beta,                                      \
                    const GPUContext& dev_ctx,                                \
                    const in_type* const a,                                   \
                    const in_type* const b,                                   \
                    const out_type* const c,                                  \
                    out_type* const d,                                        \
                    const int m,                                              \
                    const int n,                                              \
                    const int k,                                              \
                    const int32_t* a_indices,                                 \
                    const int32_t* b_indices,                                 \
                    const int32_t* c_d_indices,                               \
                    void* const workspace_ptr) {                              \
    cutlass::gemm::GemmCoord problem_size_real({m, n, k});                    \
    using Gemm = typename Config::Gemm;                                       \
    int split_k_slices = std::max(std::min(64, k / 128), 1);                  \
    typename Gemm::Arguments arguments{                                       \
        Config::Mode,                                                         \
        problem_size_real,                                                    \
        split_k_slices,                                                       \
        {static_cast<const typename Gemm::Base::ElementAccumulator>(          \
             static_cast<const float>(alpha)),                                \
         static_cast<const typename Gemm::Base::ElementAccumulator>(          \
             static_cast<const float>(beta))},                                \
        reinterpret_cast<const typename Gemm::Base::ElementA* const>(a),      \
        reinterpret_cast<const typename Gemm::Base::ElementB* const>(b),      \
        reinterpret_cast<const typename Gemm::Base::ElementC* const>(c),      \
        reinterpret_cast<typename Gemm::Base::ElementC* const>(d),            \
        m * k,                                                                \
        k * n,                                                                \
        m * n,                                                                \
        m * n,                                                                \
        std::is_same<typename Gemm::Base::LayoutA,                            \
                     cutlass::layout::RowMajor>::value                        \
            ? problem_size_real.k()                                           \
            : problem_size_real.m(),                                          \
        std::is_same<typename Gemm::Base::LayoutB,                            \
                     cutlass::layout::RowMajor>::value                        \
            ? problem_size_real.n()                                           \
            : problem_size_real.k(),                                          \
        std::is_same<typename Gemm::Base::LayoutC,                            \
                     cutlass::layout::RowMajor>::value                        \
            ? problem_size_real.n()                                           \
            : problem_size_real.m(),                                          \
        std::is_same<typename Gemm::Base::LayoutC,                            \
                     cutlass::layout::RowMajor>::value                        \
            ? problem_size_real.n()                                           \
            : problem_size_real.m(),                                          \
        a_indices,                                                            \
        b_indices,                                                            \
        c_d_indices};                                                         \
    cutlass::device_memory::allocation<uint8_t>* const real_workspace_ptr =   \
        static_cast<cutlass::device_memory::allocation<uint8_t>* const>(      \
            workspace_ptr);                                                   \
    if (Config::Mode ==                                                       \
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) {              \
      size_t current_workspace_size = Gemm::get_workspace_size(arguments);    \
      if (current_workspace_size > workspace_size) {                          \
        workspace_size = current_workspace_size;                              \
        real_workspace_ptr->reset(workspace_size);                            \
      }                                                                       \
                                                                              \
      arguments.ptr_D = real_workspace_ptr->get();                            \
    }                                                                         \
    Gemm gemm_op;                                                             \
    cutlass::Status status = gemm_op.can_implement(arguments);                \
    GATHER_GEMM_SCATTER_CHECK(status);                                        \
    if (Config::Mode ==                                                       \
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) {              \
      status = gemm_op.initialize(arguments, real_workspace_ptr->get());      \
    } else {                                                                  \
      cutlass::device_memory::allocation<uint8_t> empty_workspace(0);         \
      status = gemm_op.initialize(arguments, empty_workspace.get());          \
    }                                                                         \
    GATHER_GEMM_SCATTER_CHECK(status);                                        \
    gemm_op(dev_ctx.stream());                                                \
    if (Config::Mode ==                                                       \
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) {              \
      using ReductionOp = cutlass::reduction::thread::ReduceAdd<              \
          typename Gemm::ElementAccumulator,                                  \
          typename Gemm::EpilogueOutputOp::ElementAccumulator,                \
          Gemm::EpilogueOutputOp::kCount>;                                    \
                                                                              \
      using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<       \
          cutlass::MatrixShape<4, 32 * Gemm::EpilogueOutputOp::kCount>,       \
          typename Gemm::EpilogueOutputOp,                                    \
          ReductionOp>;                                                       \
      using ReductionDevice =                                                 \
          typename cutlass::reduction::device::ReduceSplitK<ReductionKernel>; \
      ReductionDevice reduction_op;                                           \
      int splitk_gemm_stride = n;                                             \
      cutlass::layout::RowMajor splitk_gemm_layout(splitk_gemm_stride);       \
      void* workspace_gemm_ptr = real_workspace_ptr->get();                   \
      cutlass::TensorRef<typename Gemm::ElementAccumulator,                   \
                         cutlass::layout::RowMajor>                           \
          ref_workspace(reinterpret_cast<typename Gemm::ElementAccumulator*>( \
                            workspace_gemm_ptr),                              \
                        splitk_gemm_layout);                                  \
      cutlass::TensorRef<typename Gemm::Base::ElementC,                       \
                         typename Gemm::Base::LayoutC>                        \
          ref_c(reinterpret_cast<typename Gemm::Base::ElementC* const>(d),    \
                splitk_gemm_layout);                                          \
      cutlass::TensorRef<typename Gemm::Base::ElementC,                       \
                         typename Gemm::Base::LayoutC>                        \
          ref_d(reinterpret_cast<typename Gemm::Base::ElementC* const>(d),    \
                splitk_gemm_layout);                                          \
      typename ReductionDevice::Arguments reduction_args(                     \
          problem_size_real.mn(),                                             \
          split_k_slices,                                                     \
          static_cast<size_t>(problem_size_real.m() * problem_size_real.n()), \
          ref_workspace,                                                      \
          ref_d,                                                              \
          ref_c,                                                              \
          {static_cast<const typename Gemm::Base::ElementAccumulator>(        \
               static_cast<const float>(alpha)),                              \
           static_cast<const typename Gemm::Base::ElementAccumulator>(        \
               static_cast<const float>(beta))});                             \
      status = reduction_op.initialize(reduction_args);                       \
      GATHER_GEMM_SCATTER_CHECK(status);                                      \
      reduction_op(dev_ctx.stream());                                         \
    }                                                                         \
  }

TYPEDEF_KERNEL_POINTER(gather_hgemm_scatter, phi::dtype::float16, phi::float16)
TYPEDEF_KERNEL_POINTER(gather_sgemm_scatter, float, float)
TYPEDEF_KERNEL_POINTER(gather_sgemm_f16_scatter, phi::dtype::float16, float)

DEFINE_LAUNCH_KERNEL(phi::dtype::float16, phi::dtype::float16)
DEFINE_LAUNCH_KERNEL(float, float)
DEFINE_LAUNCH_KERNEL(phi::dtype::float16, float)

}  // namespace sparse
}  // namespace phi
#endif
