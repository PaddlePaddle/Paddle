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

#ifdef PADDLE_WITH_CUTLASS
#include "cutlass/arch/mma.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "examples/common/helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
namespace phi {
namespace sparse {
#define TYPEDEF_KERNEL_POINTER(kernel, dtype)       \
  typedef void (*kernel)(dtype const alpha,         \
                         dtype const beta,          \
                         const GPUContext& dev_ctx, \
                         const dtype* const a,      \
                         const dtype* const b,      \
                         const dtype* const c,      \
                         dtype* const d,            \
                         const int m,               \
                         const int n,               \
                         const int k,               \
                         const int32_t* a_indices,  \
                         const int32_t* b_indices,  \
                         const int32_t* c_d_indices);
#define GATHER_GEMM_SCATTER_CHECK(status)                      \
  {                                                            \
    cutlass::Status error = status;                            \
    if (error != cutlass::Status::kSuccess) {                  \
      throw std::runtime_error(cutlassGetStatusString(error)); \
    }                                                          \
  }
#define DEFINE_LAUNCH_KERNEL(dtype, cutlass_type)                              \
  template <typename Config>                                                   \
  void launchKernel(dtype const alpha,                                         \
                    dtype const beta,                                          \
                    const GPUContext& dev_ctx,                                 \
                    const dtype* const a,                                      \
                    const dtype* const b,                                      \
                    const dtype* const c,                                      \
                    dtype* const d,                                            \
                    const int m,                                               \
                    const int n,                                               \
                    const int k,                                               \
                    const int32_t* a_indices,                                  \
                    const int32_t* b_indices,                                  \
                    const int32_t* c_d_indices) {                              \
    cutlass::gemm::GemmCoord problem_size_real({m, n, k});                     \
    using Gemm = typename Config::Gemm;                                        \
    int split_k_slices = std::max(std::min(128, k / 128), 1);                  \
    typename Gemm::Arguments arguments{                                        \
        Config::Mode,                                                          \
        problem_size_real,                                                     \
        split_k_slices,                                                        \
        {static_cast<const cutlass_type>(static_cast<const float>(alpha)),     \
         static_cast<const cutlass_type>(static_cast<const float>(beta))},     \
        reinterpret_cast<const cutlass_type* const>(a),                        \
        reinterpret_cast<const cutlass_type* const>(b),                        \
        reinterpret_cast<const cutlass_type* const>(c),                        \
        reinterpret_cast<cutlass_type* const>(d),                              \
        typename Gemm::Base::LayoutA().capacity(problem_size_real.mk()),       \
        typename Gemm::Base::LayoutB().capacity(problem_size_real.kn()),       \
        typename Gemm::Base::LayoutC().capacity(problem_size_real.mn()),       \
        typename Gemm::Base::LayoutC().capacity(problem_size_real.mn()),       \
        std::is_same<typename Gemm::Base::LayoutA,                             \
                     cutlass::layout::RowMajor>::value                         \
            ? problem_size_real.k()                                            \
            : problem_size_real.m(),                                           \
        std::is_same<typename Gemm::Base::LayoutB,                             \
                     cutlass::layout::RowMajor>::value                         \
            ? problem_size_real.n()                                            \
            : problem_size_real.k(),                                           \
        std::is_same<typename Gemm::Base::LayoutC,                             \
                     cutlass::layout::RowMajor>::value                         \
            ? problem_size_real.n()                                            \
            : problem_size_real.m(),                                           \
        std::is_same<typename Gemm::Base::LayoutC,                             \
                     cutlass::layout::RowMajor>::value                         \
            ? problem_size_real.n()                                            \
            : problem_size_real.m(),                                           \
        a_indices,                                                             \
        b_indices,                                                             \
        c_d_indices};                                                          \
    size_t workspace_size = Gemm::get_workspace_size(arguments);               \
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);     \
    Gemm gemm_op;                                                              \
    cutlass::Status status = gemm_op.can_implement(arguments);                 \
    GATHER_GEMM_SCATTER_CHECK(status);                                         \
    status = gemm_op.initialize(arguments, workspace.get());                   \
    GATHER_GEMM_SCATTER_CHECK(status);                                         \
    gemm_op(dev_ctx.stream());                                                 \
    if (Config::Mode != cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) \
      return;                                                                  \
    fuck = 0;                                                                  \
    using ReductionOp = cutlass::reduction::thread::ReduceAdd<                 \
        typename Gemm::ElementAccumulator,                                     \
        typename Gemm::EpilogueOutputOp::ElementAccumulator,                   \
        Gemm::EpilogueOutputOp::kCount>;                                       \
    fuck2 = 0;                                                                 \
                                                                               \
    using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<          \
        cutlass::MatrixShape<4, 32 * Gemm::EpilogueOutputOp::kCount>,          \
        typename Gemm::EpilogueOutputOp,                                       \
        ReductionOp>;                                                          \
    fuck3 = 0;                                                                 \
                                                                               \
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle         \
        threadblock_swizzle;                                                   \
    fuck4 = 0;                                                                 \
    typename cutlass::gemm::GemmCoord grid_shape =                             \
        threadblock_swizzle.get_tiled_shape(problem_size_real,                 \
                                            {Gemm::ThreadblockShape::kM,       \
                                             Gemm::ThreadblockShape::kN,       \
                                             Gemm::ThreadblockShape::kK},      \
                                            split_k_slices);                   \
    fuck5 = 0;                                                                 \
                                                                               \
    cutlass::TensorRef<typename Gemm::ElementAccumulator,                      \
                       cutlass::layout::RowMajor>                              \
        ref_workspace(                                                         \
            static_cast<typename Gemm::ElementAccumulator*>(workspace),        \
            problem_size_real.n());                                            \
    fuck6 = 0;                                                                 \
    cutlass::TensorRef<Gemm const, Gemm::Base::LayoutC> ref_C(                 \
        reinterpret_cast<const cutlass_type* const>(c),                        \
        cutlass::layout::RowMajor);                                            \
    fuck7 = 0;                                                                 \
    cutlass::TensorRef<typename cutlass_type, cutlass::layout::RowMajor>       \
        ref_D(reinterpret_cast<cutlass_type* const>(d),                        \
              cutlass::layout::RowMajor);                                      \
    fuck8 = 0;                                                                 \
    int64_t partition_stride =                                                 \
        int64_t(problem_size_real.m()) * int64_t(problem_size_real.n());       \
    fuck9 = 0;                                                                 \
    typename ReductionKernel::Params reduction_params_;                        \
    fuck10 = 0;                                                                \
    reduction_params_ =                                                        \
        typename ReductionKernel::Params(problem_size_real.mn(),               \
                                         grid_shape.k(),                       \
                                         partition_stride,                     \
                                         ref_workspace,                        \
                                         ref_D,                                \
                                         ref_C.non_const_ref(),                \
                                         {alpha, beta});                       \
                                                                               \
    fuck11 = 0;                                                                \
    dim3 block = ReductionKernel::block_shape();                               \
    dim3 grid = ReductionKernel::grid_shape(gemm_params_.problem_size.mn());   \
    Kernel<ReductionKernel>                                                    \
        <<<grid, block, 0, dev_ctx.stream()>>>(reduction_params_);             \
  }

TYPEDEF_KERNEL_POINTER(fp16_gather_gemm_scatter, phi::dtype::float16)
TYPEDEF_KERNEL_POINTER(fp32_gather_gemm_scatter, float)

DEFINE_LAUNCH_KERNEL(phi::dtype::float16, cutlass::half_t)
DEFINE_LAUNCH_KERNEL(float, float)

}  // namespace sparse
}  // namespace phi
#endif
