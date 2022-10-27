// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/sparse/gpu/gather_gemm_scatter.h"
namespace phi {
namespace sparse {
fp16_gather_gemm_scatter getBestFp16Kernel(const int M,
                                           const int N,
                                           const int K) {
  if (K == 4 && N == 16) {
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4;
  }
  if (K == 16 && N == 16) {
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
  }
  if (K == 16 && N == 32) {
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
  }
  if (K == 32 && N == 32) {
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
  }
  if (K == 32 && N == 64) {
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
  }
  if (K == 64 && N == 64) {
    if (M > 100000) cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_nn_align8;
    if (M > 20000) cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align8;
    if (M > 15000) return cutlass_tensorop_h1688gemm_128x64_32x2_nn_align8;
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
  }
  if (K == 128) {
    if (M >= 5000) return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
    return cutlass_tensorop_h16816gemm_64x64_64x5_nn_align8;
  }
  if (N == 128) {
    return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8;
  }
  return cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4;
}
fp32_gather_gemm_scatter getBestFp32Kernel(const int M,
                                           const int N,
                                           const int K) {
  if (K == 4 && N == 16) {
    return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
  }
  if (K == 16 && N == 16) {
    return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
  }
  if (K == 16 && N == 32) {
    if (M >= 10000) return cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4;
    return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
  }
  if (K == 32 && N == 32) {
    if (M >= 10000) return cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4;
    return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
  }
  if (K == 32 && N == 64) {
    if (M >= 10000) return cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4;
    return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
  }
  if (K == 64 && N == 64) {
    if (M >= 15000) return cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4;
    return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
  }
  if (K == 128) {
    if (M >= 100000)
      return cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4;
    if (M >= 5000) return cutlass_tensorop_s1688f16gemm_256x64_16x4_nn_align4;
    return cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4;
  }
  if (N == 128) {
    if (M >= 100000)
      return cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4;
    if (M >= 5000) return cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4;
    return cutlass_tensorop_s1688f16gemm_64x128_16x6_nn_align4;
  }
  return cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4;
}
fp64_gather_gemm_scatter getBestFp64Kernel(const int M,
                                           const int N,
                                           const int K) {
  if (K == 4 && N == 16) {
    return cutlass_tensorop_d884gemm_16x32_16x5_nn_align1;
  }
  if (K == 16 && N == 16) {
    if (M >= 10000) return cutlass_tensorop_d884gemm_32x16_16x5_nn_align1;
    return cutlass_tensorop_d884gemm_16x32_16x5_nn_align1;
  }
  if (K == 16 && N == 32) {
    return cutlass_tensorop_d884gemm_32x16_16x5_nn_align1;
  }
  if (K == 32 && N == 32) {
    return cutlass_tensorop_d884gemm_16x32_16x5_nn_align1;
  }
  if (K == 32 && N == 64) {
    return cutlass_tensorop_d884gemm_32x16_16x5_nn_align1;
  }
  if (K == 64 && N == 64) {
    return cutlass_tensorop_d884gemm_32x16_16x5_nn_align1;
  }
  return cutlass_tensorop_d884gemm_32x16_16x5_nn_align1;
}
void cutlass_tensorop_h1688gemm_128x64_32x2_nn_align8(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      8,
      cutlass::half_t,
      cutlass::half_t>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<128, 64, 32>,
      cutlass::gemm::GemmShape<64, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_h1688gemm_64x128_32x2_nn_align8(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      8,
      cutlass::half_t,
      cutlass::half_t>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<64, 128, 32>,
      cutlass::gemm::GemmShape<32, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_h1688gemm_128x64_32x2_nn_align4(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      4,
      cutlass::half_t,
      cutlass::half_t>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<128, 64, 32>,
      cutlass::gemm::GemmShape<64, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      2,
      4,
      4,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      4,
      cutlass::half_t,
      cutlass::half_t>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<64, 64, 32>,
                                           cutlass::gemm::GemmShape<32, 32, 32>,
                                           cutlass::gemm::GemmShape<16, 8, 8>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           2,
                                           4,
                                           4,
                                           cutlass::arch::OpMultiplyAdd,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      8,
      cutlass::half_t,
      cutlass::half_t>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<64, 64, 32>,
                                           cutlass::gemm::GemmShape<32, 32, 32>,
                                           cutlass::gemm::GemmShape<16, 8, 8>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           2,
                                           8,
                                           8,
                                           cutlass::arch::OpMultiplyAdd,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_h16816gemm_64x64_64x5_nn_align8(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      8,
      cutlass::half_t,
      cutlass::half_t>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<64, 64, 64>,
                                           cutlass::gemm::GemmShape<32, 32, 64>,
                                           cutlass::gemm::GemmShape<16, 8, 16>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           5,
                                           8,
                                           8,
                                           cutlass::arch::OpMultiplyAdd,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
}
void cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_nn_align8(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      8,
      float,
      float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<64, 128, 32>,
      cutlass::gemm::GemmShape<32, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align8(
    const GPUContext& dev_ctx,
    const cutlass::half_t* const a,
    const cutlass::half_t* const b,
    const cutlass::half_t* const c,
    cutlass::half_t* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    cutlass::half_t const alpha,
    cutlass::half_t const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm75;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,  // <- data type of output matrix
      8,
      float,
      float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           float,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<64, 64, 32>,
                                           cutlass::gemm::GemmShape<32, 32, 32>,
                                           cutlass::gemm::GemmShape<16, 8, 8>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           2,
                                           8,
                                           8,
                                           cutlass::arch::OpMultiplyAdd,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4(
    const GPUContext& dev_ctx,
    const float* const a,
    const float* const b,
    const float* const c,
    float* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    float const alpha,
    float const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<float,  // <- data type of
                                                           // output matrix
                                                   4,
                                                   float,
                                                   float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<float,
                                           cutlass::layout::RowMajor,
                                           float,
                                           cutlass::layout::RowMajor,
                                           float,
                                           cutlass::layout::RowMajor,
                                           float,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<64, 64, 16>,
                                           cutlass::gemm::GemmShape<32, 32, 16>,
                                           cutlass::gemm::GemmShape<16, 8, 8>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           10,
                                           4,
                                           4,
                                           cutlass::arch::OpMultiplyAddFastF16,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4(
    const GPUContext& dev_ctx,
    const float* const a,
    const float* const b,
    const float* const c,
    float* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    float const alpha,
    float const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<float,  // <- data type of
                                                           // output matrix
                                                   4,
                                                   float,
                                                   float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<128, 128, 16>,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      3,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_s1688f16gemm_256x64_16x4_nn_align4(
    const GPUContext& dev_ctx,
    const float* const a,
    const float* const b,
    const float* const c,
    float* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    float const alpha,
    float const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<float,  // <- data type of
                                                           // output matrix
                                                   4,
                                                   float,
                                                   float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<256, 64, 16>,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      4,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4(
    const GPUContext& dev_ctx,
    const float* const a,
    const float* const b,
    const float* const c,
    float* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    float const alpha,
    float const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<float,  // <- data type of
                                                           // output matrix
                                                   4,
                                                   float,
                                                   float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<256, 128, 16>,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      3,
      4,
      4,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_s1688f16gemm_64x128_16x6_nn_align4(
    const GPUContext& dev_ctx,
    const float* const a,
    const float* const b,
    const float* const c,
    float* const d,
    const int m,
    const int n,
    const int k,
    const int32_t* a_indices,
    const int32_t* c_d_indices,
    float const alpha,
    float const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<float,  // <- data type of
                                                           // output matrix
                                                   4,
                                                   float,
                                                   float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      MMAOp,
      SmArch,
      cutlass::gemm::GemmShape<64, 128, 16>,
      cutlass::gemm::GemmShape<32, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOp,
      SwizzleThreadBlock,
      6,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,  /*GatherA*/
      false, /*GatherB*/
      true   /*ScatterD*/
      >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
#if 0
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
#endif
}
void cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4(const GPUContext& dev_ctx,
                                                     const float* const a,
                                                     const float* const b,
                                                     const float* const c,
                                                     float* const d,
                                                     const int m,
                                                     const int n,
                                                     const int k,
                                                     const int32_t* a_indices,
                                                     const int32_t* c_d_indices,
                                                     float const alpha,
                                                     float const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<float,  // <- data type of
                                                           // output matrix
                                                   4,
                                                   float,
                                                   float>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<float,
                                           cutlass::layout::RowMajor,
                                           float,
                                           cutlass::layout::RowMajor,
                                           float,
                                           cutlass::layout::RowMajor,
                                           float,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<64, 64, 16>,
                                           cutlass::gemm::GemmShape<32, 32, 16>,
                                           cutlass::gemm::GemmShape<16, 8, 8>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           3,
                                           4,
                                           4,
                                           cutlass::arch::OpMultiplyAddFastF32,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
}
void cutlass_tensorop_d884gemm_16x32_16x5_nn_align1(const GPUContext& dev_ctx,
                                                    const double* const a,
                                                    const double* const b,
                                                    const double* const c,
                                                    double* const d,
                                                    const int m,
                                                    const int n,
                                                    const int k,
                                                    const int32_t* a_indices,
                                                    const int32_t* c_d_indices,
                                                    double const alpha,
                                                    double const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<double,  // <- data type of
                                                            // output matrix
                                                   1,
                                                   double,
                                                   double>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<double,
                                           cutlass::layout::RowMajor,
                                           double,
                                           cutlass::layout::RowMajor,
                                           double,
                                           cutlass::layout::RowMajor,
                                           double,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<16, 32, 16>,
                                           cutlass::gemm::GemmShape<16, 16, 16>,
                                           cutlass::gemm::GemmShape<8, 8, 4>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           5,
                                           1,
                                           1,
                                           cutlass::arch::OpMultiplyAdd,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
}
void cutlass_tensorop_d884gemm_32x16_16x5_nn_align1(const GPUContext& dev_ctx,
                                                    const double* const a,
                                                    const double* const b,
                                                    const double* const c,
                                                    double* const d,
                                                    const int m,
                                                    const int n,
                                                    const int k,
                                                    const int32_t* a_indices,
                                                    const int32_t* c_d_indices,
                                                    double const alpha,
                                                    double const beta) {
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<double,  // <- data type of
                                                            // output matrix
                                                   1,
                                                   double,
                                                   double>;

  // Number of pipelines you want to use
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm =
      cutlass::gemm::device::GemmUniversal<double,
                                           cutlass::layout::RowMajor,
                                           double,
                                           cutlass::layout::RowMajor,
                                           double,
                                           cutlass::layout::RowMajor,
                                           double,
                                           MMAOp,
                                           SmArch,
                                           cutlass::gemm::GemmShape<32, 16, 16>,
                                           cutlass::gemm::GemmShape<16, 16, 16>,
                                           cutlass::gemm::GemmShape<8, 8, 4>,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           5,
                                           1,
                                           1,
                                           cutlass::arch::OpMultiplyAdd,
                                           cutlass::ComplexTransform::kNone,
                                           cutlass::ComplexTransform::kNone,
                                           true,  /*GatherA*/
                                           false, /*GatherB*/
                                           true   /*ScatterD*/
                                           >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,  // <- problem size of matrix multiplication
      split_k_slices,     // <- k-dimension split factor
      {alpha, beta},      // <- alpha, beta
      a,                  // <- reference to matrix A on device
      b,                  // <- reference to matrix B on device
      c,                  // <- reference to matrix C on device
      d,                  // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,     // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};  // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation

  status = gemm_op(dev_ctx.stream());
}
}  // namespace sparse
}  // namespace phi
