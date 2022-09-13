/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Unit tests for threadblock-level GEMM
*/

#include "mma_multistage_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x64x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x64x64_64x32x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x128x64_32x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x128x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_256x256x384_128x128x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_512x256x384_256x128x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x64x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x64x32_64x32x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x128x32_32x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x128x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_256x256x384_128x128x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_512x256x768_256x128x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x64x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x64x32_64x32x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x128x32_32x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x128x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_256x256x192_128x128x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_512x256x384_256x128x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x64x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x64x16_64x32x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x128x16_32x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x128x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_256x256x192_128x128x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_512x256x384_256x128x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x64_32x32x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x64_64x32x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x64_32x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x384_128x128x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x64_64x64x64_16x8x16_3stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x32_32x32x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x32_64x32x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x32_32x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x384_128x128x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x32_64x64x32_16x8x16_4stage) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x32_32x32x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x32_64x32x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x32_32x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x192_128x128x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x192_256x128x32_64x64x32_16x8x8_3stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x16_32x32x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x16_64x32x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x16_32x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x192_128x128x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x192_256x128x16_64x64x16_16x8x8_4stage) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x128_64x64x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x128_32x32x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x128_64x32x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x128_32x64x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x128_64x64x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x768_128x128x128_64x64x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x128_64x64x128_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x64_64x64x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x64_32x32x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x64_64x32x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x64_32x64x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x64_64x64x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x768_128x128x64_64x64x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x64_64x64x64_16x8x32_4stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x256_64x64x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 256>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x256_32x32x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 256>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x256_64x32x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 256>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x256x256_32x64x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 256>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x256x256_64x64x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 256>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x1536_128x256x256_64x64x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 1536);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 256>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x1536_256x256x256_64x64x256_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 1536);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 256>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x128_64x64x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x128_32x32x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x128_64x32x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x256x128_32x64x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x256x128_64x64x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 1024);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x1536_128x256x128_64x64x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 1536);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x1536_256x256x128_64x64x128_16x8x64_4stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 1536);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x1024_64x64x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x1024_32x32x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x1024_64x32x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x1024x1024_32x64x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x1024x1024_64x64x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x6144_128x1024x1024_64x64x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 6144);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x6144_256x1024x1024_64x64x1024_16x8x256_3stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 6144);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 1024>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 1024>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x512_64x64x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x64x512_32x32x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x64x512_64x32x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_64x128x512_32x64x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 512>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     tensor_op_128x128x512_64x64x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 4096);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x6144_128x128x512_64x64x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 6144);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x6144_256x128x512_64x64x512_16x8x256_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 6144);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_64x64x16_32x64x16_8x8x4_3stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 16);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 2, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     tensor_op_128x128x16_32x64x16_8x8x4_3stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     tensor_op_64x128x64_32x64x64_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<32>;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<32>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     tensor_op_128x128x64_64x64x64_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<32>;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<32>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     multicta_256x256x384_128x128x64_64x64x64_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<32>;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<32>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     multicta_512x256x384_256x128x64_64x64x64_16x8x32_3stage) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<32>;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<32>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     tensor_op_64x128x128_32x64x128_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<64>;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<64>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     tensor_op_128x128x128_64x64x128_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<64>;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<64>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     multicta_256x256x768_128x128x128_64x64x128_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<64>;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<64>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 768);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_interleaved,
     multicta_512x256x1536_256x128x128_64x64x128_16x8x64_3stage) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::ColumnMajorInterleaved<64>;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::RowMajorInterleaved<64>;
  using ElementC = int;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 1536);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise_f64,
     tensor_op_32x32x16_16x16x16_8x8x4_4stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(32, 32, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

TEST(SM80_gemm_threadblock_crosswise_f64,
     tensor_op_64x64x16_32x32x16_8x8x4_4stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

TEST(SM80_gemm_threadblock_crosswise_f64,
     tensor_op_64x128x16_32x64x16_8x8x4_4stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

TEST(SM80_gemm_threadblock_crosswise_f64,
     tensor_op_128x64x16_64x32x16_8x8x4_4stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

TEST(SM80_gemm_threadblock_crosswise_f64,
     tensor_op_128x128x16_32x64x16_8x8x4_3stage) {
  using ElementA = double;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = double;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  int const Stages = 3;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

#endif
