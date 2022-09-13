/**************************************************************************************************
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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "multistage_testbed.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#if (CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 128x256x128_64x64x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 256x128x128_64x64x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 128x128x128_64x64x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 256x64x128_64x64x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 64, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 64x256x128_64x64x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 256, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 64x128x128_32x64x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 128>,
      cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 128x64x128_64x32x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 64, 128>,
      cutlass::gemm::GemmShape<64, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 64 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 64x64x128_32x32x128, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 128>,
      cutlass::gemm::GemmShape<32, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 64 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 128x256x64_64x64x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 256x128x64_64x64x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 128x128x64_64x64x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 256x64x64_64x64x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 64x256x64_64x64x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 256, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 64x128x64_32x64x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 64>,
      cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 128x64x64_64x32x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 64 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

CUTLASS_TEST_L0(SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32, 64x64x64_32x32x64, {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 64 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  test::gemm::device::MultistageTestbed<Gemm> testbed;

  EXPECT_TRUE(testbed.run_all());
} )

////////////////////////////////////////////////////////////////////////////////
#endif // #if (CUTLASS_ARCH_MMA_SM80_SUPPORTED)

