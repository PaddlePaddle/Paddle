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
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"


#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed_interleaved.h"

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  256x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  128x256_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  256x64_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  64x256_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  128x64_64x2_64x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  64x128_64x2_32x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Analytic_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  64x64_64x2_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  256x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  128x256_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  256x64_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  64x256_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  128x64_64x2_64x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  64x128_64x2_32x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2, 
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv2d_Fprop_Optimized_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
  64x64_64x2_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2d<Conv2dFprop, 32>()));
}

////////////////////////////////////////////////////////////////////////////////
#endif  // CUTLASS_ARCH_MMA_SM75_SUPPORTED
