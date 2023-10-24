/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/arch/mma.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename TypeA, typename TypeB, typename arch, typename Enable = void>
struct MixedGemmArchTraits {};

template <typename arch>
struct MixedGemmArchTraits<float, float, arch> {
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassSimt;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA = 1;
  static constexpr int ElementsPerAccessB = 1;
  static constexpr int ElementsPerAccessC = 1;
  static constexpr int ThreadblockK = 8;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

// // ========================= Volta Traits ===========================
// // Volta will always dequantize after the global memory load.
// // This will instantiate any HMMA tensorcore kernels for Volta.
// // Note that volta does not have native bfloat support so weights and
// activations will be casted to fp16
// // and compute will happen in fp16 then will be converted for bf16 output.
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm70,
    typename cutlass::platform::enable_if<
        cutlass::platform::is_same<TypeA, cutlass::half_t>::value ||
        cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
 private:
  using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm70>;

 public:
  static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = typename LayoutDetails::Layout;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<TypeA>::value;
  static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<TypeA>::value;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  using Operator = typename LayoutDetails::Operator;
};

// ======================= Turing Traits ==============================
// Note that turing does not have native bfloat support so weights and
// activations will be casted to fp16 and compute will happen in fp16 then will
// be converted for bf16 output.
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm75,
    typename cutlass::platform::enable_if<
        cutlass::platform::is_same<TypeA, cutlass::half_t>::value ||
        cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
 private:
  using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm75>;

 public:
  static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = typename LayoutDetails::Layout;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<TypeA>::value;
  static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<TypeA>::value;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Operator = typename LayoutDetails::Operator;
};

// ======================= Ampere Traits ==============================
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm80,
    typename cutlass::platform::enable_if<
        cutlass::platform::is_same<TypeA, cutlass::half_t>::value ||
        cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
 private:
  using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm80>;

 public:
  static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = typename LayoutDetails::Layout;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<TypeA>::value;
  static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<TypeA>::value;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Operator = typename LayoutDetails::Operator;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
