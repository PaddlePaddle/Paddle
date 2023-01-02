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

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename TypeA, typename TypeB, typename arch>
struct MoeArchTraits {};

template <typename arch>
struct MoeArchTraits<float, float, arch> {
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassSimt;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA = 1;
  static constexpr int ElementsPerAccessB = 1;
  static constexpr int ElementsPerAccessC = 1;
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

// ========================= Volta Traits ===========================
// Volta will always dequantize after the global memory load.
template <typename TypeB>
struct MoeArchTraits<cutlass::half_t, TypeB, cutlass::arch::Sm70> {
 private:
  static constexpr int ThreadblockK = 32;

 public:
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  static constexpr int ElementsPerAccessB =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename TypeB>
struct MoeArchTraits<cutlass::bfloat16_t, TypeB, cutlass::arch::Sm70> {
 private:
  static constexpr int ThreadblockK = 32;

 public:
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int ElementsPerAccessB =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

// ======================= Turing Traits ==============================
// Turing will dequantize after LDSM

// fp16 x fp16 specialization
template <>
struct MoeArchTraits<cutlass::half_t, cutlass::half_t, cutlass::arch::Sm75> {
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  static constexpr int ElementsPerAccessB =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

// bf16 x bf16 specialization
template <>
struct MoeArchTraits<cutlass::bfloat16_t,
                     cutlass::bfloat16_t,
                     cutlass::arch::Sm75> {
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int ElementsPerAccessB =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <>
struct MoeArchTraits<float, float, cutlass::arch::Sm80> {
  static constexpr int Stages = 3;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA = 4;
  static constexpr int ElementsPerAccessB = 4;
  static constexpr int ElementsPerAccessC = 4;
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <>
struct MoeArchTraits<cutlass::half_t, cutlass::half_t, cutlass::arch::Sm80> {
  static constexpr int Stages = 3;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  static constexpr int ElementsPerAccessB =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<cutlass::half_t>::value;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <>
struct MoeArchTraits<cutlass::bfloat16_t,
                     cutlass::bfloat16_t,
                     cutlass::arch::Sm80> {
  static constexpr int Stages = 3;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int ElementsPerAccessB =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
