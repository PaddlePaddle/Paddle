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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/fused_geglu/device/dual_gemm.h"

namespace phi {

namespace fusion {

namespace cutlass_internal {

struct LaunchParams {
  const void* x;
  const void* weight;
  const void* bias;
  void* output;
  const int32_t m;
  const int32_t n;
  const int32_t k;
  bool requires_grad;
};

template <typename T, typename AccT, typename arch, typename Context>
void LaunchGeGLUKenrel(LaunchParams params, Context ctx) {
  constexpr int kStages = 3;  // TODO(test 5?)
  constexpr bool kSplitKSerial = false;
  constexpr bool kUseBias = true;
  constexpr auto kScaleType =
      kUseBias
          ? cutlass::epilogue::thread::ScaleType::NoBetaScaling
          : (
                // No bias
                kSplitKSerial ? cutlass::epilogue::thread::ScaleType::Default
                              : cutlass::epilogue::thread::ScaleType::Nothing);

  using ElementOperandA = T;
  using ElementOperandB = T;
  using ElementOutput = T;
  using ElementAccumulator = AccT;
  using ElementCompute = AccT;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  // maybe remove instruct shape
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // TODO(zhengzekang) set as false. Optionally, we might not need intermediate
  // GEMM outputs
  constexpr bool kStoreD0 = false;
  constexpr bool kStoreD1 = false;

  using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      kScaleType>;
  using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      kScaleType>;
  // TODO(zhengzekang): Fix it!
  using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementOutput,
      ElementCompute>;

  using DualGemm = cutlass::gemm::device::DualGemm<
      ElementOperandA,
      cutlass::layout::RowMajor,
      ElementOperandB,
      cutlass::layout::RowMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp0,
      EpilogueOutputOp1,
      EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      kStages,
      kStoreD0,
      kStoreD1,
      kSplitKSerial>;

  // int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
  int split_k_slices = 1;

  using TensorInputRef = TensorRef<ElementOperandA const, RowMajor>;
  using TensorStoreRef =
      TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>;
  using TensorOutRef = TensorRef<ElementOperandA, cutlass::layout::RowMajor>;

  TensorInputRef tensor_a0(reinterpret_cast<const T*>(params.x), params.k);
  TensorInputRef tensor_b0(reinterpret_cast<const T*>(params.weight), params.n);
  TensorInputRef tensor_b1(
      reinterpret_cast<const T*>(params.weight + params.n / 2), params.n);
  TensorInputRef tensor_bias0(reinterpret_cast<const T*>(params.bias), 0);
  TensorInputRef tensor_bias1(
      reinterpret_cast<const T*>(params.bias + params.n / 2), 0);
  TensorInputRef tensor_d0(reinterpret_cast<const T*>(nullptr, params.n / 2));
  TensorInputRef tensor_d1(reinterpret_cast<const T*>(nullptr, params.n / 2));
  TensorInputRef tensor_output(reinterpret_cast<T*>(params.output, n));

  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

  typename DualGemm::Arguments arguments{problem_size,
                                         tensor_a0,
                                         tensor_b0,
                                         tensor_bias0,
                                         tensor_d0,
                                         tensor_b1,
                                         tensor_bias1,
                                         tensor_d1,
                                         tensor_output,
                                         {alpha0, beta0},
                                         {alpha1, beta1},
                                         {},
                                         split_k_slices};

  DualGemm fused_geglu_op;
  cutlass::Status status = fused_geglu_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  fused_geglu_op.initialize(
      arguments, nullptr /*workspace todo*/, ctx.stream());
  CUTLASS_CHECK(status);
  status = fused_geglu_op();
  CUTLASS_CHECK(status);
}

template <typename T, typename Context>
void FusedGeGLUForwardKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& weight,
                             const DenseTensor& bias,
                             const std::string& act_type,
                             DenseTensor* output) {
  // TODO(zhengzekang): Allocate for d0 and d1?
  ctx.template Alloc<T>(output);

  //   struct LaunchParams{
  //     const void* x;
  //     const void* weight;
  //     const void* bias;
  //     void* output;
  //     const int32_t m;
  //     const int32_t n;
  //     const int32_t k;
  //     bool requires_grad;
  //   };
  LaunchParams params{};
  params.x = x.data();
  params.weight = weight.data();
  params.bias = bias.data();
  params.output = output->data();
  auto x_mat_dims = phi::flatten_to_2d(x.dims(), x.dims().size() - 1);
  const int64_t m = x_mat_dims.dims()[0];
  const int64_t k = x_mat_dims.dims()[1];
  const int64_t n = weight.dims()[1];
  PADDLE_ENFORCE_EQ(
      k,
      weight_dims()[0],
      phi::errors::InvalidArgument("The matmul dim is not matched, the x "
                                   "dim[1] should be equal to weight dim[0]"));
  params.m = m;
  params.m = n;
  params.k = k;
  // TODO(zhengzekang): fix it.
  params.requires_grad = false;

  LaunchGeGLUKenrel<cutlass::half_t, float, cutlass::arch::Sm86, Context>(
      params, ctx);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_geglu,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FusedGeGLUForwardKernel,
                   // float,
                   phi::dtype::float16) {}
