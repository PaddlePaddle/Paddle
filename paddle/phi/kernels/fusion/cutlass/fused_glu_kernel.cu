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

#include "paddle/phi/kernels/fusion/cutlass/cutlass_helper.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_glu/device/dual_gemm.h"
// #include
// "paddle/phi/kernels/fusion/cutlass/fused_glu/thread/right_act_and_mul.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_glu/kernel/dual_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/fused_glu/autogen/cutlass_dual_gemm.h"

DECLARE_bool(gemm_use_half_precision_compute_type);

namespace phi {

namespace fusion {

namespace cutlass_internal {

struct LaunchParams {
  phi::DataType dtype;
  const void* x;
  const void* weight;
  const void* bias;
  std::string act_type;
  void* output;
  void* store_d0;
  void* store_d1;
  int32_t m;
  int32_t n;
  int32_t k;
  bool requires_grad;
};

// template <typename T,
//           typename AccT,
//           typename Arch,
//           bool StoreD,
//           template <typename>
//           class ActivationType>
// void LaunchGeGLUKenrel(LaunchParams params, const phi::GPUContext& ctx) {
//   constexpr int kStages = 3;
//   constexpr bool kSplitKSerial = false;
//   constexpr bool kUseBias = true;
//   constexpr auto kScaleType =
//       cutlass::epilogue::thread::ScaleType::NoBetaScaling;

//   using ElementOperandA = T;
//   using ElementOperandB = T;
//   using ElementOutput = T;
//   using ElementAccumulator = AccT;
//   using ElementCompute = AccT;

//   using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput,
//       128 / cutlass::sizeof_bits<ElementOutput>::value,
//       ElementAccumulator,
//       ElementCompute,
//       kScaleType>;
//   using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput,
//       128 / cutlass::sizeof_bits<ElementOutput>::value,
//       ElementAccumulator,
//       ElementCompute,
//       kScaleType>;

//   using EpilogueOutputOp2 = cutlass::epilogue::thread::RightActAndMul<
//       ActivationType,
//       ElementOutput,
//       128 / cutlass::sizeof_bits<ElementOutput>::value,
//       ElementOutput,
//       ElementCompute>;

//   using DualGemm = cutlass::gemm::device::DualGemm<
//       ElementOperandA,
//       cutlass::layout::RowMajor,
//       ElementOperandB,
//       cutlass::layout::RowMajor,
//       ElementOutput,
//       cutlass::layout::RowMajor,
//       ElementAccumulator,
//       Arch,
//       EpilogueOutputOp0,
//       EpilogueOutputOp1,
//       EpilogueOutputOp2,
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
//       kStages,
//       StoreD,
//       StoreD,
//       kSplitKSerial>;

//   int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;

//   using TensorInputRef = typename cutlass::TensorRef<ElementOperandA const,
//                                                      cutlass::layout::RowMajor>;
//   using TensorStoreRef =
//       typename cutlass::TensorRef<typename DualGemm::ElementC,
//                                   typename DualGemm::LayoutC>;
//   using TensorOutRef =
//       typename cutlass::TensorRef<ElementOperandA,
//       cutlass::layout::RowMajor>;

//   TensorInputRef tensor_a0(reinterpret_cast<const T*>(params.x), params.k);
//   TensorInputRef tensor_b0(reinterpret_cast<const T*>(params.weight),
//                            params.n * 2);
//   TensorInputRef tensor_b1(reinterpret_cast<const T*>(params.weight) +
//   params.n,
//                            params.n * 2);

//   TensorInputRef tensor_bias0(reinterpret_cast<const T*>(params.bias), 0);
//   TensorInputRef tensor_bias1(
//       reinterpret_cast<const T*>(params.bias) + params.n, 0);

//   TensorStoreRef tensor_d0(reinterpret_cast<T*>(params.store_d0), params.n);
//   TensorStoreRef tensor_d1(reinterpret_cast<T*>(params.store_d1), params.n);

//   TensorOutRef tensor_output(reinterpret_cast<T*>(params.output), params.n);

//   cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

//   const ElementCompute alpha0 = ElementCompute(1);
//   const ElementCompute beta0 = ElementCompute(kUseBias ? 1 : 0);
//   const ElementCompute alpha1 = ElementCompute(1);
//   const ElementCompute beta1 = ElementCompute(kUseBias ? 1 : 0);

//   typename DualGemm::Arguments arguments{problem_size,
//                                          tensor_a0,
//                                          tensor_b0,
//                                          tensor_bias0,
//                                          tensor_d0,
//                                          tensor_b1,
//                                          tensor_bias1,
//                                          tensor_d1,
//                                          tensor_output,
//                                          {alpha0, beta0},
//                                          {alpha1, beta1},
//                                          {},
//                                          split_k_slices};

//   DualGemm fused_geglu_op;
//   paddle::memory::AllocationPtr workspace_ptr{nullptr};
//   workspace_ptr = paddle::memory::Alloc(
//       ctx.GetPlace(),
//       fused_geglu_op.get_workspace_size(arguments),
//       phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));

//   cutlass::Status status = fused_geglu_op.can_implement(arguments);
//   PD_CUTLASS_CHECK(status);
//   status =
//       fused_geglu_op.initialize(arguments, workspace_ptr->ptr(),
//       ctx.stream());
//   PD_CUTLASS_CHECK(status);
//   status = fused_geglu_op(ctx.stream());
//   PD_CUTLASS_CHECK(status);
// }

template <typename T,
          typename AccT,
          typename Arch,
          bool StoreD,
          template <typename>
          class ActivationType>
void LaunchGeGLUKenrel(LaunchParams params, const phi::GPUContext& ctx) {
  constexpr int kStages = 3;
  constexpr bool kSplitKSerial = false;
  constexpr bool kUseBias = true;
  constexpr auto kScaleType =
      cutlass::epilogue::thread::ScaleType::NoBetaScaling;

  using ElementOperandA = T;
  using ElementOperandB = T;
  using ElementOutput = T;
  using ElementAccumulator = AccT;
  using ElementCompute = AccT;

  using DualGemmKernel = cutlass::gemm::kernel::DualGemm<ElementOperandA,
                                                         ElementAccumulator,
                                                         StoreD,
                                                         ActivationType,
                                                         Arch>;

  int split_k_slices = DualGemmKernel::kSplitKSerial ? 2 : 1;

  using TensorInputRef = typename cutlass::TensorRef<ElementOperandA const,
                                                     cutlass::layout::RowMajor>;
  using TensorStoreRef =
      typename cutlass::TensorRef<typename DualGemmKernel::ElementA,
                                  typename DualGemmKernel::LayoutC>;
  using TensorOutRef =
      typename cutlass::TensorRef<ElementOperandA, cutlass::layout::RowMajor>;

  TensorInputRef tensor_a0(reinterpret_cast<const T*>(params.x), params.k);
  TensorInputRef tensor_b0(reinterpret_cast<const T*>(params.weight),
                           params.n * 2);
  TensorInputRef tensor_b1(reinterpret_cast<const T*>(params.weight) + params.n,
                           params.n * 2);

  TensorInputRef tensor_bias0(reinterpret_cast<const T*>(params.bias), 0);
  TensorInputRef tensor_bias1(
      reinterpret_cast<const T*>(params.bias) + params.n, 0);

  TensorStoreRef tensor_d0(reinterpret_cast<T*>(params.store_d0), params.n);
  TensorStoreRef tensor_d1(reinterpret_cast<T*>(params.store_d1), params.n);

  TensorOutRef tensor_output(reinterpret_cast<T*>(params.output), params.n);

  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
  printf("problem size: %d, %d, %d. \n", params.m, params.n, params.k);
  const ElementCompute alpha0 = ElementCompute(1);
  const ElementCompute beta0 = ElementCompute(kUseBias ? 1 : 0);
  const ElementCompute alpha1 = ElementCompute(1);
  const ElementCompute beta1 = ElementCompute(kUseBias ? 1 : 0);

  printf("=====1");
  cudaDeviceSynchronize();
  // Initialize the Params structure
  typename DualGemmKernel::Params p{
      problem_size,
      split_k_slices,
      tensor_a0.non_const_ref(),
      tensor_b0.non_const_ref(),
      tensor_bias0.non_const_ref(),
      tensor_d0,
      tensor_b1.non_const_ref(),
      tensor_bias1.non_const_ref(),
      tensor_d1,
      tensor_output,
      {alpha0, beta0},
      {alpha1, beta1},
      {},
      nullptr,
  };

  printf("=====2");
  cudaDeviceSynchronize();

  paddle::memory::AllocationPtr workspace_ptr{nullptr};
  workspace_ptr = paddle::memory::Alloc(
      ctx.GetPlace(),
      p.get_workspace_size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  p.set_workspace_ptr(reinterpret_cast<int*>(workspace_ptr->ptr()));

  constexpr auto kernel_fn = DualKernel<DualGemmKernel>;

  cudaError_t result;
  printf("=====3");
  cudaDeviceSynchronize();

  int smem_size =
      static_cast<int>(sizeof(typename DualGemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    result = cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    if (result != cudaSuccess) {
      // return Status::kErrorInternal;
    }
  }

  printf("=====4");
  printf("smem_size is: %d\n", smem_size);
  cudaDeviceSynchronize();
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_size, ctx.stream()>>>(
      p);
}

// template <typename T, typename AccT, typename Arch, bool StoreD>
// void DispatchActivationType(LaunchParams params, const phi::GPUContext& ctx)
// {
//   if (params.act_type == "sigmoid") {
//     return LaunchGeGLUKenrel<T,
//                              AccT,
//                              Arch,
//                              StoreD,
//                              cutlass::epilogue::thread::Sigmoid>(params,
//                              ctx);
//   } else if (params.act_type == "swish") {
//     return LaunchGeGLUKenrel<T,
//                              AccT,
//                              Arch,
//                              StoreD,
//                              cutlass::epilogue::thread::SiLu>(params, ctx);
//   } else if (params.act_type == "gelu") {
//     // TODO(zhengzekang): Here we follow cublasLt to use fast gelu to
//     compute,
//     // it's faster but the accuracy maybe lower(it use tanh approximate
//     // algorithm).
//     return LaunchGeGLUKenrel<T,
//                              AccT,
//                              Arch,
//                              StoreD,
//                              cutlass::epilogue::thread::GELU_taylor>(params,
//                                                                      ctx);
//   } else {
//     PADDLE_THROW(phi::errors::Unimplemented(
//         "Currently FusedGLU Kernel only accept act_type with `sigmoid`, "
//         "`swish`, `gelu`. "));
//     return;
//   }
// }

// template <typename T, typename AccT, typename Arch>
// void DispatchStoreForBackward(LaunchParams params, const phi::GPUContext&
// ctx) {
//   if (params.requires_grad) {
//     // If x requires grad, we need to save intermediate matmul result.
//     return DispatchActivationType<T, AccT, Arch, true>(params, ctx);
//   } else {
//     return DispatchActivationType<T, AccT, Arch, false>(params, ctx);
//   }
// }

// template <typename T, typename AccT>
// void DispatchArch(LaunchParams params, const phi::GPUContext& ctx) {
//   const int compute_capability = ctx.GetComputeCapability();
//   if (compute_capability == 70) {
//     return DispatchStoreForBackward<T, AccT, cutlass::arch::Sm70>(params,
//     ctx);
//   } else if (compute_capability >= 75) {
//     return DispatchStoreForBackward<T, AccT, cutlass::arch::Sm75>(params,
//     ctx);
//   } else if (compute_capability >= 80 && compute_capability < 90) {
//     return DispatchStoreForBackward<T, AccT, cutlass::arch::Sm80>(params,
//     ctx);
//   } else {
//     PADDLE_THROW(phi::errors::Unimplemented(
//         "Currently cutlass FusedGLU kernel only support SM70, SM75, SM80.
//         "));
//     return;
//   }
// }

// template <typename T>
// void DispatchAccumulateType(LaunchParams params, const phi::GPUContext& ctx)
// {
//   if (std::is_same<T, cutlass::half_t>::value &&
//       FLAGS_gemm_use_half_precision_compute_type) {
//     return DispatchArch<cutlass::half_t, cutlass::half_t>(params, ctx);
//   } else {
//     return DispatchArch<T, float>(params, ctx);
//   }
// }

// void DispatchFusedGLUKernel(LaunchParams params, const phi::GPUContext& ctx)
// {
//   if (params.dtype == DataType::FLOAT16) {
//     return DispatchAccumulateType<cutlass::half_t>(params, ctx);
//   } else {
//     PADDLE_THROW(
//         phi::errors::Unimplemented("Currently cutlass FusedGLU kernel "
//                                    "only support float16 datatype. "));
//     return;
//   }
// }

template <typename T, typename Context>
void FusedGLUForwardKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& weight,
                           const DenseTensor& bias,
                           const std::string& act_type,
                           const bool requires_grad,
                           DenseTensor* output,
                           DenseTensor* matmul_result0,
                           DenseTensor* matmul_result1) {
  ctx.template Alloc<T>(output);
  if (requires_grad) {
    // When x requires grad, we need to save the intermediate result.
    ctx.template Alloc<T>(matmul_result0);
    ctx.template Alloc<T>(matmul_result1);
  }
  LaunchParams params{};
  params.dtype = x.dtype();
  params.x = x.data();
  params.weight = weight.data();
  params.bias = bias.data();
  params.act_type = act_type;
  params.output = output->data();
  params.store_d0 = matmul_result0->data();
  params.store_d1 = matmul_result1->data();

  auto x_mat_dims = phi::flatten_to_2d(x.dims(), x.dims().size() - 1);
  const int64_t m = x_mat_dims[0];
  const int64_t k = x_mat_dims[1];
  const int64_t n = weight.dims()[1];
  PADDLE_ENFORCE_EQ(
      k,
      weight.dims()[0],
      phi::errors::InvalidArgument("The matmul dim is not matched, the x "
                                   "dim[1] should be equal to weight dim[0]"));
  params.m = m;
  params.n = n / 2;
  params.k = k;
  params.requires_grad = requires_grad;
  // DispatchFusedGLUKernel(params, ctx);

  //   LaunchGeGLUKenrel<cutlass::half_t, float, cutlass::arch::Sm80, false,
  //   cutlass::epilogue::thread::GELU_taylor>(params, ctx);
  LaunchGeGLUKenrel<cutlass::half_t,
                    float,
                    cutlass::arch::Sm80,
                    false,
                    cutlass::epilogue::thread::SiLu>(params, ctx);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_glu,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FusedGLUForwardKernel,
                   phi::dtype::float16) {}
