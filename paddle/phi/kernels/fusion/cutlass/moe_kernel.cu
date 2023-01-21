// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/fusion/moe_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/fusion/cutlass/moe_kernel_impl.h"

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/numeric_conversion.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/fusion/cutlass/default_moe_fc_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/linear_combination_ft_gelu.h"
#include "paddle/phi/kernels/fusion/cutlass/moe_cutlass_kernel.h"
#pragma GCC diagnostic pop
namespace phi {

namespace {
inline int getSMVersion() {
  const int device = phi::backends::gpu::GetCurrentDeviceId();
  const phi::gpuDeviceProp prop =
      phi::backends::gpu::GetDeviceProperties(device);
  return prop.major * 10 + prop.minor;
}

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpNoBias {};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator,
          typename Op>
struct Epilogue {};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBiasReLU> {
  using Op = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBiasFtGelu> {
  using Op = cutlass::epilogue::thread::LinearCombinationFtGelu<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpNoBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::Nothing>;
};

}  // namespace

namespace fusion {

template <typename T>
void InitExpertChoiceRouteKernelLauncher(
    int* expert_for_source_row,
    int* source_row,
    int* expanded_source_row_to_expanded_dest_row,
    int64_t* total_rows_before_expert,
    T* attr_mask,
    const int num_experts,
    const int num_rows,
    const int k,
    const int batch_size,
    cudaStream_t stream) {
  const int threads = 128;
  const int blocks = num_experts;

  initialize_expert_choice_route_kernel<<<blocks, threads, 0, stream>>>(
      expert_for_source_row,
      source_row,
      expanded_source_row_to_expanded_dest_row,
      total_rows_before_expert,
      attr_mask,
      num_rows,
      k,
      batch_size);
}

#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                \
  block.x /= ITEMS_PER_THREAD;                                          \
  assert(block.x <= 1024);                                              \
  if (is_half2) {                                                       \
    if (grid.x % 4 == 0) {                                              \
      grid.x /= 4;                                                      \
      softmax_kernel_v5_half2<__half, ITEMS_PER_THREAD, 4>              \
          <<<grid, block, 0, stream>>>(reinterpret_cast<half*>(buffer), \
                                       (const half*)attr_mask,          \
                                       batch_size,                      \
                                       seq_len);                        \
    } else {                                                            \
      softmax_kernel_v4_half2<__half, ITEMS_PER_THREAD>                 \
          <<<grid, block, 0, stream>>>(reinterpret_cast<half*>(buffer), \
                                       (const half*)attr_mask,          \
                                       batch_size,                      \
                                       seq_len);                        \
    }                                                                   \
  } else {                                                              \
    softmax_kernel_v4<ITEMS_PER_THREAD, T><<<grid, block, 0, stream>>>( \
        buffer, buffer_src, attr_mask, batch_size, seq_len);            \
  }

template <typename T>
void invokeMaskedSoftMax(T* buffer,
                         const T* buffer_src,
                         const T* attr_mask,
                         const int batch_size,
                         const int seq_len,
                         cudaStream_t stream) {
  // NOTE: attention scores shape (batch_size, seq_len)
  dim3 grid(1, batch_size, 1);
  if (batch_size > 360) {
    grid.x = ceil(static_cast<float>(1) / 32.0f);
  }

  bool is_half2 = sizeof(T) == 2 && sizeof(T) == 2 && seq_len % 2 == 0;
  dim3 block((seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

  if (block.x > 2048 && block.x <= 4096) {
    SOFTMAX_KERNEL(4)
  } else if (block.x > 1024) {
    SOFTMAX_KERNEL(2)
  } else if (block.x > 0) {
    SOFTMAX_KERNEL(1)
  } else {
    PADDLE_ENFORCE_EQ(true,
                      false,
                      phi::errors::InvalidArgument(
                          "Softmax kernel only support columns in 0 - 4096. "));
  }
}

template <typename T>
void InvokeTransposeAxis01(T* out,
                           T* in,
                           const int dim0,
                           const int dim1,
                           const int dim2,
                           cudaStream_t stream) {
  dim3 block(512);
  dim3 grid(static_cast<int>(ceil(dim0 * dim1 * dim2 / 512.)));
  transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
}

template <typename T>
void InvokePadding(T* output1,
                   int* output2,
                   const T* input1,
                   const int* input2,
                   const int* input_lengths,
                   const int num_tokens,
                   const int batch_size,
                   const int max_seq_len,
                   const int num_experts,
                   cudaStream_t stream) {
  assert(max_seq_len <= 1024);
  dim3 block(max_seq_len);
  dim3 grid(num_experts);
  paddingKernel<<<grid, block, 0, stream>>>(output1,
                                            output2,
                                            input1,
                                            input2,
                                            input_lengths,
                                            num_tokens,
                                            batch_size,
                                            max_seq_len,
                                            num_experts);
}

template <typename T>
void InvokeGeneralTopKPairSort(T* out_keys,
                               int* out_values,
                               T* in_keys,
                               int* in_values,
                               const int m,
                               const int n,
                               cudaStream_t stream) {
  assert(n <= 4096);
  const int blocks = m;

  if (n == 128) {
    general_topk_pair_sort<T, 32, 4>
        <<<blocks, 32, 0, stream>>>(out_keys, out_values, in_keys, in_values);
  }
  if (n == 256) {
    general_topk_pair_sort<T, 64, 4>
        <<<blocks, 64, 0, stream>>>(out_keys, out_values, in_keys, in_values);
  }
  if (n == 1024) {
    general_topk_pair_sort<T, 256, 4>
        <<<blocks, 256, 0, stream>>>(out_keys, out_values, in_keys, in_values);
  } else if (n == 2048) {
    general_topk_pair_sort<T, 512, 4>
        <<<blocks, 512, 0, stream>>>(out_keys, out_values, in_keys, in_values);
  } else if (n == 4096) {
    general_topk_pair_sort<T, 1024, 4>
        <<<blocks, 1024, 0, stream>>>(out_keys, out_values, in_keys, in_values);
  }
}

template <typename T>
void InitMoeRoutingKernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int num_experts,
    const int num_rows,
    const int active_rows,
    const int cols,
    const int k,
    const int batch_size,
    const int max_seq_len,
    bool ec_route,
    cudaStream_t stream) {
  const int blocks = ec_route ? num_experts * k * batch_size : num_rows * k;
  if (ec_route) {
    constexpr int max_pack_size = 16 / sizeof(T);
    const int threads = std::min(cols / max_pack_size, 1024);
    if (cols % max_pack_size == 0) {
      initialize_moe_routing_kernel<T, max_pack_size>
          <<<blocks, threads, 0, stream>>>(
              unpermuted_input,
              permuted_output,
              expanded_dest_row_to_expanded_source_row,
              expanded_source_row_to_expanded_dest_row,
              num_rows,
              batch_size * k * num_experts,
              cols,
              k,
              max_seq_len,
              ec_route);
    } else {
      initialize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(
          unpermuted_input,
          permuted_output,
          expanded_dest_row_to_expanded_source_row,
          expanded_source_row_to_expanded_dest_row,
          num_rows,
          batch_size * k * num_experts,
          cols,
          k,
          max_seq_len,
          ec_route);
    }
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Currently only support `ec_route = True`. "));
  }
}

template <typename T, typename WeightType, typename arch, typename EpilogueType>
void GenericMoeGemmKernelLauncher(const T* A,
                                  const T* B,
                                  const T* weight_scales,
                                  const T* biases,
                                  T* C,
                                  int64_t* total_rows_before_expert,
                                  int64_t gemm_n,
                                  int64_t gemm_k,
                                  int num_experts,
                                  const int multi_processor_count,
                                  cudaStream_t stream) {
  static_assert(cutlass::platform::is_same<T, half>::value ||
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for half, float");
  static_assert(
      cutlass::platform::is_same<T, WeightType>::value ||
          cutlass::platform::is_same<WeightType, uint8_t>::value ||
          cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
      "cutlass weight type only support float, half, uint8_t, uint4b_t");
  // The cutlass type for the input elements. This is needed to convert to
  // cutlass::half_t if necessary.
  using ElementType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<T, half>::value,
      cutlass::half_t,
      T>::type;
  using ElementType = ElementType_;
  using CutlassWeightType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<WeightType, half>::value,
      cutlass::half_t,
      WeightType>::type;
  using CutlassWeightType = CutlassWeightType_;

  // We need separate config for each architecture since we will target
  // different tensorcore instructions. For float, we do not target TCs.
  using MoeArchTraits = cutlass::gemm::kernel::
      MoeArchTraits<ElementType, CutlassWeightType, arch>;
  using ElementAccumulator = typename MoeArchTraits::AccType;
  using EpilogueOp = typename Epilogue<ElementType,
                                       MoeArchTraits::ElementsPerAccessC,
                                       ElementAccumulator,
                                       EpilogueType>::Op;

  // Finally, set up the kernel.
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementType,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      MoeArchTraits::ElementsPerAccessA,
      CutlassWeightType,
      typename MoeArchTraits::LayoutB,
      cutlass::ComplexTransform::kNone,
      MoeArchTraits::ElementsPerAccessB,
      ElementType,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      typename MoeArchTraits::OperatorClass,
      arch,
      typename MoeArchTraits::ThreadBlockShape,
      typename MoeArchTraits::WarpShape,
      typename MoeArchTraits::InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      MoeArchTraits::Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      typename MoeArchTraits::Operator>::GemmKernel;

  using GemmKernel =
      cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma,
                                       typename GemmKernel_::Epilogue,
                                       typename GemmKernel_::ThreadblockSwizzle,
                                       GemmKernel_::kGroupScheduleMode>;
  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  int occupancy = GemmGrouped::maximum_active_blocks();
  const int threadblock_count = multi_processor_count * occupancy;
  if (occupancy == 0) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "[MoE Runner] GPU lacks the shared memory resources to run GroupedGEMM "
        "kernel"));
  }

  typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
                                          ElementAccumulator(1.f));
  typename GemmGrouped::Arguments args(
      num_experts,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<const ElementType*>(A),
      reinterpret_cast<const CutlassWeightType*>(B),
      reinterpret_cast<const ElementType*>(weight_scales),
      reinterpret_cast<const ElementType*>(biases),
      reinterpret_cast<ElementType*>(C),
      total_rows_before_expert,
      gemm_n,
      gemm_k);
  GemmGrouped gemm;
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "MoEFC kernel will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    PADDLE_THROW(paddle::platform::errors::Fatal("[MoE Runner] " + err_msg));
  }
  auto init_status = gemm.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize cutlass variable batched gemm. Error: " +
        std::string(cutlassGetStatusString(init_status));
    PADDLE_THROW(paddle::platform::errors::Fatal("[MoE Runner] " + err_msg));
  }
  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to run cutlass variable batched gemm. Error: " +
        std::string(cutlassGetStatusString(run_status));
    PADDLE_THROW(paddle::platform::errors::Fatal("[MoE Runner] " + err_msg));
  }
}

template <typename T>
void gemm_bias_act(const T* A,
                   const T* B,
                   const T* weight_scales,
                   const T* biases,
                   T* C,
                   int64_t* total_rows_before_expert,
                   int64_t gemm_n,
                   int64_t gemm_k,
                   int num_experts,
                   int sm,
                   int multi_processor_count,
                   const std::string& act_type,
                   cudaStream_t stream) {
  if (act_type == "gelu") {
    if (sm == 75) {
      GenericMoeGemmKernelLauncher<T,
                                   T,
                                   cutlass::arch::Sm75,
                                   EpilogueOpBiasFtGelu>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          multi_processor_count,
          stream);
    } else if (sm == 80 || sm == 86) {
      GenericMoeGemmKernelLauncher<T,
                                   T,
                                   cutlass::arch::Sm80,
                                   EpilogueOpBiasFtGelu>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          multi_processor_count,
          stream);
    } else {
      GenericMoeGemmKernelLauncher<T,
                                   T,
                                   cutlass::arch::Sm70,
                                   EpilogueOpBiasFtGelu>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          multi_processor_count,
          stream);
    }
  } else {
    // act type is relu.
    if (sm == 75) {
      GenericMoeGemmKernelLauncher<T,
                                   T,
                                   cutlass::arch::Sm75,
                                   EpilogueOpBiasReLU>(A,
                                                       B,
                                                       weight_scales,
                                                       biases,
                                                       C,
                                                       total_rows_before_expert,
                                                       gemm_n,
                                                       gemm_k,
                                                       num_experts,
                                                       multi_processor_count,
                                                       stream);
    } else if (sm == 80 || sm == 86) {
      GenericMoeGemmKernelLauncher<T,
                                   T,
                                   cutlass::arch::Sm80,
                                   EpilogueOpBiasReLU>(A,
                                                       B,
                                                       weight_scales,
                                                       biases,
                                                       C,
                                                       total_rows_before_expert,
                                                       gemm_n,
                                                       gemm_k,
                                                       num_experts,
                                                       multi_processor_count,
                                                       stream);
    } else {
      GenericMoeGemmKernelLauncher<T,
                                   T,
                                   cutlass::arch::Sm70,
                                   EpilogueOpBiasReLU>(A,
                                                       B,
                                                       weight_scales,
                                                       biases,
                                                       C,
                                                       total_rows_before_expert,
                                                       gemm_n,
                                                       gemm_k,
                                                       num_experts,
                                                       multi_processor_count,
                                                       stream);
    }
  }
}

template <typename T>
void gemm(const T* A,
          const T* B,
          const T* weight_scales,
          T* C,
          int64_t* total_rows_before_expert,
          const int gemm_n,
          const int gemm_k,
          const int num_experts,
          int sm,
          int multi_processor_count,
          cudaStream_t stream) {
  if (sm == 75) {
    GenericMoeGemmKernelLauncher<T, T, cutlass::arch::Sm75, EpilogueOpNoBias>(
        A,
        B,
        weight_scales,
        nullptr,
        C,
        total_rows_before_expert,
        gemm_n,
        gemm_k,
        num_experts,
        multi_processor_count,
        stream);
  } else if (sm == 80 || sm == 86) {
    GenericMoeGemmKernelLauncher<T, T, cutlass::arch::Sm80, EpilogueOpNoBias>(
        A,
        B,
        weight_scales,
        nullptr,
        C,
        total_rows_before_expert,
        gemm_n,
        gemm_k,
        num_experts,
        multi_processor_count,
        stream);
  } else {
    GenericMoeGemmKernelLauncher<T, T, cutlass::arch::Sm70, EpilogueOpNoBias>(
        A,
        B,
        weight_scales,
        nullptr,
        C,
        total_rows_before_expert,
        gemm_n,
        gemm_k,
        num_experts,
        multi_processor_count,
        stream);
  }
}

template <typename T>
void finalize_moe_routing_kernelLauncher(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* skip,
    const T* bias,
    const T* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row,
    const int num_experts,
    const int num_rows,
    const int cols,
    const int k,
    bool ec_route,
    cudaStream_t stream) {
  const int blocks = num_rows;
  const int threads = std::min(cols, 1024);
  {
    finalize_moe_routing_kernel<T><<<blocks, threads, 0, stream>>>(
        expanded_permuted_rows,
        reduced_unpermuted_output,
        skip,
        bias,
        scales,
        expanded_source_row_to_expanded_dest_row,
        expert_for_source_row,
        cols,
        num_experts,
        ec_route);
  }
}

template <typename T, typename Context>
void MoeKernel(const Context& ctx,
               const DenseTensor& x,
               const DenseTensor& gate,
               const DenseTensor& bmm0,
               const DenseTensor& bias0,
               const DenseTensor& bmm1,
               const DenseTensor& bias1,
               const std::string& act_type,
               DenseTensor* output) {
  const T* input_activations = x.data<T>();
  T* gating_output = const_cast<T*>(gate.data<T>());
  const T* fc1_expert_weights = bmm0.data<T>();
  const T* fc1_expert_biases = bias0.data<T>();
  const T* fc2_expert_weights = bmm1.data<T>();
  const T* fc2_expert_biases = bias1.data<T>();
  // int moe_act = static_cast<int>(act);
  T* output_ = ctx.template Alloc<T>(output);
  auto stream = ctx.stream();

  auto input_dims = x.dims();
  auto bmm0_dims = bmm0.dims();
  const bool IS_FP16 = std::is_same<T, phi::dtype::float16>::value;

  const int num_rows = input_dims[0] * input_dims[1];
  const int hidden_size = input_dims[2];
  const int inter_size = bmm0_dims[2];
  const int num_experts = bmm0_dims[0];
  const int k = input_dims[1] / 16;
  const int batch_size = input_dims[0];
  const int max_seq_len = 128;
  int64_t bytes = getWorkspaceSize<T>(num_rows,
                                      hidden_size,
                                      inter_size,
                                      num_experts,
                                      k,
                                      batch_size,
                                      max_seq_len);

  // Pointers
  int* source_rows;
  int* padded_source_rows;
  int* permuted_rows;
  int* permuted_experts;
  char* sorter_ws_;
  T* permuted_data;
  T* padded_expert_scales;
  int64_t* total_rows_before_expert;
  T* sorted_softmax_output;
  T* attr_mask;
  T* fc1_result;

  phi::DenseTensor ws_ptr_tensor = phi::Empty<int8_t>(ctx, {bytes});
  int8_t* ws_ptr = ws_ptr_tensor.data<int8_t>();

  const int buf_size = AlignTo16(num_experts * batch_size * k * hidden_size);
  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(num_experts * num_rows);
  // padded_num_moe_inputs for topk sort
  int padded_num_moe_inputs = num_experts * batch_size * max_seq_len;

  source_rows = reinterpret_cast<int*>(ws_ptr);
  padded_source_rows = source_rows + num_moe_inputs;
  permuted_rows = padded_source_rows + padded_num_moe_inputs;
  permuted_experts = permuted_rows + padded_num_moe_inputs;
  permuted_data = reinterpret_cast<T*>(permuted_experts + num_experts * k);
  padded_expert_scales = reinterpret_cast<T*>(permuted_data + buf_size);
  total_rows_before_expert =
      reinterpret_cast<int64_t*>(padded_expert_scales + padded_num_moe_inputs);
  sorted_softmax_output =
      reinterpret_cast<T*>(total_rows_before_expert + padded_experts);
  attr_mask =
      reinterpret_cast<T*>(sorted_softmax_output + padded_num_moe_inputs);
  fc1_result = reinterpret_cast<T*>(attr_mask + num_moe_inputs);

  phi::DenseTensor expert_for_source_row_tensor =
      phi::Empty<int>(ctx, {num_experts, num_rows});
  int* expert_for_source_row = expert_for_source_row_tensor.data<int>();
  phi::DenseTensor expanded_source_row_to_expanded_dest_row_tensor =
      phi::Empty<int>(ctx, {num_experts, num_rows});
  int* expanded_source_row_to_expanded_dest_row =
      expanded_source_row_to_expanded_dest_row_tensor.data<int>();
  phi::DenseTensor expert_scales_tensor =
      phi::Empty<T>(ctx, {num_experts, num_rows});
  T* expert_scales = expert_scales_tensor.data<T>();
  phi::DenseTensor fc2_output_tensor =
      phi::Empty<T>(ctx, {num_experts * batch_size * k, hidden_size});
  T* fc2_result = fc2_output_tensor.data<T>();
  phi::DenseTensor input_lengths_tensor = phi::Empty<int>(ctx, {batch_size});
  int* input_lengths = input_lengths_tensor.data<int>();
  funcs::SetConstant<Context, int> set_len;
  set_len(ctx, &input_lengths_tensor, static_cast<int>(max_seq_len));

  int sm = getSMVersion();
  int multi_processor_count = phi::backends::gpu::GetGPUMultiProcessors(
      phi::backends::gpu::GetCurrentDeviceId());

  InitExpertChoiceRouteKernelLauncher<T>(
      expert_for_source_row,
      source_rows,
      expanded_source_row_to_expanded_dest_row,
      total_rows_before_expert,
      attr_mask,
      num_experts,
      num_rows,
      k,
      batch_size,
      ctx.stream());
  if (IS_FP16) {
    invokeMaskedSoftMax<__half>(reinterpret_cast<__half*>(gating_output),
                                reinterpret_cast<const __half*>(gating_output),
                                reinterpret_cast<const __half*>(attr_mask),
                                /*batch_size=*/num_rows,
                                /*seq_len=*/num_experts,
                                ctx.stream());
  } else {
    invokeMaskedSoftMax<float>(reinterpret_cast<float*>(gating_output),
                               reinterpret_cast<const float*>(gating_output),
                               reinterpret_cast<const float*>(attr_mask),
                               /*batch_size=*/num_rows,
                               /*seq_len=*/num_experts,
                               ctx.stream());
  }
  InvokeTransposeAxis01(
      expert_scales, gating_output, num_rows, num_experts, 1, ctx.stream());

  int padded_max_seq_len = max_seq_len <= 128 ? 128 : 256;
  InvokePadding(padded_expert_scales,
                padded_source_rows,
                expert_scales,
                source_rows,
                input_lengths,
                num_rows,
                batch_size,
                padded_max_seq_len,
                num_experts,
                ctx.stream());
  if (IS_FP16) {
    InvokeGeneralTopKPairSort<__half>(
        reinterpret_cast<__half*>(sorted_softmax_output),
        permuted_rows,
        reinterpret_cast<__half*>(padded_expert_scales),
        padded_source_rows,
        num_experts * batch_size,
        padded_max_seq_len,
        ctx.stream());
  } else {
    InvokeGeneralTopKPairSort<float>(
        reinterpret_cast<float*>(sorted_softmax_output),
        permuted_rows,
        reinterpret_cast<float*>(padded_expert_scales),
        padded_source_rows,
        num_experts * batch_size,
        padded_max_seq_len,
        ctx.stream());
  }
  InitMoeRoutingKernelLauncher(input_activations,
                               permuted_data,
                               permuted_rows,
                               expanded_source_row_to_expanded_dest_row,
                               num_experts,
                               num_rows,
                               num_rows,
                               hidden_size,
                               k,
                               batch_size,
                               max_seq_len,
                               true,
                               ctx.stream());

  const T* fc1_scales = nullptr;
  const T* fc2_scales = nullptr;
  if (IS_FP16) {
    gemm_bias_act(reinterpret_cast<const __half*>(permuted_data),
                  reinterpret_cast<const __half*>(fc1_expert_weights),
                  reinterpret_cast<const __half*>(fc1_scales),
                  reinterpret_cast<const __half*>(fc1_expert_biases),
                  reinterpret_cast<__half*>(fc1_result),
                  total_rows_before_expert,
                  inter_size,
                  hidden_size,
                  num_experts,
                  sm,
                  multi_processor_count,
                  act_type,
                  ctx.stream());
    gemm(reinterpret_cast<const __half*>(fc1_result),
         reinterpret_cast<const __half*>(fc2_expert_weights),
         reinterpret_cast<const __half*>(fc2_scales),
         reinterpret_cast<__half*>(fc2_result),
         total_rows_before_expert,
         hidden_size,
         inter_size,
         num_experts,
         sm,
         multi_processor_count,
         ctx.stream());
  } else {
    gemm_bias_act<float>(reinterpret_cast<const float*>(permuted_data),
                         reinterpret_cast<const float*>(fc1_expert_weights),
                         reinterpret_cast<const float*>(fc1_scales),
                         reinterpret_cast<const float*>(fc1_expert_biases),
                         reinterpret_cast<float*>(fc1_result),
                         total_rows_before_expert,
                         inter_size,
                         hidden_size,
                         num_experts,
                         sm,
                         multi_processor_count,
                         act_type,
                         ctx.stream());
    gemm<float>(reinterpret_cast<const float*>(fc1_result),
                reinterpret_cast<const float*>(fc2_expert_weights),
                reinterpret_cast<const float*>(fc2_scales),
                reinterpret_cast<float*>(fc2_result),
                total_rows_before_expert,
                hidden_size,
                inter_size,
                num_experts,
                sm,
                multi_processor_count,
                ctx.stream());
  }

  finalize_moe_routing_kernelLauncher(fc2_result,
                                      output_,
                                      input_activations,
                                      fc2_expert_biases,
                                      expert_scales,
                                      expanded_source_row_to_expanded_dest_row,
                                      expert_for_source_row,
                                      num_experts,
                                      num_rows,
                                      hidden_size,
                                      k,
                                      true,
                                      ctx.stream());
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    moe, GPU, ALL_LAYOUT, phi::fusion::MoeKernel, float, phi::dtype::float16) {}
