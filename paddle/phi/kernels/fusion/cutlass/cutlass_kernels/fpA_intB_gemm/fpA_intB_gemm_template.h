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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma once

#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/compute_occupancy.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/epilogue_helpers.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/ft_gemm_configs.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/fpA_intB_gemm_split_k.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/default_mma.h"
#pragma GCC diagnostic pop

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/autogen/arch_define.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"
namespace phi {

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          bool FineGrained,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_mixed_gemm_kernelLauncher(const T* A,
                                       const WeightType* B,
                                       const T* weight_scales,
                                       const T* biases,
                                       T* C,
                                       int m,
                                       int n,
                                       int k,
                                       int group_size,
                                       CutlassGemmConfig gemm_config,
                                       char* workspace,
                                       size_t workspace_bytes,
                                       cudaStream_t stream,
                                       int* occupancy) {
  static_assert(cutlass::platform::is_same<T, half>::value ||
#ifdef PADDLE_CUDA_BF16
                    cutlass::platform::is_same<T, __nv_bfloat16>::value ||
#endif
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for bfloat16, half, float");

  static_assert(
      cutlass::platform::is_same<T, WeightType>::value ||
          cutlass::platform::is_same<WeightType, uint8_t>::value ||
          cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
      "");

  // The cutlass type for the input elements. This is needed to convert to
  // cutlass::half_t if necessary.
  using ElementType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<T, half>::value,
      cutlass::half_t,
      T>::type;
#ifdef PADDLE_CUDA_BF16
  using ElementType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
      cutlass::bfloat16_t,
      ElementType_>::type;
#endif
  using CutlassWeightType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<WeightType, half>::value,
      cutlass::half_t,
      WeightType>::type;

#ifdef PADDLE_CUDA_BF16
  using CutlassWeightType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
      cutlass::bfloat16_t,
      CutlassWeightType_>::type;
#endif

  // We need separate config for each architecture since we will target
  // different tensorcore instructions. For float, we do not target TCs.
  using MixedGemmArchTraits = cutlass::gemm::kernel::
      MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;

  using EpilogueOp = typename Epilogue<ElementType,
                                       MixedGemmArchTraits::ElementsPerAccessC,
                                       ElementAccumulator,
                                       EpilogueTag>::Op;

  if (gemm_config.split_k_style == SplitKStyle::NO_SPLIT_K ||
      FineGrained == true) {
    using Operator = typename MixedGemmArchTraits::Operator;
    using TaggedOperator =
        typename cutlass::arch::TagOperator<Operator,
                                            FineGrained>::TaggedOperator;
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
        ElementType,
        cutlass::layout::RowMajor,
        MixedGemmArchTraits::ElementsPerAccessA,
        CutlassWeightType,
        typename MixedGemmArchTraits::LayoutB,
        MixedGemmArchTraits::ElementsPerAccessB,
        ElementType,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        arch,
        ThreadblockShape,
        WarpShape,
        typename MixedGemmArchTraits::InstructionShape,
        EpilogueOp,
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        true,
        TaggedOperator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<
        typename GemmKernel_::Mma,
        typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch,  // Ensure top level arch is used for dispatch
        GemmKernel_::kSplitKSerial,
        FineGrained>;

    if (occupancy != nullptr) {
      *occupancy = compute_occupancy_for_kernel<GemmKernel>();
      return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

    const int ldb =
        cutlass::platform::is_same<cutlass::layout::RowMajor,
                                   typename MixedGemmArchTraits::LayoutB>::value
            ? n
            : k * GemmKernel::kInterleave;

    typename Gemm::Arguments args(
        {m, n, k},
        group_size,
        {reinterpret_cast<ElementType*>(const_cast<T*>(A)), k},
        {reinterpret_cast<CutlassWeightType*>(const_cast<WeightType*>(B)), ldb},
        {reinterpret_cast<ElementType*>(const_cast<T*>(weight_scales)), n},
        {reinterpret_cast<ElementType*>(const_cast<T*>(biases)), 0},
        {reinterpret_cast<ElementType*>(C), n},
        gemm_config.split_k_factor,
        {ElementAccumulator(1.f), ElementAccumulator(0.f)});

    // This assertion is enabled because because for the column interleaved
    // layout, K MUST be a multiple of threadblockK. The reason for this is that
    // the default pitchlinear iterators are used to handle walking over the
    // interleaved matrix. The way masking in handled in these do not map to the
    // interleaved layout. We need to write our own predicated iterator in order
    // to relax this limitation.
    if (GemmKernel::kInterleave > 1 &&
        ((k % MixedGemmArchTraits::ThreadblockK) ||
         ((k / gemm_config.split_k_factor) %
          MixedGemmArchTraits::ThreadblockK))) {
      throw std::runtime_error(
          "Temp assertion: k must be multiple of threadblockK");
    }

    Gemm gemm;
    if (gemm.get_workspace_size(args) > workspace_bytes) {
      // TODO(wangbojun) here to reset the split-k in gemm args, but no work for
      // now to run bf16 mixgemm, we have set the split-k factor to 1
      VLOG(1) << "Requested split-k but workspace size insufficient. Falling "
                 "back to non-split-k implementation.";
      VLOG(1) << "need workspace size of: " << gemm.get_workspace_size(args)
              << ", but got " << workspace_bytes;
      VLOG(1) << "args.batch_stride_D:" << args.batch_stride_D;
      VLOG(1) << "args.batch_count:" << args.batch_count;
      // If requested split-k factor will require more workspace bytes, revert
      // to standard gemm.
      //
      args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      std::string err_msg =
          "fpA_intB cutlass kernel will fail for params. Error: " +
          std::string(cutlassGetStatusString(can_implement));
      throw std::runtime_error("[fpA_intB Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to initialize cutlass fpA_intB gemm. Error: " +
          std::string(cutlassGetStatusString(init_status));
      throw std::runtime_error("[fpA_intB Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
      std::string err_msg = "Failed to run cutlass fpA_intB gemm. Error: " +
                            std::string(cutlassGetStatusString(run_status));
      throw std::runtime_error("[fpA_intB Runner] " + err_msg);
    }

  } else /* Per-Channel mode */ {
    // for stream-k, we set gemm_config.split_k_factor = 1 to use default load
    // balance.
    gemm_config.split_k_factor = 1;
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmUniversal<
        ElementType,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessA,
        CutlassWeightType,
        typename MixedGemmArchTraits::LayoutB,
        cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessB,
        ElementType,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        arch,
        ThreadblockShape,
        WarpShape,
        typename MixedGemmArchTraits::InstructionShape,
        EpilogueOp,
        typename cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
        Stages,
        typename MixedGemmArchTraits::Operator,
        cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;
    using GemmKernel = cutlass::gemm::kernel::GemmFpAIntBSplitK<
        typename GemmKernel_::Mma,
        typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch  // Ensure top level arch is used for dispatch
        >;

    if (occupancy != nullptr) {
      *occupancy = compute_occupancy_for_kernel2<GemmKernel>();
      return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

    const int ldb =
        cutlass::platform::is_same<cutlass::layout::RowMajor,
                                   typename MixedGemmArchTraits::LayoutB>::value
            ? n
            : k * GemmKernel::kInterleave;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        {reinterpret_cast<ElementType*>(const_cast<T*>(A)), k},
        {reinterpret_cast<CutlassWeightType*>(const_cast<WeightType*>(B)), ldb},
        {reinterpret_cast<ElementType*>(const_cast<T*>(weight_scales)), 0},
        {reinterpret_cast<ElementType*>(const_cast<T*>(biases)), 0},
        {reinterpret_cast<ElementType*>(C), n},
        gemm_config.split_k_factor,
        {ElementAccumulator(1.f), ElementAccumulator(0.f)});

    // This assertion is enabled because because for the column interleaved
    // layout, K MUST be a multiple of threadblockK. The reason for this is that
    // the default pitchlinear iterators are used to handle walking over the
    // interleaved matrix. The way masking in handled in these do not map to the
    // interleaved layout. We need to write our own predicated iterator in order
    // to relax this limitation.
    if (GemmKernel::kInterleave > 1 &&
        ((k % MixedGemmArchTraits::ThreadblockK) ||
         ((k / gemm_config.split_k_factor) %
          MixedGemmArchTraits::ThreadblockK))) {
      throw std::runtime_error(
          "Temp assertion: k must be multiple of threadblockK");
    }

    Gemm gemm;
    if (gemm.get_workspace_size(args) > workspace_bytes) {
      VLOG(1) << "Requested split-k but workspace size insufficient. Falling "
                 "back to non-split-k implementation.";
      VLOG(1) << "Requested workspace_size: " << gemm.get_workspace_size(args);
      VLOG(1) << "get workspace_size: " << workspace_bytes;
      // If requested split-k factor will require more workspace bytes, revert
      // to standard gemm.
      args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      std::string err_msg =
          "fpA_intB cutlass kernel will fail for params. Error: " +
          std::string(cutlassGetStatusString(can_implement));
      throw std::runtime_error("[fpA_intB_gemm Error][fpA_intB Runner] " +
                               err_msg);
    }

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to initialize cutlass fpA_intB gemm. Error: " +
          std::string(cutlassGetStatusString(init_status));
      throw std::runtime_error("[fpA_intB_gemm Error][fpA_intB Runner] " +
                               err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
      std::string err_msg = "Failed to run cutlass fpA_intB gemm. Error: " +
                            std::string(cutlassGetStatusString(run_status));
      throw std::runtime_error("[fpA_intB_gemm Error][fpA_intB Runner] " +
                               err_msg);
    }
  }
}

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          bool FineGrained,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_mixed_gemm_kernelLauncher_template(const T* A,
                                                const WeightType* B,
                                                const T* weight_scales,
                                                const T* biases,
                                                T* C,
                                                int m,
                                                int n,
                                                int k,
                                                int group_size,
                                                CutlassGemmConfig gemm_config,
                                                char* workspace,
                                                size_t workspace_bytes,
                                                cudaStream_t stream,
                                                int* occupancy);

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          bool FineGrained,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages,
          typename Enable = void>
struct dispatch_stages {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       int group_size,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    // VLOG(3)<<__PRETTY_FUNCTION__;
    std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) +
                          " with stages set to " + std::to_string(Stages);
    throw std::runtime_error("[dispatch_stages::dispatch] " + err_msg);
  }
};
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          bool FineGrained,
          typename ThreadblockShape,
          typename WarpShape>
struct dispatch_stages<T,
                       WeightType,
                       arch,
                       EpilogueTag,
                       FineGrained,
                       ThreadblockShape,
                       WarpShape,
                       2> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       int group_size,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    // VLOG(3)<<__PRETTY_FUNCTION__;

    generic_mixed_gemm_kernelLauncher_template<T,
                                               WeightType,
                                               arch,
                                               EpilogueTag,
                                               FineGrained,
                                               ThreadblockShape,
                                               WarpShape,
                                               2>(A,
                                                  B,
                                                  weight_scales,
                                                  biases,
                                                  C,
                                                  m,
                                                  n,
                                                  k,
                                                  group_size,
                                                  gemm_config,
                                                  workspace,
                                                  workspace_bytes,
                                                  stream,
                                                  occupancy);
  }
};

#if defined(USE_FPAINTB_GEMM_WITH_SM80)
template <typename T,
          typename WeightType,
          typename EpilogueTag,
          bool FineGrained,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
struct dispatch_stages<T,
                       WeightType,
                       cutlass::arch::Sm80,
                       EpilogueTag,
                       FineGrained,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       int group_size,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_mixed_gemm_kernelLauncher_template<T,
                                               WeightType,
                                               cutlass::arch::Sm80,
                                               EpilogueTag,
                                               FineGrained,
                                               ThreadblockShape,
                                               WarpShape,
                                               Stages>(A,
                                                       B,
                                                       weight_scales,
                                                       biases,
                                                       C,
                                                       m,
                                                       n,
                                                       k,
                                                       group_size,
                                                       gemm_config,
                                                       workspace,
                                                       workspace_bytes,
                                                       stream,
                                                       occupancy);
  }
};
#endif

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          bool FineGrained,
          typename ThreadblockShape,
          typename WarpShape>
void dispatch_gemm_config(const T* A,
                          const WeightType* B,
                          const T* weight_scales,
                          const T* biases,
                          T* C,
                          int m,
                          int n,
                          int k,
                          int group_size,
                          CutlassGemmConfig gemm_config,
                          char* workspace,
                          size_t workspace_bytes,
                          cudaStream_t stream,
                          int* occupancy);

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          bool FineGrained>
void dispatch_gemm_to_cutlass(const T* A,
                              const WeightType* B,
                              const T* weight_scales,
                              const T* biases,
                              T* C,
                              int m,
                              int n,
                              int k,
                              int group_size,
                              char* workspace,
                              size_t workspace_bytes,
                              CutlassGemmConfig gemm_config,
                              cudaStream_t stream,
                              int* occupancy);

}  // namespace phi
