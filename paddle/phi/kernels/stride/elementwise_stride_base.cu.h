// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

namespace phi {

// Not Support Vectorized Kernel For Now
#define STRIDE_VEC_SIZE 1

template <typename Functor,
          typename OutT,
          int Arity,
          int NumOuts,
          int VecSize,
          int vt>
__global__ void BinaryElementwiseKernel(
    Array<const _ptr_ char *__restrict__, Arity> ins,
    Array<_ptr_ OutT *, NumOuts> outs,
    uint32_t numel,
    int read_lens,
    Functor func,
    funcs::OffsetCalculator<Arity + NumOuts> offset_calc) {
  int64_t tid = THREAD_ID_X;
  int64_t nv = BLOCK_NUM_X * vt;
  int64_t idx = nv * BLOCK_ID_X + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < numel) {
      auto offsets = offset_calc.get(idx);
      using Traits = phi::funcs::FunctionTraits<Functor>;
      using ArgsT = typename Traits::ArgsTuple;
      __simd__ ArgsT args[VecSize];
      __simd__ ConditionalT<OutT, NumOuts> result[VecSize];
      std::get<0>(args[idx]) =
          *(reinterpret_cast<const _ptr_ std::tuple_element_t<0, ArgsT> *>(
              reinterpret_cast<const _ptr_ char *>(ins[0]) + offsets[1]));
      std::get<1>(args[idx]) =
          *(reinterpret_cast<const _ptr_ std::tuple_element_t<1, ArgsT> *>(
              reinterpret_cast<const _ptr_ char *>(ins[1]) + offsets[2]));
      funcs::SameDimsElementwisePrimitiveCaller<ConditionalT<OutT, NumOuts>,
                                                VecSize,
                                                Functor,
                                                ArgsT,
                                                Arity>()(
          func, args, result, read_lens);
      char *out_ptr = reinterpret_cast<char *>(outs[0]) + offsets[0];
      *reinterpret_cast<OutT *>(out_ptr) =
          *reinterpret_cast<const OutT *>(&(result[0]));
      idx += BLOCK_NUM_X;
    }
  }
}

template <typename Functor,
          typename OutT,
          int Arity,
          int NumOuts,
          int VecSize,
          int vt>
__global__ void UnaryElementwiseKernel(
    Array<const _ptr_ char *__restrict__, Arity> ins,
    Array<_ptr_ OutT *, NumOuts> outs,
    uint32_t numel,
    int read_lens,
    Functor func,
    funcs::OffsetCalculator<Arity + NumOuts> offset_calc) {
  int64_t tid = THREAD_ID_X;
  int64_t nv = BLOCK_NUM_X * vt;
  int64_t idx = nv * BLOCK_ID_X + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < numel) {
      auto offsets = offset_calc.get(idx);
      using Traits = phi::funcs::FunctionTraits<Functor>;
      using ArgsT = typename Traits::ArgsTuple;
      __simd__ ArgsT args[VecSize];
      __simd__ ConditionalT<OutT, NumOuts> result[VecSize];
      std::get<0>(args[idx]) =
          *(reinterpret_cast<const _ptr_ std::tuple_element_t<0, ArgsT> *>(
              reinterpret_cast<const _ptr_ char *>(ins[0]) + offsets[1]));
      funcs::SameDimsElementwisePrimitiveCaller<ConditionalT<OutT, NumOuts>,
                                                VecSize,
                                                Functor,
                                                ArgsT,
                                                Arity>()(
          func, args, result, read_lens);
      char *out_ptr = reinterpret_cast<char *>(outs[0]) + offsets[0];
      *reinterpret_cast<OutT *>(out_ptr) =
          *reinterpret_cast<const OutT *>(&(result[0]));
      idx += BLOCK_NUM_X;
    }
  }
}

template <typename OutT, typename Context, typename Functor, int NumOuts = 1>
void BinaryStrideBroadcastKernel(const Context &dev_ctx,
                                 const std::vector<const DenseTensor *> &ins,
                                 std::vector<DenseTensor *> *outs,
                                 Functor func,
                                 int axis = -1) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int Arity = Traits::arity;
  for (auto i = 0; i < outs->size(); ++i) {
    if (i > 0) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          common::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, but "
              "%d-th output tensor`s shape is not.",
              i));
    }
    dev_ctx.template Alloc<OutT>((*outs)[i]);
  }
  if ((*outs)[0]->numel() == 0) {
    return;
  }
  int max_rank = 0;
  int min_rank = phi::DDim::kMaxRank;
  for (auto *in : ins) {
    max_rank = std::max(max_rank, in->dims().size());
    min_rank = std::min(min_rank, in->dims().size());
  }
  if (ins.size() == 1) {
    max_rank = std::max(max_rank, (*outs)[0]->dims().size());
  }
  axis = axis == -1 ? max_rank - min_rank : axis;
  auto classifier =
      funcs::BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(
          ins, outs, axis);
  DenseTensorIteratorConfig config;
  config.add_output(*((*outs)[0]));
  config.add_const_input(*(ins[0]));
  config.add_const_input(*(ins[1]));
  DenseTensorIterator iter = config.build();
  const int &numel = iter.numel();
  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<3>(iter);
  constexpr int unroll_factor = sizeof(OutT) >= 4 ? 2 : 4;
  auto stream = dev_ctx.stream();
  auto threads = 128;
  auto blocks = (numel + 128 * unroll_factor - 1) / (128 * unroll_factor);
  int vec_size = STRIDE_VEC_SIZE;
  BinaryElementwiseKernel<Functor,
                          OutT,
                          Arity,
                          NumOuts,
                          STRIDE_VEC_SIZE,
                          unroll_factor>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       numel,
                                       vec_size,
                                       func,
                                       offset_calc);
}

template <typename OutT, typename Context, typename Functor, int NumOuts = 1>
void BinaryStrideElementwiseKernel(const Context &dev_ctx,
                                   const std::vector<const DenseTensor *> &ins,
                                   std::vector<DenseTensor *> *outs,
                                   Functor func) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int Arity = Traits::arity;
  bool have_0_size = false;
  for (int i = 0; i < outs->size(); ++i) {
    if (outs->at(i)->numel() == 0) {
      have_0_size = true;
    }
    if (i > 0) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          common::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
    dev_ctx.template Alloc<OutT>((*outs)[i]);
  }
  if (have_0_size) {
    return;
  }
  int max_rank = 0;
  int min_rank = phi::DDim::kMaxRank;
  for (auto *in : ins) {
    max_rank = std::max(max_rank, in->dims().size());
    min_rank = std::min(min_rank, in->dims().size());
  }
  if (ins.size() == 1) {
    max_rank = std::max(max_rank, (*outs)[0]->dims().size());
  }
  int axis = max_rank - min_rank;
  auto classifier =
      funcs::BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(
          ins, outs, axis);
  DenseTensorIteratorConfig config;
  config.add_output(*((*outs)[0]));
  config.add_const_input(*(ins[0]));
  config.add_const_input(*(ins[1]));
  DenseTensorIterator iter = config.build();
  const int &numel = iter.numel();
  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<3>(iter);
  constexpr int unroll_factor = sizeof(OutT) >= 4 ? 2 : 4;
  auto stream = dev_ctx.stream();
  auto threads = 128;
  auto blocks = (numel + 128 * unroll_factor - 1) / (128 * unroll_factor);
  int vec_size = STRIDE_VEC_SIZE;
  BinaryElementwiseKernel<Functor,
                          OutT,
                          Arity,
                          NumOuts,
                          STRIDE_VEC_SIZE,
                          unroll_factor>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       numel,
                                       vec_size,
                                       func,
                                       offset_calc);
}

template <typename OutT, typename Context, typename Functor, int NumOuts = 1>
void UnaryStrideElementwiseKernel(const Context &dev_ctx,
                                  const std::vector<const DenseTensor *> &ins,
                                  std::vector<DenseTensor *> *outs,
                                  Functor func) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int Arity = Traits::arity;
  bool have_0_size = false;
  for (int i = 0; i < outs->size(); ++i) {
    if (outs->at(i)->numel() == 0) {
      have_0_size = true;
    }
    if (i > 0) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          common::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
    dev_ctx.template Alloc<OutT>((*outs)[i]);
  }
  if (have_0_size) {
    return;
  }
  int max_rank = 0;
  int min_rank = phi::DDim::kMaxRank;
  for (auto *in : ins) {
    max_rank = std::max(max_rank, in->dims().size());
    min_rank = std::min(min_rank, in->dims().size());
  }
  if (ins.size() == 1) {
    max_rank = std::max(max_rank, (*outs)[0]->dims().size());
  }
  int axis = max_rank - min_rank;
  auto classifier =
      funcs::BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(
          ins, outs, axis);
  DenseTensorIteratorConfig config;
  config.add_output(*((*outs)[0]));
  config.add_const_input(*(ins[0]));
  DenseTensorIterator iter = config.build();
  const int &numel = iter.numel();
  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<2>(iter);
  constexpr int unroll_factor = sizeof(OutT) >= 4 ? 2 : 4;
  auto stream = dev_ctx.stream();
  auto threads = 128;
  auto blocks = (numel + 128 * unroll_factor - 1) / (128 * unroll_factor);
  int vec_size = STRIDE_VEC_SIZE;
  UnaryElementwiseKernel<Functor,
                         OutT,
                         Arity,
                         NumOuts,
                         STRIDE_VEC_SIZE,
                         unroll_factor>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       numel,
                                       vec_size,
                                       func,
                                       offset_calc);
}

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context &dev_ctx,
                                   const phi::DenseTensor &tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

#undef STRIDE_VEC_SIZE

}  // namespace phi

#endif
