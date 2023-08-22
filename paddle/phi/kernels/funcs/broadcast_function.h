/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <sstream>
#include "paddle/phi/kernels/funcs/elementwise_base.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

namespace kps = phi::kps;

#endif

namespace phi {
namespace funcs {

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)

enum BroadcastLoadType { kMixed = 1, kBroadcast = 2, kElementwise = 3 };

template <int Index>
struct UseBroadcast {
  template <typename ArgsT, typename Array1, typename Array2>
  static HOSTDEVICE void Apply(
      const std::vector<const DenseTensor *> &ins_tensor,
      const ArgsT &args,
      int64_t numel,
      Array1 *ins_data,
      Array2 *use_broadcast,
      int *broadcast_num,
      bool *all_elementwise) {
    (*ins_data)[Index] = (const _ptr_ char *)(ins_tensor[Index]->data());
    bool is_same_dim = ins_tensor[Index]->numel() == numel;
    if (is_same_dim) {
      (*use_broadcast)[Index] = false;
    } else {
      (*use_broadcast)[Index] = true;
      (*broadcast_num)++;
    }
    *all_elementwise &= is_same_dim;
  }
};

template <typename OutT, int Arity, typename Functor>
struct LoaderTypeClassifier {
 public:
  int64_t numel{0};
  int vec_size{4};
  int broadcast_num{0};
  bool all_elementwise{true};
  phi::Array<bool, Arity> use_broadcast;
  phi::Array<const _ptr_ char *__restrict__, Arity> ins_data;

  LoaderTypeClassifier() {}
  LoaderTypeClassifier(const std::vector<const DenseTensor *> &ins,
                       std::vector<DenseTensor *> *outs) {
    using Traits = phi::funcs::FunctionTraits<Functor>;
    using ArgsT = typename Traits::ArgsTuple;
    ArgsT arg;
    uint64_t out_addr = reinterpret_cast<uint64_t>((*outs)[0]->data<OutT>());

    UnrollerWithoutVecSize<VecSizeGetter, Arity>::step(ins, arg, &vec_size);

    for (auto i = 1; i < outs->size(); ++i) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          phi::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, but "
              "%d-th output tensor`s shape is not.",
              i));
      out_addr =
          (out_addr | reinterpret_cast<uint64_t>((*outs)[i]->data<OutT>()));
    }

    vec_size = std::min(
        vec_size,
        phi::GetVectorizedSize<OutT>(reinterpret_cast<OutT *>(out_addr)));
    numel = (*outs)[0]->numel();
    UnrollerWithoutVecSize<UseBroadcast, Arity>::step(ins,
                                                      arg,
                                                      numel,
                                                      &ins_data,
                                                      &use_broadcast,
                                                      &broadcast_num,
                                                      &all_elementwise);
  }
};

// Common broadcast/elementwise Loader.
template <int Index, int VecSize, bool IsBoundary, int LoadType>
struct BroadcastDataLoader {
  template <typename Array1, typename Array2, typename Array3, typename ArgsT>
  static __device__ __forceinline__ void Apply(const Array1 &ins,
                                               ArgsT *args,
                                               const Array2 &configs,
                                               const Array3 &use_broadcast,
                                               const int block_offset,
                                               const int num,
                                               const uint32_t numel,
                                               int read_lens) {
    using Type = std::tuple_element_t<Index, ArgsT>;
#ifdef PADDLE_WITH_XPU_KP
    kps::Init<Type, ArgsT, Index, VecSize>(
        args, static_cast<Type>(1.0f), read_lens);
    if (use_broadcast[Index]) {
      kps::ReadDataBc<Type, VecSize, 1, ArgsT, Index, IsBoundary>(
          args,
          reinterpret_cast<const _ptr_ Type *>(ins[Index]),
          block_offset,
          configs[Index],
          numel,
          read_lens);
    } else {
      kps::ReadData<Type, VecSize, 1, ArgsT, Index, IsBoundary>(
          args,
          reinterpret_cast<const _ptr_ Type *>(ins[Index]) + block_offset,
          num,
          read_lens);
    }
#else
    kps::Init<Type, ArgsT, Index, VecSize>(args, static_cast<Type>(1.0f));
    if (use_broadcast[Index]) {
      kps::ReadDataBc<Type, VecSize, 1, ArgsT, Index, IsBoundary>(
          args,
          reinterpret_cast<const _ptr_ Type *>(ins[Index]),
          block_offset,
          configs[Index],
          numel,
          VecSize);
    }
    // NOTE: If use if...else... with condition `use_broadcast[Index]` here,
    // there will be some errs with clang12 while compiling in ROCm.
    // When the compiler is upgraded, if...else... may be used.
    if (!use_broadcast[Index]) {
      kps::ReadData<Type, VecSize, 1, ArgsT, Index, IsBoundary>(
          args,
          reinterpret_cast<const _ptr_ Type *>(ins[Index]) + block_offset,
          num,
          VecSize);
    }
#endif
  }
};

/* BroadcastDataLoaders Partial specialization */
#ifndef PADDLE_WITH_XPU_KP
// Scalar elementwise Loader with consideration of IsBoundary.
template <int Index, int VecSize>
struct BroadcastDataLoader<Index, VecSize, true, kElementwise> {
  template <typename Array1, typename Array2, typename Array3, typename ArgsT>
  static __device__ __forceinline__ void Apply(const Array1 &ins,
                                               ArgsT *args,
                                               const Array2 &configs,
                                               const Array3 &use_broadcast,
                                               const int block_offset,
                                               const int num,
                                               const uint32_t numel,
                                               int read_lens) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    int thread_offset = threadIdx.x * VecSize + block_offset;
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      std::get<Index>(args[idx]) = static_cast<Type>(1);
      int index = thread_offset + idx;
      if (index < numel) {
        std::get<Index>(args[idx]) =
            reinterpret_cast<const _ptr_ Type *>(ins[Index])[index];
      }
    }
  }
};

// Vectorized elementwise Loader without consideration of IsBoundary.
template <int Index, int VecSize>
struct BroadcastDataLoader<Index, VecSize, false, kElementwise> {
  template <typename Array1, typename Array2, typename Array3, typename ArgsT>
  static __device__ __forceinline__ void Apply(const Array1 &ins,
                                               ArgsT *args,
                                               const Array2 &configs,
                                               const Array3 &use_broadcast,
                                               const int block_offset,
                                               const int num,
                                               const uint32_t numel,
                                               int read_lens) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    using VecType = phi::kps::details::VectorType<Type, VecSize>;
    VecType vec_temp;

    int thread_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const VecType *__restrict__ vec_input =
        reinterpret_cast<const VecType *__restrict__>(ins[Index]);
    vec_temp = vec_input[thread_offset];
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      std::get<Index>(args[idx]) = vec_temp.val[idx];
    }
  }
};

template <int Index, int VecSize>
struct BroadcastDataInit {
  template <typename ArgsT>
  static __device__ __forceinline__ void Apply(ArgsT *args) {
    using Type = std::tuple_element_t<Index, ArgsT>;
#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
      std::get<Index>(args[k]) = static_cast<Type>(1);
    }
  }
};

template <int Index, int VecSize>
struct BroadcastDataSetter {
  template <typename Array, typename ArgsT>
  static __device__ __forceinline__ void Apply(const Array &ins,
                                               ArgsT *args,
                                               uint32_t index_bc[][VecSize]) {
    using Type = std::tuple_element_t<Index, ArgsT>;
#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
      std::get<Index>(args[k]) =
          reinterpret_cast<const _ptr_ Type *>(ins[Index])[index_bc[Index][k]];
    }
  }
};

#endif

// static broadcast unroller
template <template <int Index, int VecSize, bool IsBoundary, int LoadType>
          typename Func,
          bool IsBoundary,
          int LoadType,
          int VecSize,
          int End,
          int Begin = 0>
struct BcUnroller {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&...args) {
    Func<Begin, VecSize, IsBoundary, LoadType>::Apply(
        std::forward<Args>(args)...);
    BcUnroller<Func, IsBoundary, LoadType, VecSize, End, Begin + 1>::step(
        args...);
  }
};

template <template <int Index, int VecSize, bool IsBoundary, int LoadType>
          typename Func,
          bool IsBoundary,
          int LoadType,
          int VecSize,
          int End>
struct BcUnroller<Func, IsBoundary, LoadType, VecSize, End, End> {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&...args) {}
};

template <typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary,
          int LoadType>
__device__ void VectorizedBroadcastKernelImpl(
    const phi::Array<const _ptr_ char *__restrict__, Arity> &ins,
    phi::Array<_ptr_ OutT *, NumOuts> outs,
    const phi::Array<bool, Arity> &use_broadcast,
    const uint32_t numel,
    const phi::Array<kps::details::BroadcastConfig, Arity> &configs,
    int num,
    int block_offset,
    int read_lens,
    Functor func) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;
  __simd__ ArgsT args[VecSize];
  __simd__ ConditionalT<OutT, NumOuts> result[VecSize];

#ifdef PADDLE_WITH_XPU_KP
  BcUnroller<BroadcastDataLoader, IsBoundary, LoadType, VecSize, Arity>::step(
      ins, args, configs, use_broadcast, block_offset, num, numel, read_lens);
#else
  if (LoadType == kBroadcast) {
    uint32_t index_bc[Arity][VecSize] = {0};
    Unroller<BroadcastDataInit, VecSize, Arity>::step(args);
    uint32_t thread_offset = block_offset + threadIdx.x * VecSize;
#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
      uint32_t idx = thread_offset + k;
      if (IsBoundary && idx == numel) break;
#pragma unroll
      for (int i = 0; i < phi::DDim::kMaxRank; ++i) {
        if (i == configs[0].rank) break;
        auto fast_divmoder = configs[0].divmoders[i].Divmod(idx);
        idx = fast_divmoder.val[0];
#pragma unroll
        for (int j = 0; j < Arity; ++j) {
          index_bc[j][k] += fast_divmoder.val[1] * configs[j].strides[i];
        }
      }
    }
    Unroller<BroadcastDataSetter, VecSize, Arity>::step(ins, args, index_bc);
  } else {
    BcUnroller<BroadcastDataLoader, IsBoundary, LoadType, VecSize, Arity>::step(
        ins, args, configs, use_broadcast, block_offset, num, numel, read_lens);
  }
#endif
  SameDimsElementwisePrimitiveCaller<ConditionalT<OutT, NumOuts>,
                                     VecSize,
                                     Functor,
                                     ArgsT,
                                     Arity>()(func, args, result, read_lens);
  phi::funcs::
      ElementwiseWriteDataCallerBc<OutT, VecSize, IsBoundary, NumOuts>()(
          outs, result, block_offset, num, read_lens);
}

template <typename Functor,
          typename OutT,
          int Arity,
          int NumOuts,
          int VecSize,
          int LoadType>
__global__ void VectorizedBroadcastKernel(
    phi::Array<const _ptr_ char *__restrict__, Arity> ins,
    phi::Array<_ptr_ OutT *, NumOuts> outs,
    phi::Array<bool, Arity> use_broadcast,
    uint32_t numel,
    phi::Array<kps::details::BroadcastConfig, Arity> configs,
    int main_offset,
    int tail_tid,
    int read_lens,
    Functor func) {
#ifdef PADDLE_WITH_XPU_KP
  int block_offset = BLOCK_ID_X * BLOCK_NUM_X * read_lens;
  int stride = BLOCK_NUM_X * GRID_NUM_X * read_lens;
  for (; block_offset < main_offset; block_offset += stride) {
    VectorizedBroadcastKernelImpl<OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  false,
                                  LoadType>(ins,
                                            outs,
                                            use_broadcast,
                                            numel,
                                            configs,
                                            BLOCK_NUM_X * read_lens,
                                            block_offset,
                                            read_lens,
                                            func);
  }
  int num = numel - block_offset;
  if (num > 0) {
    VectorizedBroadcastKernelImpl<OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  true,
                                  LoadType>(ins,
                                            outs,
                                            use_broadcast,
                                            numel,
                                            configs,
                                            num,
                                            block_offset,
                                            read_lens,
                                            func);
  }
#else
  int block_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  if (block_offset < main_offset) {
    VectorizedBroadcastKernelImpl<OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  false,
                                  LoadType>(ins,
                                            outs,
                                            use_broadcast,
                                            numel,
                                            configs,
                                            BLOCK_NUM_X * VecSize,
                                            block_offset,
                                            read_lens,
                                            func);
  } else {
    VectorizedBroadcastKernelImpl<OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  true,
                                  LoadType>(ins,
                                            outs,
                                            use_broadcast,
                                            numel,
                                            configs,
                                            tail_tid,
                                            block_offset,
                                            read_lens,
                                            func);
  }
#endif
}

template <typename OutT, typename Functor, int Arity, int NumOuts, int VecSize>
void LaunchBroadcastKernel(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    Functor func,
    const phi::Array<kps::details::BroadcastConfig, Arity> &configs,
    const LoaderTypeClassifier<OutT, Arity, Functor> &loader_classifier) {
  phi::Array<_ptr_ OutT *, NumOuts> outs_data;
  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (_ptr_ OutT *)(ctx.Alloc<OutT>((*outs)[i]));
  }

#ifdef PADDLE_WITH_XPU_KP
  int numel = (*outs)[0]->numel();
  const int threads = 64;
  const int blocks = 8;
  int read_lens = configs[0].buf_len;
  auto stream = ctx.x_context()->xpu_stream;
  int main_offset = (numel / (read_lens * threads)) * read_lens * threads;
  int tail_tid = numel % (read_lens * threads);

  VectorizedBroadcastKernel<Functor, OutT, Arity, NumOuts, VecSize, false>
      <<<blocks, threads, 0, stream>>>(loader_classifier.ins_data,
                                       outs_data,
                                       loader_classifier.use_broadcast,
                                       numel,
                                       configs,
                                       main_offset,
                                       tail_tid,
                                       read_lens,
                                       func);
#else
  const auto &numel = loader_classifier.numel;
  auto gpu_config =
      phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, VecSize);
  auto stream = ctx.stream();
  auto threads = gpu_config.GetBlockSize();
  auto blocks = gpu_config.block_per_grid;
  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);

  if (loader_classifier.all_elementwise) {
    VectorizedBroadcastKernel<Functor,
                              OutT,
                              Arity,
                              NumOuts,
                              VecSize,
                              kElementwise>
        <<<blocks, threads, 0, stream>>>(loader_classifier.ins_data,
                                         outs_data,
                                         loader_classifier.use_broadcast,
                                         numel,
                                         configs,
                                         main_offset,
                                         tail_tid,
                                         VecSize,
                                         func);
  } else if (loader_classifier.broadcast_num > (Arity >> 1)) {
    constexpr BroadcastLoadType type_ = (Arity > 1) ? kBroadcast : kMixed;
    VectorizedBroadcastKernel<Functor, OutT, Arity, NumOuts, VecSize, type_>
        <<<blocks, threads, 0, stream>>>(loader_classifier.ins_data,
                                         outs_data,
                                         loader_classifier.use_broadcast,
                                         numel,
                                         configs,
                                         main_offset,
                                         tail_tid,
                                         VecSize,
                                         func);
  } else {
    VectorizedBroadcastKernel<Functor, OutT, Arity, NumOuts, VecSize, kMixed>
        <<<blocks, threads, 0, stream>>>(loader_classifier.ins_data,
                                         outs_data,
                                         loader_classifier.use_broadcast,
                                         numel,
                                         configs,
                                         main_offset,
                                         tail_tid,
                                         VecSize,
                                         func);
  }
#endif
}

#ifndef PADDLE_WITH_XPU_KP
HOSTDEVICE static int64_t ConvertSrcIdxToDstIdx(
    int64_t src_idx,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1> &src_strides,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1> &dst_strides,
    int rank) {
  int64_t dst_idx = 0;
  int64_t old_src_idx = src_idx;
  for (int k = 0; k < rank; ++k) {
    auto local_idx = src_idx / src_strides[k + 1];
    src_idx -= local_idx * src_strides[k + 1];

    if (dst_strides[k] != dst_strides[k + 1]) {
      dst_idx += local_idx * dst_strides[k + 1];
    }
  }
  return dst_idx;
}

template <int N>
struct MaxWithOne {
  static constexpr auto kValue = (N >= 1 ? N : 1);
};

template <int Index, int VecSize>
struct ReadVecDataWithInt64Index {
  template <typename Array1, typename Array2, typename Array3, typename ArgsT>
  static __device__ __forceinline__ void Apply(
      const Array1 &in,
      ArgsT *args,
      int64_t idx,
      const Array2 &need_broadcast,
      const phi::Array<int64_t, phi::DDim::kMaxRank + 1> &src_strides,
      const Array3 &dst_strides,
      int rank,
      bool is_boundary) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    if (is_boundary) {
#pragma unroll
      for (int i = 0; i < VecSize; ++i) {
        std::get<Index>(args[i]) = in[Index][ConvertSrcIdxToDstIdx(
            idx + i, src_strides, dst_strides[Index], rank)];
      }
    } else {
      if (!need_broadcast[Index]) {
        kps::ReadData<Type, VecSize, 1, ArgsT, Index, false>(
            args, reinterpret_cast<const _ptr_ Type *>(in[Index]) + idx, 1);
      } else {
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
          std::get<Index>(args[i]) = in[Index][ConvertSrcIdxToDstIdx(
              idx + i, src_strides, dst_strides[Index], rank)];
        }
      }
    }
  }
};

template <typename OutT, typename Functor, int VecSize, int NumIns>
__global__ void BroadcastKernelWithInt64Index(
    const phi::Array<const _ptr_ char *__restrict__, MaxWithOne<NumIns>::kValue>
        &ins,
    OutT *out,
    phi::Array<phi::Array<int64_t, phi::DDim::kMaxRank + 1>,
               MaxWithOne<NumIns>::kValue> ins_strides,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> out_strides,
    phi::Array<bool, MaxWithOne<NumIns>::kValue> need_broadcasts,
    int rank,
    Functor functor) {
  int64_t numel = out_strides[0];
  int64_t idx =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VecSize;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * VecSize;
  int64_t limit = numel - VecSize;

  using Traits = phi::funcs::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;

  ArgsT args[VecSize];
  phi::AlignedVector<OutT, VecSize> out_vec;
  for (; idx <= limit; idx += stride) {
    Unroller<ReadVecDataWithInt64Index, VecSize, NumIns>::step(
        ins, args, idx, need_broadcasts, out_strides, ins_strides, rank, false);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] = static_cast<OutT>(Apply(functor, args[i]));
    }
    phi::Store<OutT, VecSize>(out_vec, out + idx);
  }

  if (idx < numel) {
    int remain = numel - idx;  // remain is always less than VecSize, therefore
                               // `int` is enough here
    Unroller<ReadVecDataWithInt64Index, VecSize, NumIns>::step(
        ins, args, idx, need_broadcasts, out_strides, ins_strides, rank, true);
    for (int i = 0; i < remain; ++i) {
      out_vec[idx + i] = static_cast<OutT>(Apply(functor, args[i]));
    }
  }
}

template <typename OutT, typename Functor, int Arity, int NumOuts, int VecSize>
struct LaunchBroadcastKernelWithInt64IndexHelper {
  static void Run(const KPDevice &ctx,
                  const std::vector<const DenseTensor *> &ins,
                  std::vector<DenseTensor *> *outs,
                  int axis,
                  Functor functor) {
    PADDLE_THROW(phi::errors::PermissionDenied(
        "Unreachable code branch. This may be a bug."));
  }
};

template <typename OutT, typename Functor, int Arity, int VecSize>
struct LaunchBroadcastKernelWithInt64IndexHelper<OutT,
                                                 Functor,
                                                 Arity,
                                                 /*NumOuts=*/1,
                                                 VecSize> {
  static void Run(const KPDevice &ctx,
                  const std::vector<const DenseTensor *> &ins,
                  std::vector<DenseTensor *> *outs,
                  int axis,
                  Functor functor) {
    phi::Array<const _ptr_ char *__restrict__, MaxWithOne<Arity>::kValue>
        ins_ptrs;
    UnrollerWithoutVecSize<InputSetter, Arity>::step(ins, &ins_ptrs);
    auto *out_tensor = (*outs)[0];
    auto *out_ptr = ctx.Alloc<OutT>(out_tensor);

    phi::Array<phi::Array<int64_t, phi::DDim::kMaxRank>,
               MaxWithOne<Arity>::kValue>
        ins_expand_dims;
    phi::Array<int64_t, phi::DDim::kMaxRank> broadcast_out_dims;
    int rank;
    if (Arity == 1) {
      rank = ins[0]->dims().size();
      for (int i = 0; i < rank; ++i) {
        broadcast_out_dims[i] = ins[0]->dims()[i];
      }
      ins_expand_dims[0] = broadcast_out_dims;
    } else if (Arity >= 2) {
      CalculateBroadcastDims(ins[0]->dims().Get(),
                             ins[1]->dims().Get(),
                             ins[0]->dims().size(),
                             ins[1]->dims().size(),
                             axis,
                             ins_expand_dims[0].GetMutable(),
                             ins_expand_dims[1].GetMutable(),
                             broadcast_out_dims.GetMutable(),
                             &rank);
      for (int i = 2; i < Arity; ++i) {
        auto tmp_dims = broadcast_out_dims;
        phi::Array<int64_t, phi::DDim::kMaxRank> tmp_expand_dims;
        int tmp_rank;
        PADDLE_ENFORCE_GE(rank,
                          ins[i]->dims().size(),
                          phi::errors::InvalidArgument(
                              "Unsupported reverse broadcast when the input "
                              "tensor number is larger than 2."));
        CalculateBroadcastDims(tmp_dims.Get(),
                               ins[i]->dims().Get(),
                               rank,
                               ins[i]->dims().size(),
                               axis,
                               tmp_expand_dims.GetMutable(),
                               ins_expand_dims[i].GetMutable(),
                               broadcast_out_dims.GetMutable(),
                               &tmp_rank);
        PADDLE_ENFORCE_EQ(rank,
                          tmp_rank,
                          phi::errors::InvalidArgument(
                              "Wrong broadcast algorithm. This may be a bug."));
      }
    }

    phi::Array<phi::Array<int64_t, phi::DDim::kMaxRank + 1>,
               MaxWithOne<Arity>::kValue>
        ins_strides;
    phi::Array<bool, MaxWithOne<Arity>::kValue> need_broadcasts;
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> out_strides;
    const auto &out_dims = out_tensor->dims();
    if (rank <= out_dims.size()) {
      out_strides = ShapeToStride(out_dims.Get(), rank);
    } else {
      out_strides = ShapeToStride(broadcast_out_dims.Get(), rank);
    }

    for (int i = 0; i < Arity; ++i) {
      ins_strides[i] = ShapeToStride(ins_expand_dims[i].Get(), rank);
      need_broadcasts[i] =
          !IsSameShape(out_strides.Get(), ins_strides[i].Get(), rank + 1);
    }

    int64_t numel = out_strides[0];
    auto gpu_config =
        phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, VecSize);

    BroadcastKernelWithInt64Index<OutT, Functor, VecSize, Arity>
        <<<gpu_config.block_per_grid,
           gpu_config.thread_per_block,
           0,
           ctx.stream()>>>(ins_ptrs,
                           out_ptr,
                           ins_strides,
                           out_strides,
                           need_broadcasts,
                           rank,
                           functor);
  }

 private:
  static void CalculateBroadcastDims(const int64_t *x_dims,
                                     const int64_t *y_dims,
                                     int nx,
                                     int ny,
                                     int axis,
                                     int64_t *x_out_dims,
                                     int64_t *y_out_dims,
                                     int64_t *broadcast_out_dims,
                                     int *length) {
    PADDLE_ENFORCE_GE(
        axis, 0, phi::errors::InvalidArgument("Invalid axis value: %d", axis));
    if (nx == ny) {
      *length = nx;
      for (int i = 0; i < nx; ++i) {
        if (x_dims[i] != y_dims[i]) {
          PADDLE_ENFORCE_EQ(
              x_dims[i] == 1 || y_dims[i] == 1,
              true,
              phi::errors::InvalidArgument("Cannot broadcast input shape where "
                                           "x_dims[%d] = %d, y_dims[%d] = %d.",
                                           i,
                                           x_dims[i],
                                           i,
                                           y_dims[i]));
        }
        broadcast_out_dims[i] = std::max(x_dims[i], y_dims[i]);
        x_out_dims[i] = x_dims[i];
        y_out_dims[i] = y_dims[i];
      }
    } else if (nx > ny) {
      *length = nx;
      for (int i = nx - axis; i < ny; ++i) {
        PADDLE_ENFORCE_EQ(
            y_dims[i],
            1,
            phi::errors::InvalidArgument(
                "The trailing Y.shape[%d] should be 1 but got %d.",
                i,
                y_dims[i]));
      }

      for (int i = 0; i < nx; ++i) {
        if (i >= axis && i - axis < ny) {
          if (x_dims[i] != y_dims[i - axis]) {
            PADDLE_ENFORCE_EQ(x_dims[i] == 1 || y_dims[i - axis] == 1,
                              true,
                              phi::errors::InvalidArgument(
                                  "Cannot broadcast input shape where "
                                  "x_dims[%d] = %d, y_dims[%d] = %d.",
                                  i,
                                  x_dims[i],
                                  i - axis,
                                  y_dims[i - axis]));
          }
          broadcast_out_dims[i] = std::max(x_dims[i], y_dims[i - axis]);
          x_out_dims[i] = x_dims[i];
          y_out_dims[i] = y_dims[i - axis];
        } else {
          broadcast_out_dims[i] = x_dims[i];
          x_out_dims[i] = x_dims[i];
          y_out_dims[i] = 1;
        }
      }
    } else {
      CalculateBroadcastDims(y_dims,
                             x_dims,
                             ny,
                             nx,
                             axis,
                             y_out_dims,
                             x_out_dims,
                             broadcast_out_dims,
                             length);
    }
  }

  static bool IsSameShape(const int64_t *x, const int64_t *y, int rank) {
    for (int i = 0; i < rank; ++i) {
      if (x[i] != y[i]) return false;
    }
    return true;
  }

  static phi::Array<int64_t, phi::DDim::kMaxRank + 1> ShapeToStride(
      const int64_t *arr, int rank) {
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> strides;
    strides[rank] = 1;
    for (int i = rank - 1; i >= 0; --i) {
      strides[i] = strides[i + 1] * arr[i];
    }
    return strides;
  }
};
#endif

template <typename OutT, typename Functor, int kArity, int NumOuts = 1>
void BroadcastKernelForDifferentVecSize(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    int axis,
    Functor func) {
#ifndef PADDLE_WITH_XPU_KP
  constexpr bool kEnabledInt64IndexKernel = (NumOuts == 1 && kArity <= 3);
  bool use_int64_index_kernel =
      kEnabledInt64IndexKernel &&
      (*outs)[0]->numel() >= std::numeric_limits<int32_t>::max();
  if (use_int64_index_kernel) {
    auto loader_classifier =
        LoaderTypeClassifier<OutT, kArity, Functor>(ins, outs);
    switch (loader_classifier.vec_size) {
      case VecSizeL: {
        LaunchBroadcastKernelWithInt64IndexHelper<OutT,
                                                  Functor,
                                                  kArity,
                                                  NumOuts,
                                                  VecSizeL>::Run(ctx,
                                                                 ins,
                                                                 outs,
                                                                 axis,
                                                                 func);
        break;
      }
      case VecSizeM: {
        LaunchBroadcastKernelWithInt64IndexHelper<OutT,
                                                  Functor,
                                                  kArity,
                                                  NumOuts,
                                                  VecSizeM>::Run(ctx,
                                                                 ins,
                                                                 outs,
                                                                 axis,
                                                                 func);
        break;
      }
      case VecSizeS: {
        LaunchBroadcastKernelWithInt64IndexHelper<OutT,
                                                  Functor,
                                                  kArity,
                                                  NumOuts,
                                                  VecSizeS>::Run(ctx,
                                                                 ins,
                                                                 outs,
                                                                 axis,
                                                                 func);
        break;
      }
      default: {
        PADDLE_THROW(phi::errors::Unimplemented(
            "Unsupported vectorized size: %d!", loader_classifier.vec_size));
        break;
      }
    }
    return;
  }
#endif

  phi::Array<kps::details::BroadcastConfig, kArity> configs;
#ifdef PADDLE_WITH_XPU_KP
  PADDLE_ENFORCE_EQ(
      ins.size(),
      2,
      phi::errors::InvalidArgument(
          "XPU only support inputs is 2, but received %d", ins.size()));

  auto loader_classifier = LoaderTypeClassifier<OutT, kArity, Functor>();
  const auto dims_simplifier =
      BroadcastDimsSimplifier(ins, (*outs)[0]->dims(), axis);
  if (VLOG_IS_ON(6)) {
    DimsSimplifiedLogger<int64_t>::Log(
        ins, outs, dims_simplifier, "XPU Broadcast");
  }
  configs[0] = kps::details::BroadcastConfig(dims_simplifier.out_dims,
                                             dims_simplifier.in_dims[0],
                                             dims_simplifier.in_dims[1],
                                             dims_simplifier.rank);
  configs[1] = kps::details::BroadcastConfig(dims_simplifier.out_dims,
                                             dims_simplifier.in_dims[1],
                                             dims_simplifier.in_dims[0],
                                             dims_simplifier.rank);
  auto type = kps::details::OptType::CanNotOptimize;
  bool is_optimize = configs[0].cmp_type != type;
  int vec_size = is_optimize ? VecSizeL : VecSizeM;
#else
  auto loader_classifier =
      LoaderTypeClassifier<OutT, kArity, Functor>(ins, outs);
  if (!loader_classifier.all_elementwise) {
    const auto dims_simplifier =
        BroadcastDimsSimplifier(ins, (*outs)[0]->dims(), axis);

    if (VLOG_IS_ON(6)) {
      DimsSimplifiedLogger<int64_t>::Log(
          ins, outs, dims_simplifier, "GPU Broadcast");
    }
    for (int i = 0; i < kArity; ++i) {
      // if data shape is[m, n], then you should set data_dim = {n, m}
      // eg: out's shape [3, 45, 1]. then out_dims = {1, 45, 3}
      // if (ins[i]->numel() != (*outs)[0]->numel()) {
      if (ins[i]->numel()) {
        configs[i] = kps::details::BroadcastConfig(dims_simplifier.out_dims,
                                                   dims_simplifier.in_dims[i],
                                                   dims_simplifier.rank);
      }
    }
  }
#endif
  switch (loader_classifier.vec_size) {
    case VecSizeL: {
      LaunchBroadcastKernel<OutT, Functor, kArity, NumOuts, VecSizeL>(
          ctx, ins, outs, func, configs, loader_classifier);
      break;
    }
    case VecSizeM: {
      LaunchBroadcastKernel<OutT, Functor, kArity, NumOuts, VecSizeM>(
          ctx, ins, outs, func, configs, loader_classifier);
      break;
    }
    case VecSizeS: {
      LaunchBroadcastKernel<OutT, Functor, kArity, NumOuts, VecSizeS>(
          ctx, ins, outs, func, configs, loader_classifier);
      break;
    }
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported vectorized size: %d!", loader_classifier.vec_size));
      break;
    }
  }
}

template <typename OutT, typename Functor, int NumOuts = 1>
void BroadcastKernel(const KPDevice &ctx,
                     const std::vector<const DenseTensor *> &ins,
                     std::vector<DenseTensor *> *outs,
                     Functor func,
                     int axis = -1) {
  // When there are multiple inputs, the outputs's rank should be equal the
  // maximum rank of all inputs.
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int kArity = Traits::arity;
  PADDLE_ENFORCE_EQ(
      ins.size(),
      kArity,
      phi::errors::InvalidArgument("The number of inputs is expected to be "
                                   "equal to the "
                                   "arity of functor. But received: the "
                                   "number of inputs "
                                   "is %d, the arity of functor is %d.",
                                   ins.size(),
                                   kArity));
  PADDLE_ENFORCE_EQ(
      outs->size(),
      NumOuts,
      phi::errors::InvalidArgument("Number of outputs shall equal to number "
                                   "of functions, "
                                   "but number of outputs is %d, of "
                                   "functions is %d.",
                                   outs->size(),
                                   NumOuts));

  int max_rank = 0;
  int min_rank = phi::DDim::kMaxRank;
  for (auto *in : ins) {
    max_rank = std::max(max_rank, in->dims().size());
    min_rank = std::min(min_rank, in->dims().size());
  }
  if (ins.size() == 1) {
    // When there is only 1 input, the input's rank may be less than outputs'
    // rank.
    max_rank = std::max(max_rank, (*outs)[0]->dims().size());
  }
  axis = axis == -1 ? max_rank - min_rank : axis;
  BroadcastKernelForDifferentVecSize<OutT, Functor, kArity, NumOuts>(
      ctx, ins, outs, axis, func);
}

template <typename Functor, typename T, typename OutType = T>
void ElementwiseCompute(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        Functor func,
                        DenseTensor *z,
                        int axis = -1) {
  std::vector<const DenseTensor *> ins = {&x, &y};
  std::vector<DenseTensor *> outs = {z};
  dev_ctx.template Alloc<OutType>(z);

  BroadcastKernel<OutType, Functor, 1>(dev_ctx, ins, &outs, func, axis);
}

template <typename DeviceContext,
          typename T,
          typename Functor,
          typename InverseFunctor>
void DefaultElementwiseOperator(const DeviceContext &dev_ctx,
                                const DenseTensor &x,
                                const DenseTensor &y,
                                DenseTensor *z,
                                int axis = -1) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  dev_ctx.template Alloc<T>(z);
  funcs::ElementwiseCompute<Functor, T>(dev_ctx, x, y, Functor(), z, axis);
}

#else

template <typename DeviceContext,
          typename T,
          typename Functor,
          typename InverseFunctor>
void DefaultElementwiseOperator(const DeviceContext &dev_ctx,
                                const DenseTensor &x,
                                const DenseTensor &y,
                                DenseTensor *z,
                                int axis = -1) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  dev_ctx.template Alloc<T>(z);
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<Functor, T>(dev_ctx, x, y, Functor(), z, axis);
  } else {
    funcs::ElementwiseCompute<InverseFunctor, T>(
        dev_ctx, x, y, InverseFunctor(), z, axis);
  }
}
#endif

}  // namespace funcs
}  // namespace phi
