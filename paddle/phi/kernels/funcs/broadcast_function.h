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

enum BroadcastType { kMixed = 1, kBroadcast = 2, kElementwise = 3 };

template <typename OutT, typename Functor, int Arity, int NumOuts>
struct BroadcastTypeClassifier {
  int64_t numel{0};
  int broadcast_num{0};              // Not used for XPU
  bool all_elementwise{true};        // Not used for XPU
  Array<bool, Arity> use_broadcast;  // Not used for XPU
  Array<kps::details::BroadcastConfig, Arity> configs;
  Array<const _ptr_ char *__restrict__, Arity> ins_data;
  Array<_ptr_ OutT *, NumOuts> outs_data;

  BroadcastTypeClassifier() {}
  BroadcastTypeClassifier(const std::vector<const DenseTensor *> &ins,
                          std::vector<DenseTensor *> *outs,
                          int axis) {
    numel = (*outs)[0]->numel();

#ifndef PADDLE_WITH_XPU_KP
    for (size_t i = 0; i < ins.size(); ++i) {
      bool is_same_dim = ins[i]->numel() == numel;
      if (is_same_dim) {
        use_broadcast[i] = false;
      } else {
        use_broadcast[i] = true;
        broadcast_num++;
      }
      all_elementwise &= is_same_dim;
    }
#endif

    InitBroadcastConfigs(ins, outs, axis);

    using Traits = phi::funcs::FunctionTraits<Functor>;
    using ArgsT = typename Traits::ArgsTuple;
    ArgsT arg;
    UnrollerWithoutVecSize<InputSetter, Arity>::step(ins, arg, &ins_data);
    for (int i = 0; i < NumOuts; ++i) {
      outs_data[i] = (*outs)[i]->data<OutT>();
    }
  }

  void InitBroadcastConfigs(const std::vector<const DenseTensor *> &ins,
                            std::vector<DenseTensor *> *outs,
                            int axis) {
#ifdef PADDLE_WITH_XPU_KP
    const auto dims_simplifier =
        BroadcastDimsSimplifier(ins, (*outs)[0]->dims(), axis);
    if (VLOG_IS_ON(6)) {
      DimsSimplifiedLogger<int64_t>::Log(
          ins, outs, dims_simplifier, "BroadcastKernel");
    }
    configs[0] = kps::details::BroadcastConfig(dims_simplifier.out_dims,
                                               dims_simplifier.in_dims[0],
                                               dims_simplifier.in_dims[1],
                                               dims_simplifier.rank);
    configs[1] = kps::details::BroadcastConfig(dims_simplifier.out_dims,
                                               dims_simplifier.in_dims[1],
                                               dims_simplifier.in_dims[0],
                                               dims_simplifier.rank);
#else
    if (!all_elementwise) {
      const auto dims_simplifier =
          BroadcastDimsSimplifier(ins, (*outs)[0]->dims(), axis);
      if (VLOG_IS_ON(6)) {
        DimsSimplifiedLogger<int64_t>::Log(
            ins, outs, dims_simplifier, "BroadcastKernel");
      }
      for (int i = 0; i < Arity; ++i) {
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
    const Array<const _ptr_ char *__restrict__, Arity> &ins,
    Array<_ptr_ OutT *, NumOuts> outs,
    const Array<bool, Arity> &use_broadcast,
    const uint32_t numel,
    const Array<kps::details::BroadcastConfig, Arity> &configs,
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
    Array<const _ptr_ char *__restrict__, Arity> ins,
    Array<_ptr_ OutT *, NumOuts> outs,
    Array<bool, Arity> use_broadcast,
    uint32_t numel,
    Array<kps::details::BroadcastConfig, Arity> configs,
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
    const BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts> &classifier,
    Functor func) {
#ifdef PADDLE_WITH_XPU_KP
  int numel = classifier.numel;
  const int threads = 64;
  const int blocks = 8;
  int read_lens = configs[0].buf_len;
  auto stream = ctx.x_context()->xpu_stream;
  int main_offset = (numel / (read_lens * threads)) * read_lens * threads;
  int tail_tid = numel % (read_lens * threads);

  VectorizedBroadcastKernel<Functor, OutT, Arity, NumOuts, VecSize, false>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       classifier.use_broadcast,
                                       numel,
                                       classifier.configs,
                                       main_offset,
                                       tail_tid,
                                       read_lens,
                                       func);
#else
  const auto &numel = classifier.numel;
  auto gpu_config =
      phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, VecSize);
  auto stream = ctx.stream();
  auto threads = gpu_config.GetBlockSize();
  auto blocks = gpu_config.block_per_grid;
  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);

  if (classifier.all_elementwise) {
    VectorizedBroadcastKernel<Functor,
                              OutT,
                              Arity,
                              NumOuts,
                              VecSize,
                              kElementwise>
        <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                         classifier.outs_data,
                                         classifier.use_broadcast,
                                         numel,
                                         classifier.configs,
                                         main_offset,
                                         tail_tid,
                                         VecSize,
                                         func);
  } else if (classifier.broadcast_num > (Arity >> 1)) {
    constexpr BroadcastType type_ = (Arity > 1) ? kBroadcast : kMixed;
    VectorizedBroadcastKernel<Functor, OutT, Arity, NumOuts, VecSize, type_>
        <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                         classifier.outs_data,
                                         classifier.use_broadcast,
                                         numel,
                                         classifier.configs,
                                         main_offset,
                                         tail_tid,
                                         VecSize,
                                         func);
  } else {
    VectorizedBroadcastKernel<Functor, OutT, Arity, NumOuts, VecSize, kMixed>
        <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                         classifier.outs_data,
                                         classifier.use_broadcast,
                                         numel,
                                         classifier.configs,
                                         main_offset,
                                         tail_tid,
                                         VecSize,
                                         func);
  }
#endif
}

template <typename OutT, typename Functor, int Arity, int NumOuts = 1>
typename std::enable_if<!NeedVectorized<OutT>::value, void>::type
BroadcastKernelForDifferentVecSize(const KPDevice &ctx,
                                   const std::vector<const DenseTensor *> &ins,
                                   std::vector<DenseTensor *> *outs,
                                   int axis,
                                   Functor func) {
  auto classifier =
      BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(ins, outs, axis);
  LaunchBroadcastKernel<OutT, Functor, Arity, NumOuts, VecSizeS>(
      ctx, classifier, func);
}

template <typename OutT, typename Functor, int Arity, int NumOuts = 1>
typename std::enable_if<NeedVectorized<OutT>::value, void>::type
BroadcastKernelForDifferentVecSize(const KPDevice &ctx,
                                   const std::vector<const DenseTensor *> &ins,
                                   std::vector<DenseTensor *> *outs,
                                   int axis,
                                   Functor func) {
#ifdef PADDLE_WITH_XPU_KP
  auto type = kps::details::OptType::CanNotOptimize;
  bool is_optimize = classifier.configs[0].cmp_type != type;
  int vec_size = is_optimize ? VecSizeL : VecSizeM;
#else
  // Calculate the max vec_size for all ins and outs.
  int vec_size = GetVectorizedSizeForTensors(ins, *outs);
#endif

  auto classifier =
      BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(ins, outs, axis);
  switch (vec_size) {
    case VecSizeL: {
      LaunchBroadcastKernel<OutT, Functor, Arity, NumOuts, VecSizeL>(
          ctx, classifier, func);
      break;
    }
    case VecSizeM: {
      LaunchBroadcastKernel<OutT, Functor, Arity, NumOuts, VecSizeM>(
          ctx, classifier, func);
      break;
    }
    case VecSizeS: {
      LaunchBroadcastKernel<OutT, Functor, Arity, NumOuts, VecSizeS>(
          ctx, classifier, func);
      break;
    }
    default: {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d!", vec_size));
      break;
    }
  }
}

static void updateStridesDims(std::vector<int64_t> *strides,
                              std::vector<int64_t> *dims) {
  for (int i = 1; i < strides->size(); i++) {
    (*strides)[i] = (*strides)[i - 1] * (*dims)[i - 1];
  }
  // reverse origin_in_dim and origin_in_stride if in's dim_size > 0
  std::reverse(strides->begin(), strides->end());
  std::reverse(dims->begin(), dims->end());
}

static void SliceTensor(DenseTensor *x,
                        const DenseTensor *share,
                        const std::vector<int64_t> &out_compute_dims,
                        int64_t offset) {
  auto new_dim = common::make_ddim(out_compute_dims);
  DenseTensorMeta meta(share->dtype(),
                       new_dim,
                       share->layout(),
                       offset * SizeOf(share->dtype()));
  x->set_meta(meta);
  x->ShareBufferWith(*(share), true);
  x->Resize(new_dim);
}

template <typename OutT, typename Functor, int kArity, int NumOuts = 1>
void BroadcastKernelSplit(const KPDevice &ctx,
                          const std::vector<const DenseTensor *> &ins,
                          std::vector<DenseTensor *> *outs,
                          int axis,
                          Functor func,
                          const int64_t compute_size) {
  const auto dims_simplifier =
      BroadcastDimsSimplifier(ins, (*outs)[0]->dims(), axis);
  if (VLOG_IS_ON(6)) {
    DimsSimplifiedLogger<int64_t>::Log(
        ins, outs, dims_simplifier, "GPU Broadcast");
  }

  int all_rank = dims_simplifier.rank;
  std::vector<int64_t> origin_out_strides(all_rank, 1);
  auto origin_in_dims = dims_simplifier.in_dims;
  auto origin_out_dims = dims_simplifier.out_dims;
  auto origin_in_strides = dims_simplifier.in_dims;

  // for split
  std::vector<int64_t> loop_num_out(all_rank, 1);
  std::vector<int64_t> loop_num_out_stride(all_rank, 1);

  // for input's offset
  std::vector<int64_t> ins_offset(kArity, 0);
  std::vector<int64_t> ins_scale_for_dim(kArity, 0);

  // init offset and check in's dim
  for (int k = 0; k < kArity; k++) {
    ins_scale_for_dim[k] = ins[k]->dims().size() == 0 ? 0 : 1;
    if (ins_scale_for_dim[k]) {
      origin_in_strides[k][0] = 1;
    }
  }

  updateStridesDims(&origin_out_strides, &origin_out_dims);
  for (int k = 0; k < kArity; k++) {
    if (ins_scale_for_dim[k]) {
      updateStridesDims(&origin_in_strides[k], &origin_in_dims[k]);
    }
  }

  // init out_split_dim and in_split_dims
  auto out_split_dim = origin_out_dims;
  auto in_split_dims = origin_in_dims;

  // init
  int64_t loop_num = 1;
  int64_t split_idx = 0;

  for (int r = 0; r < all_rank; r++) {
    // if the compute_size was too small the split_size must be 0, but the
    // dim_num must ge 1
    int64_t split_size = compute_size / origin_out_strides[r];
    out_split_dim[r] = std::max(split_size, static_cast<int64_t>(1));
    loop_num_out[r] =
        (origin_out_dims[r] + out_split_dim[r] - 1) / out_split_dim[r];
    loop_num *= loop_num_out[r];

    for (int k = 0; k < kArity; k++) {
      if (ins_scale_for_dim[k]) {
        in_split_dims[k][r] = std::min(origin_in_dims[k][r], out_split_dim[r]);
      }
    }

    // split_idx is the index for lash split dim
    if (split_size != 0) {
      split_idx = r;
      break;
    }
  }

  loop_num_out_stride[all_rank - 1] = 1;
  for (int r = all_rank - 2; r >= 0; r--) {
    loop_num_out_stride[r] = loop_num_out_stride[r + 1] * loop_num_out[r + 1];
  }

  // compute

  for (int iter = 0; iter < loop_num; iter++) {
    std::vector<const DenseTensor *> new_ins = {};
    std::vector<DenseTensor *> new_outs = {};
    phi::DenseTensor tmp_in[kArity];
    DenseTensor tmp_out[NumOuts];

    int64_t tmp_size = iter;
    int64_t out_offset = 0;
    // compute the offset before  last split dim
    for (int i = 0; i < split_idx; i++) {
      auto repeat_times = tmp_size / loop_num_out_stride[i];
      out_offset += repeat_times * origin_out_strides[i];
      for (int k = 0; k < kArity; k++) {
        if (ins_scale_for_dim[k]) {
          ins_offset[k] +=
              (repeat_times % origin_in_dims[k][i]) * origin_in_strides[k][i];
        }
      }
      tmp_size = tmp_size % loop_num_out_stride[i];
    }
    // tmp_size is the last split_dims's repeat idx
    auto pre_deal_size = tmp_size * out_split_dim[split_idx];
    out_offset += pre_deal_size * origin_out_strides[split_idx];
    // compute_size
    auto remainder_size = origin_out_dims[split_idx] - pre_deal_size;

    // get current compute size
    auto out_compute_dims = out_split_dim;
    out_compute_dims[split_idx] =
        std::min(out_split_dim[split_idx], remainder_size);

    // in + compute_size
    auto in_compute_dims = in_split_dims;
    for (int k = 0; k < kArity; k++) {
      if (ins_scale_for_dim[k]) {
        auto split_repeat =
            origin_in_dims[k][split_idx] == origin_out_dims[split_idx]
                ? tmp_size
                : 0;
        ins_offset[k] += split_repeat * in_split_dims[k][split_idx] *
                         origin_in_strides[k][split_idx];
        in_compute_dims[k][split_idx] =
            std::min(in_split_dims[k][split_idx], out_compute_dims[split_idx]);
      }
      SliceTensor(&tmp_in[k],
                  ins[k],
                  in_compute_dims[k],
                  ins_scale_for_dim[k] * ins_offset[k]);
      new_ins.emplace_back(&tmp_in[k]);
      ins_offset[k] = 0;
    }

    for (int n = 0; n < NumOuts; n++) {
      SliceTensor(&tmp_out[n], (*outs)[n], out_compute_dims, out_offset);
      new_outs.emplace_back(&tmp_out[n]);
    }

    BroadcastKernelForDifferentVecSize<OutT, Functor, kArity, NumOuts>(
        ctx, new_ins, &new_outs, axis, func);
  }
}

template <typename OutT, typename Functor, int kArity, int NumOuts = 1>
void BroadcastKernelApply(const KPDevice &ctx,
                          const std::vector<const DenseTensor *> &ins,
                          std::vector<DenseTensor *> *outs,
                          int axis,
                          Functor func) {
#ifndef PADDLE_WITH_XPU_KP
  constexpr bool kEnabledInt64IndexKernel = (NumOuts == 1 && kArity <= 3);
  // check whether need broadcast
  auto compute_size = std::numeric_limits<int32_t>::max();
  bool use_int64_index_kernel =
      kEnabledInt64IndexKernel && (*outs)[0]->numel() >= compute_size;

  if (use_int64_index_kernel) {  // use_int64_index_kernel
    BroadcastKernelSplit<OutT, Functor, kArity, NumOuts>(
        ctx, ins, outs, axis, func, compute_size);
    return;
  }
#endif
  BroadcastKernelForDifferentVecSize<OutT, Functor, kArity, NumOuts>(
      ctx, ins, outs, axis, func);
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

#ifdef PADDLE_WITH_XPU_KP
  PADDLE_ENFORCE_EQ(
      ins.size(),
      2,
      common::errors::InvalidArgument(
          "XPU only support inputs is 2, but received %d", ins.size()));
#endif

  PADDLE_ENFORCE_EQ(
      ins.size(),
      kArity,
      common::errors::InvalidArgument("The number of inputs is expected to be "
                                      "equal to the "
                                      "arity of functor. But received: the "
                                      "number of inputs "
                                      "is %d, the arity of functor is %d.",
                                      ins.size(),
                                      kArity));
  PADDLE_ENFORCE_EQ(
      outs->size(),
      NumOuts,
      common::errors::InvalidArgument("Number of outputs shall equal to number "
                                      "of functions, "
                                      "but number of outputs is %d, of "
                                      "functions is %d.",
                                      outs->size(),
                                      NumOuts));

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
    ctx.template Alloc<OutT>((*outs)[i]);
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
    // When there is only 1 input, the input's rank may be less than outputs'
    // rank.
    max_rank = std::max(max_rank, (*outs)[0]->dims().size());
  }
  axis = axis == -1 ? max_rank - min_rank : axis;
  BroadcastKernelApply<OutT, Functor, kArity, NumOuts>(
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
