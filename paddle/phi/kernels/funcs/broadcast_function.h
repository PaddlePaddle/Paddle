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

#include "paddle/phi/kernels/funcs/elementwise_base.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)

namespace kps = phi::kps;

#endif

namespace phi {
namespace funcs {

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)

struct DimensionsTransform {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(
      bool &, std::vector<DimVector> &, DimVector &, int, int);
  int64_t N;
  int64_t dim_size;
  DimVector out_dims;
  std::vector<DimVector> in_dims;

 private:
  // To compensate the lackage of input_tensors` dimension with input
  // variable 'axis'.
  void InputDimensionsExtend(int N, int axis) {
    for (auto &in_dim : in_dims) {
      int64_t in_idx = 0;
      if (in_dim.size() < dim_size) {
        DimVector tmp_dim(dim_size, 1);
        do {
          if (in_dim[in_idx] == out_dims[axis] || in_dim[in_idx] == 1) {
            tmp_dim[axis] = in_dim[in_idx];
            in_idx++;
            axis++;
          } else {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "received %d.",
                in_idx + 1,
                axis + 1,
                out_dims[axis],
                in_dim[in_idx]));
          }
        } while (in_idx < in_dim.size());
        in_dim.resize(dim_size);
        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      } else {
        do {
          if (in_dim[in_idx] == out_dims[in_idx] || in_dim[in_idx] == 1) {
            in_idx++;
          } else {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "received %d.",
                in_idx + 1,
                in_idx + 1,
                out_dims[in_idx],
                in_dim[in_idx]));
          }
        } while (in_idx < dim_size);
      }
      std::reverse(in_dim.begin(), in_dim.end());
    }
    std::reverse(out_dims.begin(), out_dims.end());
  }

  // Merge sequential dimension to shrink calculation cost for
  // offset computation in CUDA Kernel.
  template <typename MergeFunctor>
  __inline__ void MergeDimensions(MergeFunctor merge_func, int N) {
    auto VectorReorganise = [](DimVector *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] = std::accumulate(vec->begin() + l_idx,
                                          vec->begin() + m_idx,
                                          1,
                                          std::multiplies<int64_t>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };

    int64_t i = 0;
    while (i < dim_size) {
      int cnt = 0;
      int low_idx = i;
      bool equal = true;
      do {
        merge_func(equal, in_dims, out_dims, i, N);
        if (equal) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < dim_size);

      if (cnt > 1) {
        for (auto &in_dim : in_dims) {
          VectorReorganise(&in_dim, low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        dim_size -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

  // To judge whether shape of any input tensors is sequential
  // 1-value-dimensions, and metric the length of it.
  bool FindSequentialOneDim(int *swap_index) {
    int index = 0;
    int max_one_length = 0;
    for (int j = 0; j < N; ++j) {
      int seq_one_length = 0;
      bool active_seq = false;

      for (int i = 0; i < dim_size; ++i) {
        if (!active_seq && in_dims[j][i] == 1) {
          seq_one_length = 1;
          active_seq = true;
        } else if (active_seq) {
          if (in_dims[j][i] == 1) {
            seq_one_length++;
          } else {
            active_seq = false;
          }
        }
      }
      index = seq_one_length > max_one_length ? j : index;
      max_one_length = std::max(seq_one_length, max_one_length);
    }

    bool has_seq_one = max_one_length > 1;
    if (has_seq_one) {
      std::swap(in_dims[0], in_dims[index]);
      *swap_index = index;
    }
    return has_seq_one;
  }

 public:
  explicit DimensionsTransform(const std::vector<const DenseTensor *> &ins,
                               const phi::DDim &dims,
                               int axis) {
    N = std::max(static_cast<int>(ins.size()), 2);
    dim_size = dims.size();
    out_dims = phi::vectorize<int64_t>(dims);
    in_dims.resize(N);
    if (ins.size() == 1) {
      // when ins.size() = 1, broadcast input to output
      in_dims[0] = phi::vectorize<int64_t>(ins[0]->dims());
      in_dims[1] = out_dims;
      // Add out_dims to in_dims to avoid errors in dims merging
    } else {
      for (int j = 0; j < N; ++j) {
        in_dims[j] = phi::vectorize<int64_t>(ins[j]->dims());
      }
    }
    InputDimensionsExtend(N, axis);

    // To Merge the dimensions of input_tensors while the consequtive
    // equal-dimensions appears. Example below :
    //   in_1.shape = [2, 3, 4, 5]    in_1.shape = [2, 12, 5]
    //   in_2.shape = [1, 3, 4, 5] -> in_2.shape = [1, 12, 5]
    //   in_3.shape = [2, 3, 4, 1]    in_3.shape = [2, 12, 1]
    auto merge_sequential_dims = [](bool &equal,
                                    std::vector<DimVector> &in_dims,
                                    DimVector &out,
                                    int i,
                                    int num) {
      for (int j = 1; j < num; ++j) {
        equal &= (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
    MergeFunctor merge_ptr = merge_sequential_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);

    // To Merge the dimension of input_tensors while the sequential
    // 1-value-dimensions appears. Example below :
    //   in_1.shape = [2, 1, 1, 5]    in_1.shape = [2,  1, 5]
    //   in_2.shape = [2, 3, 4, 5] -> in_2.shape = [1, 12, 5]
    //   in_3.shape = [2, 3, 4, 1]    in_3.shape = [2, 12, 1]
    // Caution: Once 1-value-dimensions appears, the corresponding
    // shape position of other input tensors must be same with the
    // output tensor`s shape, or incorrect merge may occur.
    auto merge_sequential_one_dims = [](bool &equal,
                                        std::vector<DimVector> &in_dims,
                                        DimVector &out,
                                        int i,
                                        int num) {
      equal = in_dims[0][i] == 1;
      if (equal) {
        for (int j = 1; j < num; ++j) {
          equal &= in_dims[j][i] == out[i];
        }
      }
    };
    for (auto i = 0; i < dim_size; ++i) {
      int swap_idx = 0;
      bool has_seq_one = FindSequentialOneDim(&swap_idx);
      if (!has_seq_one) break;
      merge_ptr = merge_sequential_one_dims;
      MergeDimensions<MergeFunctor>(merge_ptr, N);
      std::swap(in_dims[swap_idx], in_dims[0]);
    }
  }
};

template <typename InT, typename OutT>
int GetVecsize(const std::vector<const DenseTensor *> &ins,
               std::vector<DenseTensor *> *outs) {
  int in_vec_size = 4;
  int out_vec_size = 4;
  if (outs->size() > 1) {
    for (auto i = 1; i < outs->size(); ++i) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          phi::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, but "
              "%d-th output tensor`s shape is not.",
              i));
      out_vec_size = std::min(
          phi::GetVectorizedSize<OutT>((*outs)[i]->data<OutT>()), out_vec_size);
    }
  } else {
    out_vec_size = phi::GetVectorizedSize<OutT>((*outs)[0]->data<OutT>());
  }

  for (auto *in : ins) {
    auto temp_size = phi::GetVectorizedSize<InT>(in->data<InT>());
    in_vec_size = in->dims() == (*outs)[0]->dims()
                      ? std::min(temp_size, in_vec_size)
                      : in_vec_size;
  }
  return std::min(out_vec_size, in_vec_size);
}

#ifndef PADDLE_WITH_XPU_KP
template <typename T,
          int VecSize,
          int Arity,
          bool IsBoundary,
          bool is_all_broadcast>
struct BroadcastDataLoader {
  __device__ __forceinline__ void operator()(
      T args[Arity][VecSize],
      const phi::Array<const _ptr_ T *__restrict__, Arity> &ins,
      const phi::Array<kps::details::BroadcastConfig, Arity> &configs,
      const phi::Array<int, Arity> &use_broadcast,
      const int block_offset,
      const int num,
      const uint32_t numel) {
#pragma unroll
    for (int i = 0; i < Arity; ++i) {
      kps::Init<T, VecSize>(args[i], static_cast<T>(1.0f));
      if (use_broadcast[i]) {
        kps::ReadDataBc<T, VecSize, 1, IsBoundary>(
            args[i], ins[i], block_offset, configs[i], numel, VecSize);
      } else {
        kps::ReadData<T, VecSize, 1, IsBoundary>(
            args[i], ins[i] + block_offset, num, VecSize);
      }
    }
  }
};

template <typename T, int VecSize, int Arity, bool IsBoundary>
struct BroadcastDataLoader<T, VecSize, Arity, IsBoundary, true> {
  __device__ __forceinline__ void operator()(
      T args[Arity][VecSize],
      const phi::Array<const _ptr_ T *__restrict__, Arity> &ins,
      const phi::Array<kps::details::BroadcastConfig, Arity> &configs,
      const phi::Array<int, Arity> &use_broadcast,
      const int block_offset,
      const int num,
      const uint32_t numel) {
    uint32_t index_bc[Arity][VecSize];
#pragma unroll
    for (int j = 0; j < Arity; ++j) {
#pragma unroll
      for (int k = 0; k < VecSize; ++k) {
        index_bc[j][k] = 0;
        args[j][k] = static_cast<T>(1);
      }
    }

    uint32_t thread_offset = block_offset + threadIdx.x * VecSize;
#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
      uint32_t idx = thread_offset + k;
      if (IsBoundary) {
        if (idx == numel) break;
      }

#pragma unroll
      for (int i = 0; i < phi::DDim::kMaxRank; ++i) {
        if (i == configs[0].kDims) break;
        auto fast_divmoder = configs[0].divmoders[i].Divmod(idx);
        idx = fast_divmoder.val[0];
#pragma unroll
        for (int j = 0; j < Arity; ++j) {
          index_bc[j][k] += fast_divmoder.val[1] * configs[j].strides[i];
        }
      }
    }

#pragma unroll
    for (int j = 0; j < Arity; ++j) {
#pragma unroll
      for (int k = 0; k < VecSize; ++k) {
        args[j][k] = ins[j][index_bc[j][k]];
      }
    }
  }
};
#endif

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary,
          bool IsAllBroadcast = false>
__device__ void VectorizedBroadcastKernelImpl(
    const phi::Array<const _ptr_ InT *__restrict__, Arity> &ins,
    phi::Array<_ptr_ OutT *, NumOuts> outs,
    const phi::Array<int, Arity> &use_broadcast,
    const uint32_t numel,
    const phi::Array<kps::details::BroadcastConfig, Arity> &configs,
    int num,
    int block_offset,
    int read_lens,
    Functor func) {
  __simd__ InT args[Arity][VecSize];
  __simd__ ConditionalT<OutT, NumOuts> result[VecSize];
#ifdef PADDLE_WITH_XPU_KP
#pragma unroll
  for (int i = 0; i < Arity; ++i) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f), read_lens);
    if (use_broadcast[i]) {
      kps::ReadDataBc<InT, VecSize, 1, IsBoundary>(
          args[i], ins[i], block_offset, configs[i], numel, read_lens);
    } else {
      kps::ReadData<InT, VecSize, 1, IsBoundary>(
          args[i], ins[i] + block_offset, num, read_lens);
    }
  }
#else
  BroadcastDataLoader<InT, VecSize, Arity, IsBoundary, IsAllBroadcast>()(
      args, ins, configs, use_broadcast, block_offset, num, numel);
#endif

  constexpr bool kCallElementwiseAny =
      paddle::platform::FunctionTraits<Functor>::has_pointer_args;
  phi::funcs::ElementwisePrimitiveCaller<InT,
                                         ConditionalT<OutT, NumOuts>,
                                         VecSize,
                                         Functor,
                                         Arity,
                                         kCallElementwiseAny>()(
      func, args, result, read_lens);
  phi::funcs::
      ElementwiseWriteDataCallerBc<OutT, VecSize, IsBoundary, NumOuts>()(
          outs, result, block_offset, num, read_lens);
}

template <typename Functor,
          typename InT,
          typename OutT,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsAllBroadcast>
__global__ void VectorizedBroadcastKernel(
    phi::Array<const _ptr_ InT *__restrict__, Arity> ins,
    phi::Array<_ptr_ OutT *, NumOuts> outs,
    phi::Array<int, Arity> use_broadcast,
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
    VectorizedBroadcastKernelImpl<InT,
                                  OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  false,
                                  IsAllBroadcast>(ins,
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
    VectorizedBroadcastKernelImpl<InT,
                                  OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  true,
                                  IsAllBroadcast>(ins,
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
    VectorizedBroadcastKernelImpl<InT,
                                  OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  false,
                                  IsAllBroadcast>(ins,
                                                  outs,
                                                  use_broadcast,
                                                  numel,
                                                  configs,
                                                  BLOCK_NUM_X * VecSize,
                                                  block_offset,
                                                  read_lens,
                                                  func);
  } else {
    VectorizedBroadcastKernelImpl<InT,
                                  OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  true,
                                  IsAllBroadcast>(ins,
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

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
void LaunchBroadcastKernel(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    Functor func,
    const phi::Array<kps::details::BroadcastConfig, Arity> &configs) {
  int broadcast_num = 0;
  int numel = (*outs)[0]->numel();
  phi::Array<int, Arity> use_broadcast;
  phi::Array<const _ptr_ InT *__restrict__, Arity> ins_data;
  phi::Array<_ptr_ OutT *, NumOuts> outs_data;

  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (_ptr_ OutT *)(ctx.Alloc<OutT>((*outs)[i]));
  }

  for (int i = 0; i < Arity; ++i) {
    if (ins[i]->numel() != numel) {
      broadcast_num++;
      use_broadcast[i] = true;
    } else {
      use_broadcast[i] = false;
    }
    ins_data[i] = (const _ptr_ InT *)(ins[i]->data<InT>());
  }

#ifdef PADDLE_WITH_XPU_KP
  const int threads = 64;
  const int blocks = 8;
  int read_lens = configs[0].buf_len;
  auto stream = ctx.x_context()->xpu_stream;
  int main_offset = (numel / (read_lens * threads)) * read_lens * threads;
  int tail_tid = numel % (read_lens * threads);

  VectorizedBroadcastKernel<Functor, InT, OutT, Arity, NumOuts, VecSize, false>
      <<<blocks, threads, 0, stream>>>(ins_data,
                                       outs_data,
                                       use_broadcast,
                                       numel,
                                       configs,
                                       main_offset,
                                       tail_tid,
                                       read_lens,
                                       func);
#else
  auto gpu_config =
      phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, VecSize);
  int read_lens = VecSize;
  auto stream = ctx.stream();
  auto threads = gpu_config.thread_per_block;
  auto blocks = gpu_config.block_per_grid;
  int main_offset = (numel / (read_lens * gpu_config.GetBlockSize())) *
                    read_lens * gpu_config.GetBlockSize();
  int tail_tid = numel % (read_lens * gpu_config.GetBlockSize());

  if (broadcast_num > (Arity >> 1)) {
    VectorizedBroadcastKernel<Functor,
                              InT,
                              OutT,
                              Arity,
                              NumOuts,
                              VecSize,
                              (Arity > 1)>
        <<<blocks, threads, 0, stream>>>(ins_data,
                                         outs_data,
                                         use_broadcast,
                                         numel,
                                         configs,
                                         main_offset,
                                         tail_tid,
                                         read_lens,
                                         func);
  } else {
    VectorizedBroadcastKernel<Functor,
                              InT,
                              OutT,
                              Arity,
                              NumOuts,
                              VecSize,
                              false>
        <<<blocks, threads, 0, stream>>>(ins_data,
                                         outs_data,
                                         use_broadcast,
                                         numel,
                                         configs,
                                         main_offset,
                                         tail_tid,
                                         read_lens,
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

template <typename T, int VecSize, bool IsBoundary>
HOSTDEVICE static void ReadVecDataWithInt64Index(
    const T *in,
    int64_t idx,
    bool need_broadcast,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1> &src_strides,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1> &dst_strides,
    int rank,
    int n,
    phi::AlignedVector<T, VecSize> *out) {
  if (IsBoundary) {
    for (int i = 0; i < n; ++i) {
      (*out)[i] =
          in[ConvertSrcIdxToDstIdx(idx + i, src_strides, dst_strides, rank)];
    }
  } else {
    if (!need_broadcast) {
      phi::Load<T, VecSize>(in + idx, out);
    } else {
#pragma unroll
      for (int i = 0; i < VecSize; ++i) {
        (*out)[i] =
            in[ConvertSrcIdxToDstIdx(idx + i, src_strides, dst_strides, rank)];
      }
    }
  }
}

template <typename InT,
          typename OutT,
          typename Functor,
          int VecSize,
          int NumIns>
struct ApplyFunctorWithInt64IndexHelper {
  HOSTDEVICE static OutT Run(const phi::AlignedVector<InT, VecSize> *ins_vec,
                             Functor functor,
                             int i);
};

template <typename InT, typename OutT, typename Functor, int VecSize>
struct ApplyFunctorWithInt64IndexHelper<InT, OutT, Functor, VecSize, 0> {
  HOSTDEVICE static OutT Run(const phi::AlignedVector<InT, VecSize> *ins_vec,
                             Functor functor,
                             int i) {
    return static_cast<OutT>(functor());
  }
};

template <typename InT, typename OutT, typename Functor, int VecSize>
struct ApplyFunctorWithInt64IndexHelper<InT, OutT, Functor, VecSize, 1> {
  HOSTDEVICE static OutT Run(const phi::AlignedVector<InT, VecSize> *ins_vec,
                             Functor functor,
                             int i) {
    return static_cast<OutT>(functor(ins_vec[0][i]));
  }
};

template <typename InT, typename OutT, typename Functor, int VecSize>
struct ApplyFunctorWithInt64IndexHelper<InT, OutT, Functor, VecSize, 2> {
  HOSTDEVICE static OutT Run(const phi::AlignedVector<InT, VecSize> *ins_vec,
                             Functor functor,
                             int i) {
    return static_cast<OutT>(functor(ins_vec[0][i], ins_vec[1][i]));
  }
};

template <typename InT, typename OutT, typename Functor, int VecSize>
struct ApplyFunctorWithInt64IndexHelper<InT, OutT, Functor, VecSize, 3> {
  HOSTDEVICE static OutT Run(const phi::AlignedVector<InT, VecSize> *ins_vec,
                             Functor functor,
                             int i) {
    return static_cast<OutT>(
        functor(ins_vec[0][i], ins_vec[1][i], ins_vec[2][i]));
  }
};

template <int N>
struct MaxWithOne {
  static constexpr auto kValue = (N >= 1 ? N : 1);
};

template <typename InT,
          typename OutT,
          typename Functor,
          int VecSize,
          int NumIns>
__global__ void BroadcastKernelWithInt64Index(
    phi::Array<const InT *, MaxWithOne<NumIns>::kValue> ins,
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

  phi::Array<phi::AlignedVector<InT, VecSize>, MaxWithOne<NumIns>::kValue>
      ins_vec;
  phi::AlignedVector<OutT, VecSize> out_vec;
  for (; idx <= limit; idx += stride) {
#pragma unroll
    for (int i = 0; i < NumIns; ++i) {
      ReadVecDataWithInt64Index<InT, VecSize, false>(ins[i],
                                                     idx,
                                                     need_broadcasts[i],
                                                     out_strides,
                                                     ins_strides[i],
                                                     rank,
                                                     VecSize,
                                                     &ins_vec[i]);
    }

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] = ApplyFunctorWithInt64IndexHelper<InT,
                                                    OutT,
                                                    Functor,
                                                    VecSize,
                                                    NumIns>::Run(ins_vec.Get(),
                                                                 functor,
                                                                 i);
    }

    phi::Store<OutT, VecSize>(out_vec, out + idx);
  }

  if (idx < numel) {
    int remain = numel - idx;  // remain is always less than VecSize, therefore
                               // `int` is enough here
#pragma unroll
    for (int i = 0; i < NumIns; ++i) {
      ReadVecDataWithInt64Index<InT, VecSize, true>(ins[i],
                                                    idx,
                                                    need_broadcasts[i],
                                                    out_strides,
                                                    ins_strides[i],
                                                    rank,
                                                    remain,
                                                    &ins_vec[i]);
    }

    for (int i = 0; i < remain; ++i) {
      out[idx + i] =
          ApplyFunctorWithInt64IndexHelper<InT,
                                           OutT,
                                           Functor,
                                           VecSize,
                                           NumIns>::Run(ins_vec.Get(),
                                                        functor,
                                                        i);
    }
  }
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
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

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize>
struct LaunchBroadcastKernelWithInt64IndexHelper<InT,
                                                 OutT,
                                                 Functor,
                                                 Arity,
                                                 /*NumOuts=*/1,
                                                 VecSize> {
  static void Run(const KPDevice &ctx,
                  const std::vector<const DenseTensor *> &ins,
                  std::vector<DenseTensor *> *outs,
                  int axis,
                  Functor functor) {
    phi::Array<const InT *, MaxWithOne<Arity>::kValue> ins_ptrs;
    for (int i = 0; i < Arity; ++i) {
      ins_ptrs[i] = ins[i]->data<InT>();
    }
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

    BroadcastKernelWithInt64Index<InT, OutT, Functor, VecSize, Arity>
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

template <ElementwiseType ET,
          typename InT,
          typename OutT,
          typename Functor,
          int NumOuts = 1>
void BroadcastKernelForDifferentVecSize(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    int axis,
    Functor func) {
  using Traits = paddle::platform::FunctionTraits<Functor>;
  const int kArity =
      Traits::has_pointer_args ? static_cast<int>(ET) : Traits::arity;
  PADDLE_ENFORCE_EQ(
      ins.size(),
      kArity,
      phi::errors::InvalidArgument("The number of inputs is expected to be "
                                   "equal to the "
                                   "arity of functor. But recieved: the "
                                   "number of inputs "
                                   "is %d, the arity of functor is %d.",
                                   ins.size(),
                                   kArity));
  PADDLE_ENFORCE_LE(
      kArity,
      3,
      phi::errors::InvalidArgument("Currently only broadcast of ternary is "
                                   "supported "
                                   "and verified, but received %d.",
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

#ifndef PADDLE_WITH_XPU_KP
  constexpr bool kEnabledInt64IndexKernel = (NumOuts == 1 && kArity <= 3);
  bool use_int64_index_kernel =
      kEnabledInt64IndexKernel &&
      (*outs)[0]->numel() >= std::numeric_limits<int32_t>::max();
  if (use_int64_index_kernel) {
    int vec_size = GetVecsize<InT, OutT>(ins, outs);
    switch (vec_size) {
      case VecSizeL: {
        LaunchBroadcastKernelWithInt64IndexHelper<InT,
                                                  OutT,
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
        LaunchBroadcastKernelWithInt64IndexHelper<InT,
                                                  OutT,
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
        LaunchBroadcastKernelWithInt64IndexHelper<InT,
                                                  OutT,
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
            "Unsupported vectorized size: %d!", vec_size));
        break;
      }
    }
    return;
  }
#endif

  // mergedim and get vec_size
  const auto merge_dims = DimensionsTransform(ins, (*outs)[0]->dims(), axis);
  phi::Array<kps::details::BroadcastConfig, kArity> configs;

// get vec_size
#ifdef PADDLE_WITH_XPU_KP
  PADDLE_ENFORCE_EQ(
      ins.size(),
      2,
      phi::errors::InvalidArgument(
          "XPU only support inputs is 2, but received %d", ins.size()));
  configs[0] = kps::details::BroadcastConfig(merge_dims.out_dims,
                                             merge_dims.in_dims[0],
                                             merge_dims.in_dims[1],
                                             merge_dims.dim_size);
  configs[1] = kps::details::BroadcastConfig(merge_dims.out_dims,
                                             merge_dims.in_dims[1],
                                             merge_dims.in_dims[0],
                                             merge_dims.dim_size);
  auto type = kps::details::OptType::CanNotOptimize;
  bool is_optimize = configs[0].cmp_type != type;
  int vec_size = is_optimize ? VecSizeL : VecSizeM;
#else
  for (int i = 0; i < kArity; ++i) {
    // get the broadcast config,
    // if data shape is[m, n], then you should set data_dim = {n, m}
    // eg: out's shape [3, 45, 1]. then out_dims = {1, 45, 3}
    // if (ins[i]->numel() != (*outs)[0]->numel()) {
    if (ins[i]->numel()) {
      configs[i] = kps::details::BroadcastConfig(
          merge_dims.out_dims, merge_dims.in_dims[i], merge_dims.dim_size);
    }
  }
  int vec_size = GetVecsize<InT, OutT>(ins, outs);
#endif

  switch (vec_size) {
    case VecSizeL: {
      LaunchBroadcastKernel<InT, OutT, Functor, kArity, NumOuts, VecSizeL>(
          ctx, ins, outs, func, configs);
      break;
    }
    case VecSizeM: {
      LaunchBroadcastKernel<InT, OutT, Functor, kArity, NumOuts, VecSizeM>(
          ctx, ins, outs, func, configs);
      break;
    }
    case VecSizeS: {
      LaunchBroadcastKernel<InT, OutT, Functor, kArity, NumOuts, VecSizeS>(
          ctx, ins, outs, func, configs);
      break;
    }
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported vectorized size: %d!", vec_size));
      break;
    }
  }
}

template <ElementwiseType ET,
          typename InT,
          typename OutT,
          typename Functor,
          int NumOuts = 1>
void BroadcastKernel(const KPDevice &ctx,
                     const std::vector<const DenseTensor *> &ins,
                     std::vector<DenseTensor *> *outs,
                     int axis,
                     Functor func) {
  std::vector<int> dims_size;
  dims_size.reserve(ins.size());
  for (auto *in : ins) {
    dims_size.emplace_back(in->dims().size());
  }

  axis = axis == -1 ? *std::max_element(dims_size.begin(), dims_size.end()) -
                          *std::min_element(dims_size.begin(), dims_size.end())
                    : axis;
  BroadcastKernelForDifferentVecSize<ET, InT, OutT, Functor, NumOuts>(
      ctx, ins, outs, axis, func);
}

template <typename Functor, typename T, typename OutType = T>
void ElementwiseCompute(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        int axis,
                        Functor func,
                        DenseTensor *z) {
  std::vector<const DenseTensor *> ins = {&x, &y};
  std::vector<DenseTensor *> outs = {z};
  z->mutable_data<OutType>(dev_ctx.GetPlace());
  BroadcastKernel<ElementwiseType::kBinary, T, OutType, Functor, 1>(
      dev_ctx, ins, &outs, axis, func);
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
  funcs::ElementwiseCompute<Functor, T>(dev_ctx, x, y, axis, Functor(), z);
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
    funcs::ElementwiseCompute<Functor, T>(dev_ctx, x, y, axis, Functor(), z);
  } else {
    funcs::ElementwiseCompute<InverseFunctor, T>(
        dev_ctx, x, y, axis, InverseFunctor(), z);
  }
}

#endif

}  // namespace funcs
}  // namespace phi
