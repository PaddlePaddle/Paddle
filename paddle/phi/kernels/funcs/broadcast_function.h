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

template <typename T, int VecSize, bool IsBoundary = false>
__device__ __forceinline__ void LoadData(
    T *dst,
    const _ptr_ T *src,
    uint32_t block_offset,
    const kps::details::BroadcastConfig &config,
    int numel,
    int num,
    int need_broadcast,
    int read_lens) {
  // numel : whole num of output
  // num: how many data will be deal with in this time
  if (need_broadcast) {
    kps::ReadDataBc<T, VecSize, 1, IsBoundary>(
        dst, src, block_offset, config, numel, read_lens);
  } else {
    kps::ReadData<T, VecSize, 1, IsBoundary>(
        dst, src + block_offset, num, read_lens);
  }
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary = false>
__device__ void VectorizedBroadcastKernelImpl(
    const phi::Array<const _ptr_ InT *__restrict__, Arity> &ins,
    phi::Array<_ptr_ OutT *, NumOuts> outs,
    const phi::Array<int, Arity> &use_broadcast,
    uint32_t numel,
    const phi::Array<kps::details::BroadcastConfig, Arity> &configs,
    int num,
    int block_offset,
    int read_lens,
    Functor func) {
  __simd__ InT args[Arity][VecSize];
  __simd__ ConditionalT<OutT, NumOuts> result[VecSize];

#pragma unroll
  for (int i = 0; i < Arity; ++i) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f), read_lens);
    LoadData<InT, VecSize, IsBoundary>(args[i],
                                       ins[i],
                                       block_offset,
                                       configs[i],
                                       numel,
                                       num,
                                       use_broadcast[i],
                                       read_lens);
  }
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

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
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
  int block_offset = BLOCK_ID_X * BLOCK_NUM_X * read_lens;
  int stride = BLOCK_NUM_X * GRID_NUM_X * read_lens;

#ifdef PADDLE_WITH_XPU_KP
  for (; block_offset < main_offset; block_offset += stride) {
    VectorizedBroadcastKernelImpl<InT,
                                  OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  false>(ins,
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
                                  true>(ins,
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
  if (block_offset < main_offset) {
    VectorizedBroadcastKernelImpl<InT,
                                  OutT,
                                  Functor,
                                  Arity,
                                  NumOuts,
                                  VecSize,
                                  false>(ins,
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
                                  true>(ins,
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
  int numel = (*outs)[0]->numel();
  phi::Array<int, Arity> use_broadcast;
  phi::Array<const _ptr_ InT *__restrict__, Arity> ins_data;
  phi::Array<_ptr_ OutT *, NumOuts> outs_data;

  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (_ptr_ OutT *)(ctx.Alloc<OutT>((*outs)[i]));
  }

  for (int i = 0; i < Arity; ++i) {
    use_broadcast[i] = (ins[i]->numel() != numel);
    ins_data[i] = (const _ptr_ InT *)(ins[i]->data<InT>());
  }

#ifdef PADDLE_WITH_XPU_KP
  const int threads = 64;
  const int blocks = 8;
  int read_lens = configs[0].buf_len;
  auto stream = ctx.x_context()->xpu_stream;
  int main_offset = (numel / (read_lens * threads)) * read_lens * threads;
  int tail_tid = numel % (read_lens * threads);
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
#endif
  VectorizedBroadcastKernel<InT, OutT, Functor, Arity, NumOuts, VecSize>
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
