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

#include "paddle/pten/kernels/funcs/common_shape.h"
#include "paddle/pten/kernels/funcs/cuda_kernel_config.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

#ifdef __HIPCC__
constexpr int ELEMWISE_MAX_BLOCK_DIM = 256;
#else
constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
#endif
#define BLOCK_X 32
#define BLOCK_Y 32

#define GetDivMod(dividend, divisor, div, mod) \
  do {                                         \
    const auto dividend_copy = dividend;       \
    *div = dividend_copy / divisor;            \
    *mod = dividend_copy % divisor;            \
  } while (0)

namespace pten {
// FORWARD CODE
struct DimensionsTransform {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(
      bool &, std::vector<DimVector> &, DimVector &, int, int);
  int64_t dim_size;
  DimVector out_dims;
  std::vector<DimVector> in_dims;

 private:
  // To compensate the lackage of input_tensors` dimension with input variable
  // 'axis'
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
            PADDLE_THROW(paddle::platform::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "recieved %d.",
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
            PADDLE_THROW(paddle::platform::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "recieved %d.",
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

 public:
  explicit DimensionsTransform(const std::vector<const DenseTensor *> &ins,
                               const paddle::framework::DDim &dims,
                               int axis) {
    const int N = ins.size();
    dim_size = dims.size();
    out_dims = paddle::framework::vectorize<int64_t>(dims);
    in_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      in_dims[j] = paddle::framework::vectorize<int64_t>(ins[j]->dims());
    }
    InputDimensionsExtend(N, axis);

    auto merge_sequential_dims = [](bool &equal,
                                    std::vector<DimVector> &in_dims,
                                    DimVector &out,
                                    int i,
                                    int num) {
      for (int j = 1; j < num; ++j) {
        equal &= (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
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
    // To Merge the dimensions of input_tensors while the consequtive
    // equal-dimensions appears.
    MergeFunctor merge_ptr = merge_sequential_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);

    int min_idx = 0;
    int min_val = std::accumulate(
        in_dims[0].begin(), in_dims[0].end(), 1, std::multiplies<int64_t>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(
          in_dims[j].begin(), in_dims[j].end(), 1, std::multiplies<int64_t>());
      min_val = min_val > temp ? temp : min_val;
      min_idx = min_val == temp ? j : min_idx;
    }
    std::swap(in_dims[0], in_dims[min_idx]);

    // To Merge the dimension of input_tensors while the consequtive
    // 1-value-dimensions appears.
    merge_ptr = merge_sequential_one_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);
    std::swap(in_dims[min_idx], in_dims[0]);
  }
};

template <typename T, int VecSize, int Rank, bool IsBoundary = false>
__device__ __forceinline__ void LoadData(
    T *dst,
    const _ptr_ T *src,
    uint32_t block_offset,
    const kps::details::BroadcastConfig<Rank> &config,
    int numel,
    int num,
    int need_broadcast) {
  // numel : whole num of output
  // num: how many data will be deal with in this time
  if (need_broadcast) {
    kps::ReadDataBc<T, VecSize, 1, 1, Rank, IsBoundary>(
        dst, src, block_offset, config, numel);
  } else {
    kps::ReadData<T, VecSize, 1, 1, IsBoundary>(dst, src + block_offset, num);
  }
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          int Rank,
          bool IsBoundary = false>
__device__ void ElementwiseBroadcastKernelImpl(
    const paddle::framework::Array<const _ptr_ InT *__restrict__, Arity> &ins,
    paddle::framework::Array<_ptr_ OutT *, NumOuts> outs,
    const paddle::framework::Array<int, Arity> &use_broadcast,
    uint32_t numel,
    const paddle::framework::Array<kps::details::BroadcastConfig<Rank>, Arity>
        &configs,
    int num,
    int block_offset,
    Functor func) {
  InT args[Arity][VecSize];
  ConditionalT<OutT, NumOuts> result[VecSize];

#pragma unroll
  for (int i = 0; i < Arity; i++) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f));
    LoadData<InT, VecSize, Rank, IsBoundary>(args[i],
                                             ins[i],
                                             block_offset,
                                             configs[i],
                                             numel,
                                             num,
                                             use_broadcast[i]);
  }
  constexpr bool kCallElementwiseAny =
      paddle::platform::FunctionTraits<Functor>::has_pointer_args;
  pten::funcs::ElementwisePrimitiveCaller<InT,
                                          ConditionalT<OutT, NumOuts>,
                                          VecSize,
                                          Functor,
                                          Arity,
                                          kCallElementwiseAny>()(
      func, args, result);

  pten::funcs::ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, NumOuts>()(
      outs, result, block_offset, num);
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          int Rank>
__global__ void ElementwiseBroadcastKernel(
    paddle::framework::Array<const _ptr_ InT *__restrict__, Arity> ins,
    paddle::framework::Array<_ptr_ OutT *, NumOuts> outs,
    paddle::framework::Array<int, Arity> use_broadcast,
    uint32_t numel,
    paddle::framework::Array<kps::details::BroadcastConfig<Rank>, Arity>
        configs,
    int main_offset,
    int tail_tid,
    Functor func) {
  int block_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;

#ifdef PADDLE_WITH_XPU2
  for (; block_offset < main_offset; block_offset += stride) {
    ElementwiseBroadcastKernelImpl<InT,
                                   OutT,
                                   Functor,
                                   Arity,
                                   NumOuts,
                                   VecSize,
                                   Rank,
                                   false>(ins,
                                          outs,
                                          use_broadcast,
                                          numel,
                                          configs,
                                          BLOCK_NUM_X * VecSize,
                                          block_offset,
                                          func);
  }
  int num = numel - block_offset;
  if (num > 0) {
    ElementwiseBroadcastKernelImpl<InT,
                                   OutT,
                                   Functor,
                                   Arity,
                                   NumOuts,
                                   VecSize,
                                   Rank,
                                   true>(
        ins, outs, use_broadcast, numel, configs, num, block_offset, func);
  }
#else
  if (block_offset < main_offset) {
    ElementwiseBroadcastKernelImpl<InT,
                                   OutT,
                                   Functor,
                                   Arity,
                                   NumOuts,
                                   VecSize,
                                   Rank,
                                   false>(ins,
                                          outs,
                                          use_broadcast,
                                          numel,
                                          configs,
                                          BLOCK_NUM_X * VecSize,
                                          block_offset,
                                          func);
  } else {
    ElementwiseBroadcastKernelImpl<InT,
                                   OutT,
                                   Functor,
                                   Arity,
                                   NumOuts,
                                   VecSize,
                                   Rank,
                                   true>(
        ins, outs, use_broadcast, numel, configs, tail_tid, block_offset, func);
  }
#endif
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          int Rank>
void LaunchKernel(const KPDevice &ctx,
                  const std::vector<const DenseTensor *> &ins,
                  std::vector<DenseTensor *> *outs,
                  Functor func,
                  DimensionsTransform merge_dims) {
  int numel = (*outs)[0]->numel();
  paddle::framework::Array<kps::details::BroadcastConfig<Rank>, Arity> configs;
  paddle::framework::Array<int, Arity> use_broadcast;
  paddle::framework::Array<const _ptr_ InT *__restrict__, Arity> ins_data;
  paddle::framework::Array<_ptr_ OutT *, NumOuts> outs_data;

  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (*outs)[i]->mutable_data<OutT>();
  }

  for (int i = 0; i < Arity; i++) {
    use_broadcast[i] = (ins[i]->numel() != numel);
    ins_data[i] = (_ptr_ InT *)(ins[i]->data<InT>());
    if (use_broadcast[i]) {
      // get the broadcast config,
      // if data shape is[m, n], then you should set data_dim = {n, m}
      // eg: out's shape [3, 45, 1]. then out_dims = {1, 45, 3}
      configs[i] = kps::details::BroadcastConfig<Rank>(
          merge_dims.out_dims, merge_dims.in_dims[i], merge_dims.dim_size);
    }
  }

#ifdef PADDLE_WITH_XPU2
  const int threads = 64;
  const int blocks = 8;
  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);
  auto stream = ctx.x_context()->xpu_stream;
  ElementwiseBroadcastKernel<InT,
                             OutT,
                             Functor,
                             Arity,
                             NumOuts,
                             VecSize,
                             Rank><<<blocks, threads, stream>>>(ins_data,
                                                                outs_data,
                                                                use_broadcast,
                                                                numel,
                                                                configs,
                                                                main_offset,
                                                                tail_tid,
                                                                func);
#else
  const int threads = 256;
  int blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;
  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);
  auto stream = ctx.stream();
  ElementwiseBroadcastKernel<InT,
                             OutT,
                             Functor,
                             Arity,
                             NumOuts,
                             VecSize,
                             Rank><<<blocks, threads, 0, stream>>>(
      ins_data,
      outs_data,
      use_broadcast,
      numel,
      configs,
      main_offset,
      tail_tid,
      func);
#endif
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
void LaunchBroadcastKernelForDifferentVecSize(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    int axis,
    Functor func) {
  const auto merge_dims = DimensionsTransform(ins, (*outs)[0]->dims(), axis);

#define CALL_BROADCAST_FOR_DIM_SIZE(rank)                            \
  case rank: {                                                       \
    LaunchKernel<InT, OutT, Functor, Arity, NumOuts, VecSize, rank>( \
        ctx, ins, outs, func, merge_dims);                           \
  } break;

  switch (merge_dims.dim_size) {
    CALL_BROADCAST_FOR_DIM_SIZE(1);
    CALL_BROADCAST_FOR_DIM_SIZE(2);
    CALL_BROADCAST_FOR_DIM_SIZE(3);
    CALL_BROADCAST_FOR_DIM_SIZE(4);
    CALL_BROADCAST_FOR_DIM_SIZE(5);
    CALL_BROADCAST_FOR_DIM_SIZE(6);
    CALL_BROADCAST_FOR_DIM_SIZE(7);
    CALL_BROADCAST_FOR_DIM_SIZE(8);
    default: {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "The maximum dimension of input tensor is expected to be less than "
          "%d, but recieved %d.\n",
          merge_dims.dim_size,
          paddle::framework::DDim::kMaxRank));
    }
  }
#undef CALL_BROADCAST_FOR_DIM_SIZE
}

template <ElementwiseType ET,
          typename InT,
          typename OutT,
          typename Functor,
          int NumOuts = 1>
void LaunchBroadcastElementwiseCudaKernel(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    int axis,
    Functor func) {
  using Traits = paddle::platform::FunctionTraits<Functor>;
  const int kArity =
      Traits::has_pointer_args ? static_cast<int>(ET) : Traits::arity;
  PADDLE_ENFORCE_EQ(ins.size(),
                    kArity,
                    paddle::platform::errors::InvalidArgument(
                        "The number of inputs is expected to be equal to the "
                        "arity of functor. But recieved: the number of inputs "
                        "is %d, the arity of functor is %d.",
                        ins.size(),
                        kArity));
  PADDLE_ENFORCE_LE(kArity,
                    3,
                    paddle::platform::errors::InvalidArgument(
                        "Currently only broadcast of ternary is supported "
                        "and verified, but received %d.",
                        kArity));
  PADDLE_ENFORCE_EQ(outs->size(),
                    NumOuts,
                    paddle::platform::errors::InvalidArgument(
                        "Number of outputs shall equal to number of functions, "
                        "but number of outputs is %d, of functions is %d.",
                        outs->size(),
                        NumOuts));
  int in_vec_size = 4;
  int out_vec_size = 4;
  if (NumOuts > 1) {
    for (int i = 0; i < NumOuts; ++i) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          paddle::platform::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, but "
              "%dth output tensor`s shape is not.",
              i));
      out_vec_size = std::min(
          paddle::platform::GetVectorizedSize<OutT>((*outs)[i]->data<OutT>()),
          out_vec_size);
    }
  } else {
    out_vec_size =
        paddle::platform::GetVectorizedSize<OutT>((*outs)[0]->data<OutT>());
  }

  for (auto *in : ins) {
    auto temp_size = paddle::platform::GetVectorizedSize<InT>(in->data<InT>());
    in_vec_size = in->dims() == (*outs)[0]->dims()
                      ? std::min(temp_size, in_vec_size)
                      : in_vec_size;
  }
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 4: {
      LaunchBroadcastKernelForDifferentVecSize<InT,
                                               OutT,
                                               Functor,
                                               kArity,
                                               NumOuts,
                                               4>(ctx, ins, outs, axis, func);
      break;
    }
    case 2: {
      LaunchBroadcastKernelForDifferentVecSize<InT,
                                               OutT,
                                               Functor,
                                               kArity,
                                               NumOuts,
                                               2>(ctx, ins, outs, axis, func);
      break;
    }
    case 1: {
      LaunchBroadcastKernelForDifferentVecSize<InT,
                                               OutT,
                                               Functor,
                                               kArity,
                                               NumOuts,
                                               1>(ctx, ins, outs, axis, func);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <ElementwiseType ET,
          typename InT,
          typename OutT,
          typename Functor,
          int NumOuts = 1>
void LaunchElementwiseCudaKernel(const KPDevice &ctx,
                                 const std::vector<const DenseTensor *> &ins,
                                 std::vector<DenseTensor *> *outs,
                                 int axis,
                                 Functor func) {
  std::vector<int> dims_size;
  bool no_broadcast_flag = true;
  for (auto *in : ins) {
    no_broadcast_flag &= ins[0]->dims() == in->dims();
    dims_size.emplace_back(in->dims().size());
  }
  if (no_broadcast_flag) {
    pten::funcs::
        LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT, Functor, NumOuts>(
            ctx, ins, outs, func);
  } else {
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    pten::LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT, Functor, NumOuts>(
        ctx, ins, outs, axis, func);
  }
}

// BACKWARD CODE

// Suppose only has contiguous dims
static inline bool CheckContiguousDims(const std::vector<int> &broadcast_pos) {
  for (int i = 1; i < broadcast_pos.size(); ++i) {
    if (broadcast_pos[i] != broadcast_pos[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

inline void ComputeBroadcastTranspositionArray(const int *x_one_indexs,
                                               int *x_trans_indexs,
                                               const int max_dim,
                                               const int x_one_size) {
  int diff = max_dim - x_one_size;
  std::copy_n(x_one_indexs, x_one_size, x_trans_indexs + diff);
  int p = 0;
  int q = diff;
  for (int i = 0; i < max_dim; ++i) {
    if (q < max_dim && i == x_trans_indexs[q]) {
      ++q;
    } else {
      x_trans_indexs[p++] = i;
    }
  }
}

// Check input can be split into 2 parts
static inline bool SplitDims(const std::vector<int> &y_broadcast_pos,
                             int max_dim) {
  bool can_split_dim2 = true;
  // must at start or end.
  if (y_broadcast_pos[0] != 0 &&
      y_broadcast_pos[y_broadcast_pos.size() - 1] != max_dim - 1) {
    can_split_dim2 = false;
  } else {
    for (int i = 1; i < y_broadcast_pos.size(); ++i) {
      // dim must be continue
      if (y_broadcast_pos[i] != y_broadcast_pos[i - 1] + 1) {
        can_split_dim2 = false;
        break;
      }
    }
  }
  return can_split_dim2;
}

inline void ComputeBroadcastKernelSize(int *x_dims_array,
                                       int *out_dims_array,
                                       int *x_blocks,
                                       int *x_threads,
                                       int max_dim) {
  *x_blocks = 1;
  *x_threads = 1;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] == out_dims_array[i]) {
      *x_blocks *= x_dims_array[i];
    } else {
      *x_threads *= out_dims_array[i];
    }
  }
}

template <typename T, typename OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastOneCUDAKernel(const T *x,
                                                            const T *y,
                                                            const Tout *out,
                                                            const Tout *dout,
                                                            int pre,
                                                            int n,
                                                            int post,
                                                            int y_pre,
                                                            int y_n,
                                                            int y_post,
                                                            bool is_xsize,
                                                            OP op,
                                                            T *dd) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  T val(0);
  if (is_xsize) {
    // do reduce for x
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      // Get y pre rows id with x post and y_pre.
      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int y_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  } else {
    // do reduce for y
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int x_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  }
}

template <typename T, typename DY_OP, typename DX_OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastAllCUDAKernel(
    const T *x,
    const T *y,
    const Tout *out,
    const Tout *dout,
    int pre,
    int n,
    int post,
    bool is_xsize_larger,
    DX_OP dx_op,
    DY_OP dy_op,
    T *dx,
    T *dy) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  T val(0);
  if (is_xsize_larger) {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + i * post + b_j;
      int y_offset = b_i * post + b_j;
      if (dx) {
        dx[x_offset] =
            dx_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
    }
    if (dy) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dy[bid] = val;
      }
    }
  } else {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + i * post + b_j;
      int x_offset = b_i * post + b_j;
      if (dy) {
        dy[y_offset] =
            dy_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
    }
    if (dx) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dx[bid] = val;
      }
    }
  }
}

template <typename T, typename DY_OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastCUDAKernelHeight(const T *x,
                                                               const T *y,
                                                               const Tout *out,
                                                               const Tout *dout,
                                                               int h,
                                                               int w,
                                                               DY_OP dy_op,
                                                               T *dy,
                                                               int x_h,
                                                               int x_w,
                                                               bool is_y) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = gridDim.x * blockDim.x;
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_y) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int x_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[out_offset], dout[out_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dy[m] = sdata[0][threadIdx.x];
        }
      }
    }
  } else {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int y_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[m], y[y_offset], out[out_offset], dout[out_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dy[m] = sdata[0][threadIdx.x];
        }
      }
    }
  }
}

template <typename T, typename DY_OP, typename Tout = T>
static __global__ void CommonGradBroadcast1CUDAKernelHeight(const T *x,
                                                            const T *y,
                                                            const Tout *out,
                                                            const Tout *dout,
                                                            int h,
                                                            int w,
                                                            DY_OP dy_op,
                                                            T *dy,
                                                            int x_h,
                                                            int x_w,
                                                            bool is_y) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val(0);

  if (is_y) {
    do {
      int out_offset = i * w + j;
      int x_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    do {
      int out_offset = i * w + j;
      int y_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[j], y[y_offset], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void ElemwiseGradBroadcast1CUDAKernel(const T *x,
                                                        const T *y,
                                                        const Tout *out,
                                                        const Tout *dout,
                                                        int h,
                                                        int w,
                                                        bool is_xsize_larger,
                                                        DX_OP dx_op,
                                                        DY_OP dy_op,
                                                        T *dx,
                                                        T *dy) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val(0);
  if (is_xsize_larger) {
    do {
      int x_offset = i * w + j;
      if (dx) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    do {
      int y_offset = i * w + j;
      if (dy) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dx) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

// suppose use 2D block is fast because more parallel
// and memory coalesced
template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void FastElemwiseGradBroadcast1CUDAKernel(
    const T *x,
    const T *y,
    const Tout *out,
    const Tout *dout,
    int h,
    int w,
    bool is_xsize_larger,
    DX_OP dx_op,
    DY_OP dy_op,
    T *dx,
    T *dy) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = gridDim.x * blockDim.x;
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_xsize_larger) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int x_offset = n * w + m;
        if (dx && m < w && n < h) {
          dx[x_offset] =
              dx_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
        }
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dy[m] = sdata[0][threadIdx.x];
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int y_offset = n * w + m;
        if (dy && m < w && n < h) {
          dy[y_offset] =
              dy_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx) {
          if (m < w && n < h) {
            T val = dx_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dx) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dx[m] = sdata[0][threadIdx.x];
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(const T *x,
                                                        const T *y,
                                                        const Tout *out,
                                                        const Tout *dout,
                                                        int pre,
                                                        int n,
                                                        int post,
                                                        bool is_xsize_larger,
                                                        DX_OP dx_op,
                                                        DY_OP dy_op,
                                                        T *dx,
                                                        T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast1CUDA(gpuStream_t stream,
                                       const T *x,
                                       const T *y,
                                       const Tout *out,
                                       const Tout *dout,
                                       int h,
                                       int w,
                                       bool is_xsize_larger,
                                       DX_OP dx_op,
                                       DY_OP dy_op,
                                       T *dx,
                                       T *dy) {
  // For small case use 1D block
  constexpr int half_walf = 16;
  if (w < half_walf || h < half_walf) {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
    int gird_size = w;
    ElemwiseGradBroadcast1CUDAKernel<<<gird_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else {
    // suppose perfoemance improves with h increased.
    dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
    int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
    FastElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast2CUDA(gpuStream_t stream,
                                       const T *x,
                                       const T *y,
                                       const Tout *out,
                                       const Tout *dout,
                                       int pre,
                                       int n,
                                       int post,
                                       bool is_xsize_larger,
                                       DX_OP dx_op,
                                       DY_OP dy_op,
                                       T *dx,
                                       T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
}

template <typename T, typename DX_OP, typename Tout = T>
__global__ void CommonGradBroadcastCUDAKernel(const int *x_strides_array,
                                              const int *y_strides_array,
                                              const int *out_dims_array,
                                              const int *y_strides_order,
                                              const int *y_dims_order,
                                              const T *x,
                                              const T *y,
                                              const Tout *out,
                                              const Tout *dout,
                                              T *dx,
                                              int out_size,
                                              int max_dim,
                                              int thread_num,
                                              DX_OP dx_op) {
  T val(0);
  int i = blockIdx.x;
  int tid = threadIdx.x;
  for (int j = tid; j < thread_num; j += blockDim.x) {
    const int X_index = i * thread_num + j;
    int out_index = X_index;
    int C_index = 0;
    int B_index = i * thread_num + j;
    int remainder = 0;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(B_index, y_dims_order[d], &B_index, &remainder);
      C_index += remainder * y_strides_order[d];
    }
    int x_index = 0;
    int y_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(C_index_val, out_dims_array[d], &C_index_val, &remainder);
      x_index += remainder * x_strides_array[d];
      y_index += remainder * y_strides_array[d];
    }
    out_index = C_index;
    val += dx_op(x[x_index], y[y_index], out[out_index], dout[out_index]);
  }
  val = paddle::platform::reduceSum(val, tid, thread_num);
  if (threadIdx.x == 0) {
    dx[i] = val;
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonGradBroadcastCUDA(const DenseTensor &x,
                             const DenseTensor &y,
                             const DenseTensor &out,
                             const DenseTensor &dout,
                             DenseTensor *dx,
                             DenseTensor *dy,
                             int *x_dims_array,
                             int *y_dims_array,
                             int *out_dims_array,
                             int max_dim,
                             const GPUContext &ctx,
                             DX_OP dx_op,
                             DY_OP dy_op) {
  const auto gplace = ctx.GetPlace();
  auto cplace = paddle::platform::CPUPlace();
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const Tout *out_data = out.data<Tout>();
  const Tout *dout_data = dout.data<Tout>();
  T *dx_data = dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace());
  T *dy_data = dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace());

  std::vector<int> x_one_indexs;
  std::vector<int> y_one_indexs;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] != y_dims_array[i]) {
      if (x_dims_array[i] == 1) {
        x_one_indexs.push_back(i);
      }
      if (y_dims_array[i] == 1) {
        y_one_indexs.push_back(i);
      }
    }
  }

  std::vector<int> x_trans_indexs(max_dim);
  std::vector<int> y_trans_indexs(max_dim);
  ComputeBroadcastTranspositionArray(
      x_one_indexs.data(), x_trans_indexs.data(), max_dim, x_one_indexs.size());
  ComputeBroadcastTranspositionArray(
      y_one_indexs.data(), y_trans_indexs.data(), max_dim, y_one_indexs.size());

  // compute array stride for cuda kernel;
  // e.g. x.dims=[2,3,4], x_stride=[12,4,1]
  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  std::vector<int> out_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  int z_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    out_strides_array[i] = z_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
    z_stride *= out_dims_array[i];
  }

  std::vector<int> x_strides_order(max_dim);
  std::vector<int> y_strides_order(max_dim);
  std::vector<int> x_dims_order(max_dim);
  std::vector<int> y_dims_order(max_dim);
  for (int i = 0; i < max_dim; ++i) {
    x_strides_order[i] = out_strides_array[x_trans_indexs[i]];
    y_strides_order[i] = out_strides_array[y_trans_indexs[i]];
    x_dims_order[i] = out_dims_array[x_trans_indexs[i]];
    y_dims_order[i] = out_dims_array[y_trans_indexs[i]];
  }
  std::vector<int> x_broadcast_pos;
  std::vector<int> y_broadcast_pos;

  int bytes = max_dim * sizeof(int);

  for (int i = 0; i < max_dim; ++i) {
    if (x_dims_array[i] != out_dims_array[i] && x_dims_array[i] == 1) {
      x_broadcast_pos.emplace_back(i);
    }
    if (y_dims_array[i] != out_dims_array[i] && y_dims_array[i] == 1) {
      y_broadcast_pos.emplace_back(i);
    }
  }

  auto stream = ctx.stream();
  bool can_split_x = false;
  bool can_split_y = false;

  auto FastCommonCUDAF = [&](const std::vector<int> &broadcast_pos, bool is_y) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastCommonCUDAF elementwise w:" << w << " h:" << h
            << " is_y:" << is_y;

    int split_h;
    int split_w;
    int kh = h;
    int kw = w;

    if (is_y) {
      split_h = std::accumulate(x_dims_array,
                                x_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(x_dims_array + broadcast_pos.size(),
                                x_dims_array + max_dim,
                                1,
                                std::multiplies<int>());

    } else {
      split_h = std::accumulate(y_dims_array,
                                y_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(y_dims_array + broadcast_pos.size(),
                                y_dims_array + max_dim,
                                1,
                                std::multiplies<int>());
    }

    if (h > split_h) kh = split_h;
    if (w > split_w) kw = split_w;

    if (is_y) {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                               block_size,
                                               0,
                                               stream>>>(x_data,
                                                         y_data,
                                                         out_data,
                                                         dout_data,
                                                         h,
                                                         w,
                                                         dy_op,
                                                         dy_data,
                                                         kh,
                                                         kw,
                                                         is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            h,
                                                            w,
                                                            dy_op,
                                                            dy_data,
                                                            kh,
                                                            kw,
                                                            is_y);
      }
    } else {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                               block_size,
                                               0,
                                               stream>>>(x_data,
                                                         y_data,
                                                         out_data,
                                                         dout_data,
                                                         h,
                                                         w,
                                                         dx_op,
                                                         dx_data,
                                                         kh,
                                                         kw,
                                                         is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            h,
                                                            w,
                                                            dx_op,
                                                            dx_data,
                                                            kh,
                                                            kw,
                                                            is_y);
      }
    }
  };

  auto FastBroadCastHeightCUDAF = [&](const std::vector<int> &broadcast_pos,
                                      bool x_large) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastBroadCastHeightCUDAF w:" << w << " h:" << h;

    if (w < 16 || h < 16) {
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
      int grid_size = w;
      ElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
          x_data,
          y_data,
          out_data,
          dout_data,
          h,
          w,
          x_large,
          dx_op,
          dy_op,
          dx_data,
          dy_data);
    } else {
      dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
      int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
      FastElemwiseGradBroadcast1CUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       h,
                                                       w,
                                                       x_large,
                                                       dx_op,
                                                       dy_op,
                                                       dx_data,
                                                       dy_data);
    }
  };

  auto FastBroadCastAllCUDAF = [&](
      const std::vector<int> &broadcast_pos, int max_dim, bool is_x_large) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = 1;
    int post = 1;

    if (broadcast_pos.size() == 1) {
      mid = out_dims_array[axis];
      post = std::accumulate(out_dims_array + axis + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    } else {
      mid = std::accumulate(out_dims_array + axis,
                            out_dims_array + broadcast_pos.back() + 1,
                            1,
                            std::multiplies<int>());
      post = std::accumulate(out_dims_array + broadcast_pos.back() + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    }

    VLOG(3) << "FastBroadCastAllCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;

    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
    int grid_size = pre * post;

    FastCommonGradBroadcastAllCUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x_data,
        y_data,
        out_data,
        dout_data,
        pre,
        mid,
        post,
        is_x_large,
        dx_op,
        dy_op,
        dx_data,
        dy_data);
  };

  auto FastBroadCastOneCUDAF = [&](
      const std::vector<int> &broadcast_pos, int max_dim, bool is_x) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = out_dims_array[axis];
    int post = std::accumulate(out_dims_array + axis + 1,
                               out_dims_array + max_dim,
                               1,
                               std::multiplies<int>());

    int k_pre;
    int k_mid;
    int k_post;

    if (is_x) {
      k_pre = std::accumulate(
          y_dims_array, y_dims_array + axis, 1, std::multiplies<int>());
      k_mid = y_dims_array[axis];
      k_post = std::accumulate(y_dims_array + axis + 1,
                               y_dims_array + max_dim,
                               1,
                               std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      // we need to calc y offset with blockid, so do x_pre/y_pre to get left
      // size.
      if (k_pre != pre) k_pre = pre / k_pre;

      FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       pre,
                                                       mid,
                                                       post,
                                                       k_pre,
                                                       k_mid,
                                                       k_post,
                                                       true,
                                                       dx_op,
                                                       dx_data);
    } else {
      k_pre = std::accumulate(
          x_dims_array, x_dims_array + axis, 1, std::multiplies<int>());
      k_mid = x_dims_array[axis];
      k_post = std::accumulate(x_dims_array + axis + 1,
                               x_dims_array + max_dim,
                               1,
                               std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      if (k_pre != pre) k_pre = pre / k_pre;

      FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       pre,
                                                       mid,
                                                       post,
                                                       k_pre,
                                                       k_mid,
                                                       k_post,
                                                       false,
                                                       dy_op,
                                                       dy_data);
    }
    VLOG(3) << "FastBroadCastOneCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;
  };

  // do fast elementwise if: 1. only one input need to do broadcast, we can
  // fallback
  // to old fast path.
  // 2. if both x and y need broadcast, then do it one by one.
  bool fast_broadcast = false;
  if (x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    if (can_split_y) {
      // only y need to do broadcast on h
      if (y_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(y_broadcast_pos, true);
        fast_broadcast = true;
      }
    } else if (y_broadcast_pos.size() == 1 ||
               CheckContiguousDims(y_broadcast_pos)) {  // for only one dim and
                                                        // contiguous broadcast.
      // If cannot split,  which means input has 3 parts
      FastBroadCastAllCUDAF(y_broadcast_pos, max_dim, true);
      fast_broadcast = true;
    }
  } else if (y_broadcast_pos.empty() && !x_broadcast_pos.empty()) {
    // only x need broadcast
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(x_broadcast_pos, false);
        fast_broadcast = true;
      }
    } else if (x_broadcast_pos.size() == 1 ||
               CheckContiguousDims(x_broadcast_pos)) {
      FastBroadCastAllCUDAF(x_broadcast_pos, max_dim, false);
      fast_broadcast = true;
    }
  } else if (!x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    // do x and y broadcast each.
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    bool fast_broadcast_x = false;
    bool fast_broadcast_y = false;
    if (can_split_y) {
      // begin at start.
      if (y_broadcast_pos[0] == 0) {
        FastCommonCUDAF(y_broadcast_pos, true);
        fast_broadcast_y = true;
      }
    } else if (y_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(y_broadcast_pos, max_dim, false);
      can_split_y = true;
      fast_broadcast_y = true;
    }
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastCommonCUDAF(x_broadcast_pos, false);
        fast_broadcast_x = true;
      }
    } else if (x_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(x_broadcast_pos, max_dim, true);
      can_split_x = true;
      fast_broadcast_x = true;
    }
    VLOG(3) << "CommonBroadcast can_split_y:" << can_split_y
            << " can_split_x:" << can_split_x;
    // if both x and y into fast path then return
    if (fast_broadcast_x && fast_broadcast_y) {
      fast_broadcast = true;
    }
    if (can_split_y && can_split_x && fast_broadcast) return;
  }

  // Should remove memory copy, use reg instead.
  if (fast_broadcast) {
    return;
  }
  int x_blocks = 0;
  int x_threads = 0;
  ComputeBroadcastKernelSize(
      x_dims_array, out_dims_array, &x_blocks, &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  ComputeBroadcastKernelSize(
      y_dims_array, out_dims_array, &y_blocks, &y_threads, max_dim);

  auto x_strides_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int *x_strides_array_gpu =
      reinterpret_cast<int *>(x_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       x_strides_array_gpu,
                       cplace,
                       x_strides_array.data(),
                       bytes,
                       ctx.stream());

  auto y_strides_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int *y_strides_array_gpu =
      reinterpret_cast<int *>(y_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       y_strides_array_gpu,
                       cplace,
                       y_strides_array.data(),
                       bytes,
                       ctx.stream());

  auto out_dims_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int *out_dims_array_gpu = reinterpret_cast<int *>(out_dims_array_tmp->ptr());
  paddle::memory::Copy(
      gplace, out_dims_array_gpu, cplace, out_dims_array, bytes, ctx.stream());

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, x_threads);
  int y_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, y_threads);
  if (dx) {
    auto x_strides_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *x_strides_order_gpu =
        reinterpret_cast<int *>(x_strides_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         x_strides_order_gpu,
                         cplace,
                         x_strides_order.data(),
                         bytes,
                         ctx.stream());

    auto x_dims_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *x_dims_order_gpu = reinterpret_cast<int *>(x_dims_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         x_dims_order_gpu,
                         cplace,
                         x_dims_order.data(),
                         bytes,
                         ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T,
        DX_OP,
        Tout><<<x_blocks, x_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                           y_strides_array_gpu,
                                                           out_dims_array_gpu,
                                                           x_strides_order_gpu,
                                                           x_dims_order_gpu,
                                                           x_data,
                                                           y_data,
                                                           out_data,
                                                           dout_data,
                                                           dx_data,
                                                           out_size,
                                                           max_dim,
                                                           x_threads,
                                                           dx_op);
  }
  if (dy) {
    auto y_strides_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *y_strides_order_gpu =
        reinterpret_cast<int *>(y_strides_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         y_strides_order_gpu,
                         cplace,
                         y_strides_order.data(),
                         bytes,
                         ctx.stream());

    auto y_dims_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *y_dims_order_gpu = reinterpret_cast<int *>(y_dims_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         y_dims_order_gpu,
                         cplace,
                         y_dims_order.data(),
                         bytes,
                         ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T,
        DY_OP,
        Tout><<<y_blocks, y_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                           y_strides_array_gpu,
                                                           out_dims_array_gpu,
                                                           y_strides_order_gpu,
                                                           y_dims_order_gpu,
                                                           x_data,
                                                           y_data,
                                                           out_data,
                                                           dout_data,
                                                           dy_data,
                                                           out_size,
                                                           max_dim,
                                                           y_threads,
                                                           dy_op);
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonElementwiseBroadcastBackward(const GPUContext &ctx,
                                        const DDim &x_dims,
                                        const DDim &y_dims,
                                        const DenseTensor &x,
                                        const DenseTensor &y,
                                        const DenseTensor &out,
                                        const DenseTensor &dout,
                                        int axis,
                                        DenseTensor *dx,
                                        DenseTensor *dy,
                                        DX_OP dx_op,
                                        DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  funcs::GetBroadcastDimsArrays(x_dims,
                                y_dims,
                                x_dims_array.data(),
                                y_dims_array.data(),
                                out_dims_array.data(),
                                max_dim,
                                axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << paddle::framework::make_ddim(x_dims_array)
          << " ydim:" << paddle::framework::make_ddim(y_dims_array);

  CommonGradBroadcastCUDA<T, DX_OP, DY_OP, Tout>(x,
                                                 y,
                                                 out,
                                                 dout,
                                                 dx,
                                                 dy,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 ctx,
                                                 dx_op,
                                                 dy_op);
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void ElemwiseGradComputeWithBroadcast(const GPUContext &ctx,
                                      const DDim &x_dims,
                                      const DDim &y_dims,
                                      const DenseTensor &x,
                                      const DenseTensor &y,
                                      const DenseTensor &out,
                                      const DenseTensor &dout,
                                      int axis,
                                      DenseTensor *dx,
                                      DenseTensor *dy,
                                      DX_OP dx_op,
                                      DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      paddle::platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    paddle::platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = funcs::trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    funcs::get_mid_dims(x_dims,
                        y_dims_trimed,
                        axis_trim,
                        &pre,
                        &n,
                        &post,
                        &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = funcs::trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    funcs::get_mid_dims(y_dims,
                        x_dims_trimed,
                        axis_trim,
                        &pre,
                        &n,
                        &post,
                        &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<T, DX_OP, DY_OP, Tout>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    ElemwiseGradBroadcast1CUDA(
        ctx.stream(),
        x.data<T>(),
        y.data<T>(),
        out.data<Tout>(),
        dout.data<Tout>(),
        pre,
        n,
        is_xsize_larger,
        dx_op,
        dy_op,
        dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
        dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
  } else {
    ElemwiseGradBroadcast2CUDA(
        ctx.stream(),
        x.data<T>(),
        y.data<T>(),
        out.data<Tout>(),
        dout.data<Tout>(),
        pre,
        n,
        post,
        is_xsize_larger,
        dx_op,
        dy_op,
        dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
        dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
  }
}

}  // namespace pten
