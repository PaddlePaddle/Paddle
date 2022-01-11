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

#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/function_traits.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/cuda_kernel_config.h"

#ifdef __HIPCC__
constexpr int ELEMWISE_MAX_BLOCK_DIM = 256;
#else
constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
#endif
#define BLOCK_X 32
#define BLOCK_Y 32

namespace pten {

namespace kps = paddle::operators::kernel_primitives;
enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3, kAny = -1 };

/* Packing scalar type T(float, int etc.) into Array<T, NumOuts> type
   for supporting multiple-output feature in elementwise system.*/
template <class T, int Num>
using ConditionalT =
    typename std::conditional_t<Num == 1, T, paddle::framework::Array<T, Num>>;

// FORWARD CODE
template <typename InT,
          typename OutT,
          int VecSize,
          typename Functor,
          int Arity,
          bool CallElementwiseAny = false>
struct ElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result);
};

template <typename InT, typename OutT, int VecSize, typename Functor, int Arity>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity, true> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseAny<InT, OutT, VecSize, 1, 1, Arity, Functor>(
        result, args, func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 1, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 2, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 3, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], args[2], func);
  }
};

template <typename OutT, int VecSize, bool IsBoundary, int NumOuts>
struct ElementwiseWriteDataCaller {
  __device__ __forceinline__ void operator()(
      paddle::framework::Array<OutT *, NumOuts> outs,
      ConditionalT<OutT, NumOuts> src[VecSize],
      int block_offset,
      int num) {
    OutT dst[NumOuts][VecSize];
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
#pragma unroll
      for (int j = 0; j < NumOuts; ++j) {
        dst[j][i] = (src[i])[j];
      }
    }
#pragma unroll
    for (int i = 0; i < NumOuts; ++i) {
      kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(
          outs[i] + block_offset, dst[i], num);
    }
  }
};

template <typename OutT, int VecSize, bool IsBoundary>
struct ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, 1> {
  __device__ __forceinline__ void operator()(
      paddle::framework::Array<OutT *, 1> outs,
      OutT src[VecSize],
      int block_offset,
      int num) {
    kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(
        outs[0] + block_offset, src, num);
  }
};

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary>
__device__ void VectorizedElementwiseKernelImpl(
    const paddle::framework::Array<const InT *__restrict__, Arity> &in,
    paddle::framework::Array<OutT *, NumOuts> outs,
    int num,
    int data_offset,
    Functor func) {
  InT args[Arity][VecSize];
  ConditionalT<OutT, NumOuts> result[VecSize];

#pragma unroll
  for (int i = 0; i < Arity; i++) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f));
    kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(
        args[i], in[i] + data_offset, num);
  }

  constexpr bool kCallElementwiseAny =
      paddle::platform::FunctionTraits<Functor>::has_pointer_args;
  ElementwisePrimitiveCaller<InT,
                             ConditionalT<OutT, NumOuts>,
                             VecSize,
                             Functor,
                             Arity,
                             kCallElementwiseAny>()(func, args, result);

  ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, NumOuts>()(
      outs, result, data_offset, num);
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
__global__ void VectorizedElementwiseKernel(
    paddle::framework::Array<const InT *__restrict__, Arity> ins,
    paddle::framework::Array<OutT *, NumOuts> outs,
    int size,
    int main_offset,
    Functor func) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  for (; data_offset < main_offset; data_offset += stride) {
    VectorizedElementwiseKernelImpl<InT,
                                    OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    false>(
        ins, outs, VecSize * BLOCK_NUM_X, data_offset, func);
  }

  int num = size - data_offset;
  if (num > 0) {
    VectorizedElementwiseKernelImpl<InT,
                                    OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    true>(ins, outs, num, data_offset, func);
  }
}

template <typename InT, typename OutT>
int GetVectorizedSizeForTensors(const std::vector<const DenseTensor *> &ins,
                                const std::vector<DenseTensor *> &outs) {
  int vec_size = 4;
  for (auto iter = ins.begin(); iter != ins.end(); ++iter) {
    vec_size = std::min<int>(
        vec_size, paddle::platform::GetVectorizedSize((*iter)->data<InT>()));
  }
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size = std::min<int>(
        vec_size, paddle::platform::GetVectorizedSize((*iter)->data<OutT>()));
  }
  return vec_size;
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
void ElementwiseCudaKernel(const paddle::platform::CUDADeviceContext &ctx,
                           const std::vector<const DenseTensor *> &ins,
                           std::vector<DenseTensor *> *outs,
                           Functor func) {
  auto numel = ins[0]->numel();
  int block_size = funcs::GetThreadsConfig(ctx, numel, VecSize);
  int grid_size =
      ((numel + VecSize - 1) / VecSize + block_size - 1) / block_size;
  auto stream = ctx.stream();
  paddle::framework::Array<const InT *__restrict__, Arity> ins_data;
  paddle::framework::Array<OutT *, NumOuts> outs_data;

  for (int i = 0; i < Arity; ++i) {
    ins_data[i] = ins[i]->data<InT>();
  }
  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (*outs)[i]->mutable_data<OutT>();
  }
#ifdef PADDLE_WITH_XPU2
  block_size = 128;
  grid_size = 8;
  int main_offset = (numel / (VecSize * block_size)) * VecSize * block_size;
  VectorizedElementwiseKernel<InT,
                              OutT,
                              Functor,
                              Arity,
                              NumOuts,
                              VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, outs_data, numel, main_offset, func);
#else
  int main_offset = (numel / (VecSize * block_size)) * VecSize * block_size;
  VectorizedElementwiseKernel<InT,
                              OutT,
                              Functor,
                              Arity,
                              NumOuts,
                              VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, outs_data, numel, main_offset, func);
#endif
}

template <ElementwiseType ET,
          typename InT,
          typename OutT,
          typename Functor,
          int NumOuts = 1>
void LaunchSameDimsElementwiseCudaKernel(
    const paddle::platform::CUDADeviceContext &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
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
  PADDLE_ENFORCE_EQ(outs->size(),
                    NumOuts,
                    paddle::platform::errors::InvalidArgument(
                        "Number of outputs shall equal to number of functions, "
                        "but number of outputs is %d, of functions is %d.",
                        outs->size(),
                        NumOuts));

  if (NumOuts > 1) {
    for (int i = 1; i < NumOuts; ++i) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          paddle::platform::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
  }

  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSizeForTensors<InT, OutT>(ins, *outs);
  switch (vec_size) {
    case 4:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, NumOuts, 4>(
          ctx, ins, outs, func);
      break;
    case 2:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, NumOuts, 2>(
          ctx, ins, outs, func);
      break;
    case 1:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, NumOuts, 1>(
          ctx, ins, outs, func);
      break;
    default: {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

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
    const T *__restrict__ src,
    uint32_t block_offset,
    const kps::details::BroadcastConfig<Rank> &config,
    int numel,
    int num,
    bool need_broadcast) {
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
    const paddle::framework::Array<const InT *__restrict__, Arity> &ins,
    paddle::framework::Array<OutT *, NumOuts> outs,
    const paddle::framework::Array<bool, Arity> &use_broadcast,
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
  ElementwisePrimitiveCaller<InT,
                             ConditionalT<OutT, NumOuts>,
                             VecSize,
                             Functor,
                             Arity,
                             kCallElementwiseAny>()(func, args, result);

  ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, NumOuts>()(
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
    paddle::framework::Array<const InT *__restrict__, Arity> ins,
    paddle::framework::Array<OutT *, NumOuts> outs,
    paddle::framework::Array<bool, Arity> use_broadcast,
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
  if (block_offset < numel) {
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
void LaunchKernel(const paddle::platform::CUDADeviceContext &ctx,
                  const std::vector<const DenseTensor *> &ins,
                  std::vector<DenseTensor *> *outs,
                  Functor func,
                  DimensionsTransform merge_dims) {
  int numel = (*outs)[0]->numel();
  const int threads = 256;
  int blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;

  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);
  auto stream = ctx.stream();

  paddle::framework::Array<kps::details::BroadcastConfig<Rank>, Arity> configs;
  paddle::framework::Array<bool, Arity> use_broadcast;
  paddle::framework::Array<const InT *__restrict__, Arity> ins_data;
  paddle::framework::Array<OutT *, NumOuts> outs_data;

  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (*outs)[i]->mutable_data<OutT>();
  }

  for (int i = 0; i < Arity; i++) {
    use_broadcast[i] = (ins[i]->numel() != numel);
    ins_data[i] = ins[i]->data<InT>();
    if (use_broadcast[i]) {
      // get the broadcast config,
      // if data shape is[m, n], then you should set data_dim = {n, m}
      // eg: out's shape [3, 45, 1]. then out_dims = {1, 45, 3}
      configs[i] = kps::details::BroadcastConfig<Rank>(
          merge_dims.out_dims, merge_dims.in_dims[i], merge_dims.dim_size);
    }
  }

#ifdef PADDLE_WITH_XPU2
  threads = 128;
  blocks = 8;
  main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  tail_tid = numel % (VecSize * threads);
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
    const paddle::platform::CUDADeviceContext &ctx,
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
    const paddle::platform::CUDADeviceContext &ctx,
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
void LaunchElementwiseCudaKernel(
    const paddle::platform::CUDADeviceContext &cuda_ctx,
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
    LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT, Functor, NumOuts>(
        cuda_ctx, ins, outs, func);
  } else {
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT, Functor, NumOuts>(
        cuda_ctx, ins, outs, axis, func);
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

}  // namespace pten
