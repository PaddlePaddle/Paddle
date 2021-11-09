// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.1
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"

namespace paddle {
namespace operators {

namespace kps = paddle::operators::kernel_primitives;

struct DimensionsTransform {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(bool &, std::vector<DimVector> &, DimVector &,
                               int, int);
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
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "recieved %d.",
                in_idx + 1, axis + 1, out_dims[axis], in_dim[in_idx]));
          }
        } while (in_idx < in_dim.size());
        in_dim.resize(dim_size);
        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      } else {
        do {
          if (in_dim[in_idx] == out_dims[in_idx] || in_dim[in_idx] == 1) {
            in_idx++;
          } else {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "recieved %d.",
                in_idx + 1, in_idx + 1, out_dims[in_idx], in_dim[in_idx]));
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
      (*vec)[m_idx - 1] =
          std::accumulate(vec->begin() + l_idx, vec->begin() + m_idx, 1,
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
  explicit DimensionsTransform(
      const std::vector<const framework::Tensor *> &ins,
      const framework::DDim &dims, int axis) {
    const int N = ins.size();
    dim_size = dims.size();
    out_dims = framework::vectorize<int64_t>(dims);
    in_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      in_dims[j] = framework::vectorize<int64_t>(ins[j]->dims());
    }
    InputDimensionsExtend(N, axis);

    auto merge_sequential_dims = [](bool &equal,
                                    std::vector<DimVector> &in_dims,
                                    DimVector &out, int i, int num) {
      for (int j = 1; j < num; ++j) {
        equal = (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
    auto merge_sequential_one_dims = [](bool &equal,
                                        std::vector<DimVector> &in_dims,
                                        DimVector &out, int i, int num) {
      equal = in_dims[0][i] == 1;
      if (equal) {
        for (int j = 1; j < num; ++j) {
          equal = in_dims[j][i] == out[i];
        }
      }
    };
    // To Merge the dimensions of input_tensors while the consequtive
    // equal-dimensions appears.
    MergeFunctor merge_ptr = merge_sequential_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);

    int min_idx = 0;
    int min_val = std::accumulate(in_dims[0].begin(), in_dims[0].end(), 1,
                                  std::multiplies<int64_t>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(in_dims[j].begin(), in_dims[j].end(), 1,
                                 std::multiplies<int64_t>());
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

<<<<<<< 9c59170353fcd17b44c7de560bd56760cbd5786b
=======
template <typename T, int VecSize, int Rank, bool IsBoundary = false>
__device__ __forceinline__ void LoadData(
    T *dst, const T *__restrict__ src, uint32_t block_offset,
    const kps::details::BroadcastConfig<Rank> &config, int numel, int num,
    bool need_broadcast) {
  // numel : whole num of output
  // num: how many data will be deal with in this time
  if (need_broadcast) {
    kps::ReadDataBc<T, VecSize, 1, 1, Rank, IsBoundary>(dst, src, block_offset,
                                                        config, numel);
  } else {
    kps::ReadData<T, VecSize, 1, 1, IsBoundary>(dst, src + block_offset, num);
  }
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          int Rank, bool IsBoundary = false>
__device__ void BroadcastKernelImpl(
    const framework::Array<const InT *__restrict__, Arity> &ins, OutT *out,
    const framework::Array<bool, Arity> &use_broadcast, uint32_t numel,
    const framework::Array<kps::details::BroadcastConfig<Rank>, Arity> &configs,
    int num, int block_offset, Functor func) {
  InT args[Arity][VecSize];
  OutT result[VecSize];

#pragma unroll
  for (int i = 0; i < Arity; i++) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f));
    LoadData<InT, VecSize, Rank, IsBoundary>(args[i], ins[i], block_offset,
                                             configs[i], numel, num,
                                             use_broadcast[i]);
  }

  const bool kCallElementwiseAny =
      platform::FunctionTraits<Functor>::has_pointer_args;
  ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity,
                             kCallElementwiseAny>()(func, args, result);
  kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(out + block_offset, result,
                                                  num);
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          int Rank>
__global__ void BroadcastKernel(
    framework::Array<const InT *__restrict__, Arity> ins, OutT *out,
    framework::Array<bool, Arity> use_broadcast, uint32_t numel,
    framework::Array<kps::details::BroadcastConfig<Rank>, Arity> configs,
    int main_offset, int tail_tid, Functor func) {
  int block_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  // data offset of this block
  for (; block_offset < main_offset; block_offset += stride) {
    BroadcastKernelImpl<InT, OutT, Functor, Arity, VecSize, Rank, false>(
        ins, out, use_broadcast, numel, configs, BLOCK_NUM_X * VecSize,
        block_offset, func);
  }

  if (block_offset < numel) {
    BroadcastKernelImpl<InT, OutT, Functor, Arity, VecSize, Rank, true>(
        ins, out, use_broadcast, numel, configs, tail_tid, block_offset, func);
  }
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          int Rank>
void LaunchKernel(const platform::CUDADeviceContext &ctx,
                  const std::vector<const framework::Tensor *> &ins,
                  framework::Tensor *out, Functor func,
                  DimensionsTransform merge_dims) {
  int numel = out->numel();
  const int threads = 256;
  int blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;

  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);
  auto stream = ctx.stream();
  OutT *out_data = out->data<OutT>();

  framework::Array<kps::details::BroadcastConfig<Rank>, Arity> configs;
  framework::Array<bool, Arity> use_broadcast;
  framework::Array<const InT *__restrict__, Arity> ins_data;

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
  BroadcastKernel<InT, OutT, Functor, Arity, VecSize,
                  Rank><<<blocks, threads, stream>>>(
      ins_data, out_data, use_broadcast, numel, configs, main_offset, tail_tid,
      func);
#else
  BroadcastKernel<InT, OutT, Functor, Arity, VecSize,
                  Rank><<<blocks, threads, 0, stream>>>(
      ins_data, out_data, use_broadcast, numel, configs, main_offset, tail_tid,
      func);
#endif
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize>
void LaunchBroadcastKernelForDifferentVecSize(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    int axis, Functor func) {
  const auto merge_dims = DimensionsTransform(ins, out->dims(), axis);

#define CALL_BROADCAST_FOR_DIM_SIZE(rank)                                     \
  case rank: {                                                                \
    LaunchKernel<InT, OutT, Functor, Arity, VecSize, rank>(ctx, ins, out,     \
                                                           func, merge_dims); \
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
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The maximum dimension of input tensor is expected to be less than "
          "%d, but recieved %d.\n",
          merge_dims.dim_size, framework::DDim::kMaxRank));
    }
  }
#undef CALL_BROADCAST_FOR_DIM_SIZE
}

>>>>>>> modified the elementwise_op_broadcast and elementwise_op_impl for xpu2
template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<const pten::DenseTensor *> pt_inputs;
  std::vector<pten::DenseTensor *> pt_outputs;
  // TODO(YuanRisheng) *_tmp for cache DenseTensor, because the temporary
  // DenseTensor obj
  // generated by MakePtenDenseTensor can be destroyed when exits loop. *_tmp
  // can be deleted
  // when DenseTensor support copy constructor.
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_inputs_tmp;
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_outputs_tmp;
  for (auto in : ins) {
    pt_inputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*in)));
  }
  for (auto out : *outs) {
    pt_outputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*out)));
  }
  for (int i = 0; i < pt_inputs_tmp.size(); i++) {
    pt_inputs.push_back(pt_inputs_tmp[i].get());
  }
  for (int i = 0; i < pt_outputs_tmp.size(); i++) {
    pt_outputs.push_back(pt_outputs_tmp[i].get());
  }
  pten::LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(
      ctx, pt_inputs, &pt_outputs, axis, func);
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const platform::CUDADeviceContext &cuda_ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<const pten::DenseTensor *> pt_inputs;
  std::vector<pten::DenseTensor *> pt_outputs;
  // TODO(YuanRisheng) *_tmp for cache DenseTensor, because the temporary
  // DenseTensor obj
  // generated by MakePtenDenseTensor can be destroyed when exits loop. *_tmp
  // can be deleted
  // when DenseTensor support copy constructor.
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_inputs_tmp;
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_outputs_tmp;
  for (auto in : ins) {
    pt_inputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*in)));
  }
  for (auto out : *outs) {
    pt_outputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*out)));
  }
  for (int i = 0; i < pt_inputs_tmp.size(); i++) {
    pt_inputs.push_back(pt_inputs_tmp[i].get());
  }
  for (int i = 0; i < pt_outputs_tmp.size(); i++) {
    pt_outputs.push_back(pt_outputs_tmp[i].get());
  }
  pten::LaunchElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, pt_inputs,
                                                   &pt_outputs, axis, func);
}

}  // namespace operators
}  // namespace paddle
