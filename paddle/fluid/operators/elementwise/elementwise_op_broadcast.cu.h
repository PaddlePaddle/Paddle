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

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast_impl.cu.h"

namespace paddle {
namespace operators {

struct DimensionsTransform {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(bool &, std::vector<DimVector> &, DimVector &,
                               int, int);
  int64_t dim_size;
  DimVector out_dims;
  std::vector<DimVector> in_dims;

 private:
  // 1. To compensate the lackage of input_tensors` dimension;
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
                "The %dth dimension of input tensor is expected to be equal "
                "with"
                "the %dth dimension of output tensor %d or 1, but recieved "
                "%d.\n",
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
                "The %dth dimension of input tensor is expected to be equal "
                "with"
                "the %dth dimension of output tensor %d or 1, but recieved "
                "%d.\n",
                in_idx + 1, in_idx + 1, out_dims[in_idx], in_dim[in_idx]));
          }
        } while (in_idx < dim_size);
      }
      std::reverse(in_dim.begin(), in_dim.end());
    }
    std::reverse(out_dims.begin(), out_dims.end());
  }

  template <typename MergeFunctor>
  __inline__ void DimensionsReorganise(MergeFunctor merge_func, int N) {
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
    DimensionsReorganise<MergeFunctor>(merge_ptr, N);

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
    DimensionsReorganise<MergeFunctor>(merge_ptr, N);
    std::swap(in_dims[min_idx], in_dims[0]);
  }
};

struct CalculateInputStrides {
  std::vector<std::vector<uint32_t>> strides;
  std::vector<FastDivMod> divmoders;

 private:
  // To calculate the strides of each input_tensor.
  __inline__ void CalculateStrides(
      int N, int dim_size, const std::vector<std::vector<int64_t>> &in_dims) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < dim_size; ++i) {
        strides[j][i] = in_dims[j][i] == 1 ? 0 : strides[j][i];
        strides[j][i] =
            (i != 0 && strides[j][i] != 0)
                ? std::accumulate(in_dims[j].begin(), in_dims[j].begin() + i, 1,
                                  std::multiplies<int64_t>())
                : strides[j][i];
      }
    }
  }

 public:
  explicit CalculateInputStrides(
      const int64_t &dim_size, const std::vector<std::vector<int64_t>> &in_dims,
      const std::vector<int64_t> &out_dims) {
    const auto N = in_dims.size();
    divmoders.resize(dim_size);
    strides.resize(N, std::vector<uint32_t>(dim_size, 1));

    for (int i = 0; i < dim_size; ++i) {
      divmoders[i] = FastDivMod(out_dims[i]);
    }
    CalculateStrides(N, dim_size, in_dims);
  }
};

template <typename T, ElementwiseType ET, int VecSize, int kDims>
struct BroadcastArgsWarpper {
  using DimsVec = CudaAlignedVector<T, VecSize>;

  T *out_data;
  const T *__restrict__ in_data[ET];
  uint32_t strides[ET][framework::DDim::kMaxRank];
  bool no_broadcast[ET];
  FastDivMod divmoders[kDims];
  uint32_t scalar_offset;

  HOSTDEVICE BroadcastArgsWarpper(
      const std::vector<const framework::Tensor *> &ins,
      const CalculateInputStrides &offset_calculator, framework::Tensor *out,
      int scalar_offset)
      : scalar_offset(scalar_offset) {
    for (int j = 0; j < ET; ++j) {
      in_data[j] = ins[j]->data<T>();
      no_broadcast[j] = ins[j]->dims() == out->dims() ? true : false;
      memcpy(strides[j], offset_calculator.strides[j].data(),
             kDims * sizeof(uint32_t));
    }
    out_data = out->data<T>();
    memcpy(divmoders, offset_calculator.divmoders.data(),
           kDims * sizeof(FastDivMod));
  }

  __device__ __forceinline__ uint32_t GetDivmodOffset(int idx, int in_idx) {
    uint32_t offset = 0;

#pragma unroll(kDims)
    for (int i = 0; i < kDims; ++i) {
      auto fast_divmoder = divmoders[i].Divmod(idx);
      idx = fast_divmoder.val[0];
      offset += fast_divmoder.val[1] * strides[in_idx][i];
    }
    return offset;
  }

  __device__ __forceinline__ void CommonVector(DimsVec args[], int tid,
                                               int idx) {
    const DimsVec *__restrict__ vec_data =
        reinterpret_cast<const DimsVec *__restrict__>(in_data[idx]);
    args[idx] = vec_data[tid];
  }

  __device__ __forceinline__ void DivmodVector(DimsVec args[], int tid,
                                               int idx) {
    int index = tid * VecSize;

    for (int i = 0; i < VecSize; ++i) {
      uint32_t offset = GetDivmodOffset(index + i, idx);
      args[idx].val[i] = in_data[idx][offset];
    }
  }

  __device__ __forceinline__ void CommonScalar(T args[], int tid, int idx) {
    args[idx] = in_data[idx][tid + scalar_offset];
  }

  __device__ __forceinline__ void DivmodScalar(T args[], int tid, int idx) {
    auto offset = GetDivmodOffset(tid + scalar_offset, idx);
    args[idx] = in_data[idx][offset];
  }

  __device__ __forceinline__ void LoadVector(DimsVec args[], int tid) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      if (no_broadcast[j]) {
        CommonVector(args, tid, j);
      } else {
        DivmodVector(args, tid, j);
      }
    }
  }

  __device__ __forceinline__ void LoadScalar(T args[], int tid) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      if (no_broadcast[j]) {
        CommonScalar(args, tid, j);
      } else {
        DivmodScalar(args, tid, j);
      }
    }
  }

  __device__ __forceinline__ void StoreVector(DimsVec args[], int tid) {
    DimsVec *vec_out = reinterpret_cast<DimsVec *>(out_data);
    vec_out[tid] = args[0];
  }

  __device__ __forceinline__ void StoreScalar(T args[], int tid) {
    out_data[scalar_offset + tid] = args[0];
  }
};

template <typename T, typename BroadcastArgsWarpper, ElementwiseType ET>
__device__ inline void ScalarizedBroadcastKernelImpl(
    BroadcastArgsWarpper data_transfer, int tid) {
  T args[ET];
  data_transfer.LoadScalar(args, tid);

#pragma unroll(ET)
  for (int j = 1; j < ET; ++j) {
    args[0] += args[j];
  }
  data_transfer.StoreScalar(args, tid);
}

template <typename T, typename BroadcastArgsWarpper, ElementwiseType ET,
          int VecSize>
__device__ inline void VectorizedBroadcastKernelImpl(
    BroadcastArgsWarpper data_transfer, int tid) {
  using VecT = CudaAlignedVector<T, VecSize>;
  VecT args[ET];
  data_transfer.LoadVector(args, tid);

#pragma unroll(ET)
  for (int j = 1; j < ET; ++j) {
#pragma unroll(VecSize)
    for (int i = 0; i < VecSize; ++i) {
      args[0].val[i] += args[j].val[i];
    }
  }
  data_transfer.StoreVector(args, tid);
}

template <typename T, typename BroadcastArgsWarpper, ElementwiseType ET,
          int VecSize>
__global__ void ElementwiseBroadcastKernel(BroadcastArgsWarpper data_transfer,
                                           int main_tid, int tail_tid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Aimming at vectorized calculation of major data whose length is max
  // multipler of VecSize.
  if (tid < main_tid) {
    VectorizedBroadcastKernelImpl<T, BroadcastArgsWarpper, ET, VecSize>(
        data_transfer, tid);
  }
  // Aimming at scalar calculation of rest data whose lenght cannot fulfill
  // VecSize.
  if (tid < tail_tid) {
    ScalarizedBroadcastKernelImpl<T, BroadcastArgsWarpper, ET>(data_transfer,
                                                               tid);
  }
}

template <typename T, ElementwiseType ET, int VecSize = 1>
void LaunchBroadcastKernelForDifferentDimSize(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    int axis) {
  int numel = out->numel();
  const int threads = 256;
  int blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;
  int main_tid = numel / VecSize;
  int tail_tid = numel % VecSize;
  int vec_len = main_tid * VecSize;
  auto stream = ctx.stream();

  const auto merge_dims = DimensionsTransform(ins, out->dims(), axis);
  const auto offset_calculator = CalculateInputStrides(
      merge_dims.dim_size, merge_dims.in_dims, merge_dims.out_dims);

  switch (merge_dims.dim_size) {
    case 1: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 1>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 2: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 2>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 3: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 3>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 4: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 4>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 5: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 5>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 6: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 6>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 7: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 7>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    case 8: {
      auto data_transfer = BroadcastArgsWarpper<T, ET, VecSize, 8>(
          ins, offset_calculator, out, vec_len);
      ElementwiseBroadcastKernel<T, decltype(data_transfer), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          data_transfer, main_tid, tail_tid);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The maximum dimension of input tensor is expected to be less than "
          "%d, but recieved %d.\n",
          merge_dims.dim_size, framework::DDim::kMaxRank));
    }
  }
}

template <ElementwiseType ET, typename T, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    Functor func, int axis) {
  int in_vec_size = 4;
  for (auto *in : ins) {
    auto temp_size = GetVectorizedSizeImpl<T>(in->data<T>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  int out_vec_size = GetVectorizedSizeImpl<T>(out->data<T>());
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 4: {
      LaunchBroadcastKernelForDifferentDimSize<T, ET, 4>(ctx, ins, out, axis);
      break;
    }
    case 2: {
      LaunchBroadcastKernelForDifferentDimSize<T, ET, 2>(ctx, ins, out, axis);
      break;
    }
    default: {
      LaunchBroadcastKernelForDifferentDimSize<T, ET, 1>(ctx, ins, out, axis);
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
