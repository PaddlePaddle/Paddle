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

#define MAX_INPUT_NUM 3

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

template <typename T, int VecSize, int DATA_PER_THREAD, int ShapeSize,
          bool IsBoundary = false>
__device__ __forceinline__ void LoadData(
    T *dst, const T *__restrict__ src, uint32_t fix,
    kps::details::BroadcastConfig<ShapeSize> config, int numel, int num,
    bool need_broadcast) {
  // numel : whole num of input
  // num: how many data will be deal with in this time
  if (need_broadcast) {
    kps::ReadDataBc<T, VecSize, DATA_PER_THREAD, 1, ShapeSize, IsBoundary>(
        dst, src, fix, config, numel, 1, 1);
  } else {
    kps::ReadData<T, VecSize, 1, 1, IsBoundary>(dst, src + fix, num);
  }
}

template <typename InT, typename OutT, int ShapeSize, int VecSize,
          int DATA_PER_THREAD, typename Functor>
__global__ void BroadcastKernelTernary(
    const InT *__restrict__ in0, const InT *__restrict__ in1,
    const InT *__restrict__ in2, OutT *out,
    framework::Array<bool, MAX_INPUT_NUM> use_broadcast, uint32_t numel,
    framework::Array<kps::details::BroadcastConfig<ShapeSize>, MAX_INPUT_NUM>
        configlists,
    int main_tid, int tail_tid, Functor func) {
  int fix = blockIdx.x * blockDim.x * VecSize;  // data offset of this block
  int num = tail_tid;
  InT arg0[VecSize * DATA_PER_THREAD];
  InT arg1[VecSize * DATA_PER_THREAD];
  InT arg2[VecSize * DATA_PER_THREAD];
  OutT result[VecSize * DATA_PER_THREAD];
  const bool is_boundary = true;
  if (blockIdx.x < main_tid) {
    num = blockDim.x * VecSize;  // blockIdx.x < main_tid
    // load in0, in1, in2
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize>(
        &arg0[0], in0, fix, configlists[0], numel, num, use_broadcast[0]);
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize>(
        &arg1[0], in1, fix, configlists[1], numel, num, use_broadcast[1]);
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize>(
        &arg2[0], in2, fix, configlists[2], numel, num, use_broadcast[2]);
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, arg0, arg1, arg2, func);
    kps::WriteData<OutT, VecSize, 1, 1>(out + fix, result, num);
  } else {
    // load in0, in1, in2
    kps::Init<InT, VecSize * DATA_PER_THREAD>(&arg0[0], static_cast<InT>(1.0f));
    kps::Init<InT, VecSize * DATA_PER_THREAD>(&arg1[0], static_cast<InT>(1.0f));
    kps::Init<InT, VecSize * DATA_PER_THREAD>(&arg2[0], static_cast<InT>(1.0f));
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize, is_boundary>(
        &arg0[0], in0, fix, configlists[0], numel, num, use_broadcast[0]);
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize, is_boundary>(
        &arg1[0], in1, fix, configlists[1], numel, num, use_broadcast[1]);
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize, is_boundary>(
        &arg2[0], in2, fix, configlists[2], numel, num, use_broadcast[2]);
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, arg0, arg1, arg2, func);
    kps::WriteData<OutT, VecSize, 1, 1, is_boundary>(out + fix, result, num);
  }
}

template <typename InT, typename OutT, int ShapeSize, int VecSize,
          int DATA_PER_THREAD, typename Functor>
__global__ void BroadcastKernelBinary(
    const InT *__restrict__ in0, const InT *__restrict__ in1, OutT *out,
    framework::Array<bool, MAX_INPUT_NUM> use_broadcast, uint32_t numel,
    framework::Array<kps::details::BroadcastConfig<ShapeSize>, MAX_INPUT_NUM>
        configlists,
    int main_tid, int tail_tid, Functor func) {
  int fix = blockIdx.x * blockDim.x * VecSize;  // data offset of this block
  int num = tail_tid;
  InT arg0[VecSize * DATA_PER_THREAD];
  InT arg1[VecSize * DATA_PER_THREAD];
  OutT result[VecSize * DATA_PER_THREAD];
  const bool is_boundary = true;
  if (blockIdx.x < main_tid) {
    num = blockDim.x * VecSize;  // blockIdx.x < main_tid
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize>(
        &arg0[0], in0, fix, configlists[0], numel, num, use_broadcast[0]);
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize>(
        &arg1[0], in1, fix, configlists[1], numel, num, use_broadcast[1]);
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, arg0,
                                                              arg1, func);
    kps::WriteData<OutT, VecSize, 1, 1>(out + fix, result, num);
  } else {  // reminder
    kps::Init<InT, VecSize * DATA_PER_THREAD>(&arg0[0], static_cast<InT>(1.0f));
    kps::Init<InT, VecSize * DATA_PER_THREAD>(&arg1[0], static_cast<InT>(1.0f));
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize, is_boundary>(
        &arg0[0], in0, fix, configlists[0], numel, num, use_broadcast[0]);
    LoadData<InT, VecSize, DATA_PER_THREAD, ShapeSize, is_boundary>(
        &arg1[0], in1, fix, configlists[1], numel, num, use_broadcast[1]);
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, arg0,
                                                              arg1, func);
    kps::WriteData<OutT, VecSize, 1, 1, is_boundary>(out + fix, result, num);
  }
}

template <typename InT, typename OutT, int ShapeSize, int VecSize,
          int DATA_PER_THREAD, typename Functor>
__global__ void BroadcastKernelUnary(
    const InT *__restrict__ in, OutT *out, int numel,
    kps::details::BroadcastConfig<ShapeSize> config, int main_tid, int tail_tid,
    Functor func) {
  int fix = blockIdx.x * blockDim.x;
  int num = tail_tid;
  InT arg[VecSize * DATA_PER_THREAD];
  OutT result[VecSize * DATA_PER_THREAD];
  const bool is_boundary = true;
  if (blockIdx.x < main_tid) {
    num = blockDim.x * VecSize;  // blockIdx.x < main_tid
    kps::ReadDataBc<InT, VecSize, DATA_PER_THREAD, 1, ShapeSize>(
        &arg[0], in, fix * VecSize, config, numel, 1, 1);
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(&result[0],
                                                             &arg[0], func);
    kps::WriteData<OutT, VecSize, 1, 1>(out + fix * VecSize, &result[0], num);
  } else {
    kps::Init<InT, VecSize * DATA_PER_THREAD>(&arg[0], static_cast<InT>(1.0f));
    kps::ReadDataBc<InT, VecSize, DATA_PER_THREAD, 1, ShapeSize, is_boundary>(
        &arg[0], in, fix * VecSize, config, numel, 1, 1);
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(&result[0],
                                                             &arg[0], func);
    kps::WriteData<OutT, VecSize, 1, 1, is_boundary>(out + fix * VecSize,
                                                     &result[0], num);
  }
}

template <typename InT, typename OutT, ElementwiseType ET, int VecSize,
          int Size, typename Functor>
void LaunchKernel(const platform::CUDADeviceContext &ctx,
                  const std::vector<const framework::Tensor *> &ins,
                  framework::Tensor *out, Functor func,
                  DimensionsTransform merge_dims) {
  int numel = out->numel();
  const int threads = 256;
  const int data_per_thread = 1;
  int blocks =
      ((numel + VecSize * data_per_thread - 1) / (VecSize * data_per_thread) +
       threads - 1) /
      threads;

  int main_tid = numel / (data_per_thread * VecSize * threads);
  int tail_tid = numel % (data_per_thread * VecSize * threads);
  auto stream = ctx.stream();
  OutT *out_data = out->data<OutT>();

  framework::Array<kps::details::BroadcastConfig<Size>, MAX_INPUT_NUM>
      configlists;
  framework::Array<bool, MAX_INPUT_NUM> use_broadcast;

  for (int i = 0; i < ET; i++) {
    use_broadcast[i] = (ins[i]->numel() != numel);
    if (use_broadcast[i]) {
      configlists[i] = kps::details::BroadcastConfig<Size>(
          merge_dims.out_dims, merge_dims.in_dims[i], merge_dims.dim_size);
    }
  }

  if (ET == kUnary) {
    BroadcastKernelUnary<InT, OutT, Size, VecSize, data_per_thread,
                         Functor><<<blocks, threads, 0, stream>>>(
        ins[0]->data<InT>(), out_data, numel, configlists[0], main_tid,
        tail_tid, func);
  } else if (ET == kBinary) {  // kBinary
    BroadcastKernelBinary<InT, OutT, Size, VecSize, data_per_thread,
                          Functor><<<blocks, threads, 0, stream>>>(
        ins[0]->data<InT>(), ins[1]->data<InT>(), out_data, use_broadcast,
        numel, configlists, main_tid, tail_tid, func);
  } else {  // Ternary
    BroadcastKernelTernary<InT, OutT, Size, VecSize, data_per_thread,
                           Functor><<<blocks, threads, 0, stream>>>(
        ins[0]->data<InT>(), ins[1]->data<InT>(), ins[2]->data<InT>(), out_data,
        use_broadcast, numel, configlists, main_tid, tail_tid, func);
  }
}

template <typename InT, typename OutT, ElementwiseType ET, int VecSize,
          typename Functor>
void LaunchBroadcastKernelForDifferentDimSize(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    int axis, Functor func) {
  const auto merge_dims = DimensionsTransform(ins, out->dims(), axis);

#define DIM_SIZE(size)                                                       \
  case size: {                                                               \
    LaunchKernel<InT, OutT, ET, VecSize, size, Functor>(ctx, ins, out, func, \
                                                        merge_dims);         \
  } break;

  switch (merge_dims.dim_size) {
    DIM_SIZE(1);
    DIM_SIZE(2);
    DIM_SIZE(3);
    DIM_SIZE(4);
    DIM_SIZE(5);
    DIM_SIZE(6);
    DIM_SIZE(7);
    DIM_SIZE(8);
  }
#undef DIM_SIZE
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  PADDLE_ENFORCE_EQ(ET, ElementwiseType::kBinary,
                    platform::errors::InvalidArgument(
                        "Currently, only Support binary calculation, "
                        "but received %d input tensors.\n",
                        static_cast<int>(ET)));
  int in_vec_size = 4;
  framework::Tensor *out = (*outs)[0];
  for (auto *in : ins) {
    auto temp_size = platform::GetVectorizedSize<InT>(in->data<InT>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  int out_vec_size = platform::GetVectorizedSize<OutT>(out->data<OutT>());
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 4: {
      LaunchBroadcastKernelForDifferentDimSize<InT, OutT, ET, 4>(ctx, ins, out,
                                                                 axis, func);
      break;
    }
    case 2: {
      LaunchBroadcastKernelForDifferentDimSize<InT, OutT, ET, 2>(ctx, ins, out,
                                                                 axis, func);
      break;
    }
    case 1: {
      LaunchBroadcastKernelForDifferentDimSize<InT, OutT, ET, 1>(ctx, ins, out,
                                                                 axis, func);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const platform::CUDADeviceContext &cuda_ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<int> dims_size;
  bool no_broadcast_flag = true;
  for (auto *in : ins) {
    no_broadcast_flag = ins[0]->dims() == in->dims();
    dims_size.emplace_back(in->dims().size());
  }

  if (no_broadcast_flag) {
    LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                       func);
  } else {
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                        axis, func);
  }
}

#undef MAX_INPUT_NUM

}  // namespace operators
}  // namespace paddle
