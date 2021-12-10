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

#include "paddle/fluid/operators/elementwise/elementwise_op_impl_test.cu.h"
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
            // PADDLE_THROW(platform::errors::InvalidArgument(
            //    "The %d-th dimension of input tensor is expected to be equal "
            //    "with the %d-th dimension of output tensor %d or 1, but "
            //    "recieved %d.",
            //    in_idx + 1, axis + 1, out_dims[axis], in_dim[in_idx]));
          }
        } while (in_idx < in_dim.size());
        in_dim.resize(dim_size);
        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      } else {
        do {
          if (in_dim[in_idx] == out_dims[in_idx] || in_dim[in_idx] == 1) {
            in_idx++;
          } else {
            // PADDLE_THROW(platform::errors::InvalidArgument(
            //    "The %d-th dimension of input tensor is expected to be equal "
            //    "with the %d-th dimension of output tensor %d or 1, but "
            //    "recieved %d.",
            //    in_idx + 1, in_idx + 1, out_dims[in_idx], in_dim[in_idx]));
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

template <typename T, int VecSize, int Rank, bool IsBoundary = false>
__device__ void LoadData(T *dst, const T _global_ptr_ *src, int block_offset,
                         const int _global_ptr_ *stride_out,
                         const int _global_ptr_ *stride_in,
                         const int _global_ptr_ *dim_in, int numel, int num,
                         bool need_broadcast) {
  // numel : whole num of output
  // num: how many data will be deal with in this time
  if (need_broadcast) {
    kps::ReadDataBc<T, VecSize, 1, 1, Rank, IsBoundary>(
        dst, src, stride_out, stride_in, dim_in, block_offset, numel);
  } else {
    kps::ReadData<T, VecSize, 1, 1, IsBoundary>(dst, src + block_offset, num);
  }
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          int Rank, bool IsBoundary = false>
__device__ void BroadcastKernelImpl(
    const InT _global_ptr_ *in0, const InT _global_ptr_ *in1,
    const InT _global_ptr_ *in2, OutT _global_ptr_ *out, bool is_broadcast0,
    bool is_broadcast1, bool is_broadcast2, const int _global_ptr_ *stride_in0,
    const int _global_ptr_ *stride_in1, const int _global_ptr_ *stride_in2,
    const int _global_ptr_ *stride_out, const int _global_ptr_ *dim_in0,
    const int _global_ptr_ *dim_in1, const int _global_ptr_ *dim_in2, int numel,
    int num, int block_offset, Functor func) {
  __local__ InT args[Arity][VecSize];
  __local__ OutT result[VecSize];
  // init + load
  kps::Init<InT, VecSize>(args[0], static_cast<InT>(1.0f));
  LoadData<InT, VecSize, Rank, IsBoundary>(args[0], in0, block_offset,
                                           stride_out, stride_in0, dim_in0,
                                           numel, num, is_broadcast0);
  if (Arity >= 2) {
    kps::Init<InT, VecSize>(args[1], static_cast<InT>(1.0f));
    LoadData<InT, VecSize, Rank, IsBoundary>(args[1], in1, block_offset,
                                             stride_out, stride_in1, dim_in1,
                                             numel, num, is_broadcast1);
  }
  if (Arity == 3) {
    kps::Init<InT, VecSize>(args[2], static_cast<InT>(1.0f));
    LoadData<InT, VecSize, Rank, IsBoundary>(args[2], in2, block_offset,
                                             stride_out, stride_in2, dim_in2,
                                             numel, num, is_broadcast2);
  }

  // const bool kCallElementwiseAny =
  //    platform::FunctionTraits<Functor>::has_pointer_args;
  // ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity,
  //                           kCallElementwiseAny>()(func, args, result);
  kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, args[0],
                                                            args[1], func);

  kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(out + block_offset, result,
                                                  num);
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          int Rank>
__global__ void BroadcastKernel(const InT *in0, const InT *in1, const InT *in2,
                                OutT *out, bool is_broadcast0,
                                bool is_broadcast1, bool is_broadcast2,
                                const int *stride_in0, const int *stride_in1,
                                const int *stride_in2, const int *stride_out,
                                const int *dim_in0, const int *dim_in1,
                                const int *dim_in2, int numel, int main_offset,
                                int tail_tid, Functor func) {
  int block_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
#ifdef PADDLE_WITH_XPU2
  for (; block_offset < main_offset; block_offset += stride) {
    BroadcastKernelImpl<InT, OutT, Functor, Arity, VecSize, Rank, false>(
        in0, in1, in2, out, is_broadcast0, is_broadcast1, is_broadcast2,
        stride_in0, stride_in1, stride_in2, stride_out, dim_in0, dim_in1,
        dim_in2, numel, BLOCK_NUM_X * VecSize, block_offset, func);
  }
  int num = numel - block_offset;
  if (num > 0) {
    BroadcastKernelImpl<InT, OutT, Functor, Arity, VecSize, Rank, true>(
        in0, in1, in2, out, is_broadcast0, is_broadcast1, is_broadcast2,
        stride_in0, stride_in1, stride_in2, stride_out, dim_in0, dim_in1,
        dim_in2, numel, num, block_offset, func);
  }

#else
//  if (block_offset < main_offset) {
//    BroadcastKernelImpl<InT, OutT, Functor, Arity, VecSize, Rank, false>(
//        ins, out, use_broadcast, numel, configs, BLOCK_NUM_X * VecSize,
//        block_offset, func);
//  } else {
//    BroadcastKernelImpl<InT, OutT, Functor, Arity, VecSize, Rank, true>(
//        ins, out, use_broadcast, numel, configs, tail_tid, block_offset,
//        func);
//  }
#endif
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          int Rank>
void LaunchKernel(const platform::XPUDeviceContext &ctx,
                  const std::vector<const framework::Tensor *> &ins,
                  framework::Tensor *out, Functor func,
                  DimensionsTransform merge_dims) {
  int numel = out->numel();
  const int threads = 256;
  int blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;

  int main_offset = (numel / (VecSize * threads)) * VecSize * threads;
  int tail_tid = numel % (VecSize * threads);
  OutT *out_data = out->data<OutT>();

  framework::Array<kps::details::BroadcastConfig<Rank>, Arity> configs;
  framework::Array<bool, 3> use_broadcast;
  framework::Array<const InT *__restrict__, 3> ins_data;
  use_broadcast[0] = false;
  use_broadcast[1] = false;
  use_broadcast[2] = false;

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
  // 1. malloc for broadcast config : in_stride out_stride in_dims
  // 2. memcopy from host to device
  // 3. free

  // malloc host and device
  int *stride_out = nullptr;
  int *stride_in0 = nullptr;
  int *stride_out_tmp = nullptr;
  int *stride_in1 = nullptr;
  int *stride_in2 = nullptr;
  int *dim_in0 = nullptr;
  int *dim_in1 = nullptr;
  int *dim_in2 = nullptr;
  int ret;
  ret = xpu_malloc((void **)&stride_out, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&stride_in0, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&stride_out_tmp, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&stride_in1, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&stride_in2, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&dim_in0, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&dim_in1, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void **)&dim_in2, Rank * sizeof(int));
  assert(ret == 0);
  ret = xpu_memcpy(stride_out, configs[0].strides_out, Rank * sizeof(int),
                   XPU_HOST_TO_DEVICE);
  assert(ret == 0);
  if (use_broadcast[0]) {
    ret = xpu_memcpy(stride_in0, configs[0].strides_in, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
    ret = xpu_memcpy(stride_out_tmp, configs[0].strides_out, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
    ret = xpu_memcpy(dim_in0, configs[0].in_dim, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
  }
  if (use_broadcast[1]) {
    ret = xpu_memcpy(stride_in1, configs[1].strides_in, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
    ret = xpu_memcpy(stride_out_tmp, configs[1].strides_out, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
    ret = xpu_memcpy(dim_in1, configs[1].in_dim, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
  }
  if (use_broadcast[2]) {
    ret = xpu_memcpy(stride_in2, configs[2].strides_in, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
    ret = xpu_memcpy(dim_in2, configs[2].in_dim, Rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
    assert(ret == 0);
  }
  const int cores = 64;
  int clusters = 8;
  auto stream = ctx.x_context()->xpu_stream;
  main_offset = (numel / (VecSize * cores)) * VecSize * cores;
  tail_tid = numel % (VecSize * cores);
  BroadcastKernel<InT, OutT, Functor, Arity, VecSize,
                  Rank><<<clusters, cores, stream>>>(
      ins_data[0], ins_data[1], ins_data[2], out_data, use_broadcast[0],
      use_broadcast[1], use_broadcast[2], stride_in0, stride_in1, stride_in2,
      stride_out_tmp, dim_in0, dim_in1, dim_in2, numel, main_offset, tail_tid,
      func);
  //  for(int i = Rank - 1; i >= 0; --i) {
  //   if(use_broadcast[0]) printf("this is use_0 stride_out %d  stride_in %d
  //   in_dim %d\n", configs[0].strides_out[i], configs[0].strides_in[i],
  //   configs[0].in_dim[i]);
  //   if(use_broadcast[1]) printf("use_1 stride_out %d  stride_in %d in_dim
  //   %d\n", configs[1].strides_out[i], configs[1].strides_in[i],
  //   configs[1].in_dim[i]);
  //   if(use_broadcast[2]) printf("************* use_2 stride_out %d  stride_in
  //   %d in_dim %d\n", configs[2].strides_out[i], configs[2].strides_in[i],
  //   configs[2].in_dim[i]);
  //  }
  //  int* strides_c= (int*) malloc( Rank * sizeof(int));
  //  int* stride_in0_c= (int*) malloc( Rank * sizeof(int));
  //  int* stride_in1_c= (int*) malloc( Rank * sizeof(int));
  //  int * in_dim_0 = (int*) malloc(Rank * sizeof(int));
  //  int * in_dim_1 = (int*) malloc(Rank * sizeof(int));
  //  xpu_memcpy(strides_c, stride_out_tmp, Rank * sizeof(int),
  //  XPU_DEVICE_TO_HOST);
  //  xpu_memcpy(stride_in0_c, stride_in0, Rank * sizeof(int),
  //  XPU_DEVICE_TO_HOST);
  //  xpu_memcpy(stride_in1_c, stride_in1, Rank * sizeof(int),
  //  XPU_DEVICE_TO_HOST);
  //  xpu_memcpy(in_dim_0, dim_in0, Rank * sizeof(int), XPU_DEVICE_TO_HOST);
  //  xpu_memcpy(in_dim_1, dim_in1, Rank * sizeof(int), XPU_DEVICE_TO_HOST);
  // // for(int i = Rank - 1; i >= 0; --i) {
  // //  if(use_broadcast[0]) printf("this is use_0 stride_out %d  stride_in %d
  // in_dim %d\n", strides_c[i], stride_in0_c[i], in_dim_0[i]);
  // //  if(use_broadcast[1]) printf("use_1 stride_out %d  stride_in %d in_dim
  // %d\n", strides_c[i], stride_in1_c[i], in_dim_1[i]);
  // // }
  //  free(strides_c);
  //  free(stride_in0_c);
  //  free(stride_in1_c);
  //  free(in_dim_0);
  //  free(in_dim_1);
  // ret = xpu_memcpy(output_cpu, ins_data[0], numel * sizeof(float),
  // XPU_DEVICE_TO_HOST);
  // printf("this is input 0    start\n ");
  // for(int i = 0; i < numel; i++) {
  //   printf("%d %f \n", i, output_cpu[i]);
  // }
  // ret = xpu_memcpy(output_cpu, ins_data[1], numel * sizeof(float),
  // XPU_DEVICE_TO_HOST);
  // printf("this is input 1    start \n");
  // for(int i = 0; i < numel; i++) {
  //   printf("%d %f \n", i, output_cpu[i]);
  // }
  // ret = xpu_memcpy(output_cpu, out_data, numel * sizeof(float),
  // XPU_DEVICE_TO_HOST);
  // printf("this is elementwise add printf start : index data\n");
  // for(int i = 0; i < numel; i++) {
  //   printf("%d %f \n", i, output_cpu[i]);
  // }
  // printf("this is elementwise add printf end!!!!!\n\n");

  xpu_free(stride_out);
  xpu_free(stride_in0);
  xpu_free(stride_in1);
  xpu_free(stride_in2);
  xpu_free(dim_in0);
  xpu_free(dim_in1);
  xpu_free(dim_in2);
// free(output_cpu);
#else
// auto stream = ctx.stream();
// BroadcastKernel<InT, OutT, Functor, Arity, VecSize,
//                 Rank><<<blocks, threads, 0, stream>>>(
//     ins_data, out_data, use_broadcast, numel, configs, main_offset, tail_tid,
//     func);
#endif
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize>
void LaunchBroadcastKernelForDifferentVecSize(
    const platform::XPUDeviceContext &ctx,
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
      // PADDLE_THROW(platform::errors::InvalidArgument(
      //     "The maximum dimension of input tensor is expected to be less than
      //     "
      //     "%d, but recieved %d.\n",
      //     merge_dims.dim_size, framework::DDim::kMaxRank));
    }
  }
#undef CALL_BROADCAST_FOR_DIM_SIZE
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::XPUDeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  using Traits = platform::FunctionTraits<Functor>;
  const int kArity =
      Traits::has_pointer_args ? static_cast<int>(ET) : Traits::arity;
  //  PADDLE_ENFORCE_EQ(ins.size(), kArity,
  //                    platform::errors::InvalidArgument(
  //                        "The number of inputs is expected to be equal to the
  //                        "
  //                        "arity of functor. But recieved: the number of
  //                        inputs "
  //                        "is %d, the arity of functor is %d.",
  //                        ins.size(), kArity));
  //  PADDLE_ENFORCE_EQ(kArity, 2,
  //                    platform::errors::InvalidArgument(
  //                        "Currently only broadcast of binary is supported and
  //                        "
  //                        "verified, but received %d.",
  //                        kArity));
  //
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
      LaunchBroadcastKernelForDifferentVecSize<InT, OutT, Functor, kArity, 4>(
          ctx, ins, out, axis, func);
      break;
    }
    case 2: {
      LaunchBroadcastKernelForDifferentVecSize<InT, OutT, Functor, kArity, 2>(
          ctx, ins, out, axis, func);
      break;
    }
    case 1: {
      LaunchBroadcastKernelForDifferentVecSize<InT, OutT, Functor, kArity, 1>(
          ctx, ins, out, axis, func);
      break;
    }
    default: {
      // PADDLE_THROW(platform::errors::Unimplemented(
      //    "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}
template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const platform::XPUDeviceContext &cuda_ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<int> dims_size;
  bool no_broadcast_flag = true;
  for (auto *in : ins) {
    //	std::cout<<" ***********elementwise add input dim
    //:**********"<<std::endl;
    //	for(int i = 0; i < in->dims().size(); i++) {
    //       std::cout<<" "<<in->dims()[i];
    //	}
    //    std::cout<<" "<<std::endl;

    no_broadcast_flag = ins[0]->dims() == in->dims();
    dims_size.emplace_back(in->dims().size());
  }

  if (no_broadcast_flag) {
    LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                       func);
  } else {
    //	std::cout<<" ################### braoadcast Add
    //################"<<std::endl;
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                        axis, func);
  }
}

}  // namespace operators
}  // namespace paddle
