/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cum_op.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"

using Tensor = paddle::framework::Tensor;
using LoDTensor = paddle::framework::LoDTensor;

namespace paddle {
namespace operators {

template <typename T>
__global__ void OuterScan(const T* in, T* out, int inner_dim_size,
                          int outer_dim_size, int scan_dim_size, bool exclusive,
                          bool reverse) {
  int id = blockIdx.y * blockDim.x + threadIdx.x;

  for (int outer_index = blockIdx.x; outer_index < outer_dim_size;
       outer_index += gridDim.x) {
    for (int inner_index = blockIdx.y * blockDim.x + threadIdx.x;
         inner_index < inner_dim_size; inner_index += gridDim.y * blockDim.x) {
      int scan_index_init = 0;
      int forward_direction = 1;
      int src_index =
          outer_index * scan_dim_size * inner_dim_size + inner_index;
      int dst_index =
          outer_index * scan_dim_size * inner_dim_size + inner_index;
      if (reverse) {
        src_index = src_index + (scan_dim_size - 1) * inner_dim_size;
        dst_index = dst_index + (scan_dim_size - 1) * inner_dim_size;
        forward_direction = -1;
      }
      if (exclusive) {
        scan_index_init = 1;
        out[dst_index] = 0;
        dst_index = dst_index + (forward_direction * inner_dim_size);
      }
      T acc = 0;

      for (int scan_index = scan_index_init; scan_index < scan_dim_size;
           ++scan_index) {
        acc = in[src_index] + acc;
        out[dst_index] = acc;
        src_index += (forward_direction * inner_dim_size);
        dst_index += (forward_direction * inner_dim_size);
      }
    }
  }
}

// inclusive scan
template <typename T, int num_threads_x, int num_threads_y>
__global__ void InnerMostDimInclusiveScan(const T* in, T* out,
                                          int inner_dim_size,
                                          int outer_dim_size, int scan_dim_size,
                                          bool reverse) {
  __shared__ T share_data[num_threads_y][num_threads_x * 2];
  T* share_row = share_data[threadIdx.y];
  int forward_direction = 1;
  if (reverse) forward_direction = -1;

  for (int block_row = blockIdx.x * blockDim.y; block_row < outer_dim_size;
       block_row += blockDim.y * gridDim.x) {
    int row = block_row + threadIdx.y;
    T acc = 0;
    const T* row_src = in + row * scan_dim_size;
    T* row_dst = out + row * scan_dim_size;
    int block_col = 0;
    bool loop_condition = (block_col < scan_dim_size);
    if (reverse) {
      loop_condition = (block_col >= 0);
      block_col = scan_dim_size - 1;
    }
    while (loop_condition) {
      // Load data into share memory(two value per thread)
      int col1 = block_col + threadIdx.x * forward_direction;
      int col2 = block_col + (num_threads_x + threadIdx.x) * forward_direction;
      if (row < outer_dim_size) {
        if (col1 < scan_dim_size && col1 >= 0) {
          share_row[threadIdx.x] = row_src[col1];
        } else {
          share_row[threadIdx.x] = 0;
        }

        if (col2 < scan_dim_size && col2 >= 0) {
          share_row[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          share_row[num_threads_x + threadIdx.x] = 0;
        }

        // Add the previous block acc to the result
        if (threadIdx.x == 0) {
          share_row[0] = share_row[0] + acc;
        }
      }
      __syncthreads();

      // Up-Sweep
      for (unsigned s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < outer_dim_size && threadIdx.x < s) {
          unsigned offset = (2 * threadIdx.x + 1) * d - 1;
          share_row[offset + d] = share_row[offset] + share_row[offset + d];
        }
        __syncthreads();
      }
      // Down-Sweep
      for (unsigned s = 2, d = blockDim.x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < outer_dim_size && threadIdx.x < s - 1) {
          unsigned offset = 2 * (threadIdx.x + 1) * d - 1;
          share_row[offset + d] = share_row[offset] + share_row[offset + d];
        }
        __syncthreads();
      }

      // Write to the output
      if (row < outer_dim_size) {
        if (col1 < scan_dim_size && col1 >= 0)
          row_dst[col1] = share_row[threadIdx.x];
        if (col2 < scan_dim_size && col2 >= 0)
          row_dst[col2] = share_row[num_threads_x + threadIdx.x];
      }
      acc = share_row[2 * num_threads_x - 1];
      __syncthreads();
      block_col += 2 * num_threads_x * forward_direction;
      if (reverse)
        loop_condition = (block_col >= 0);
      else
        loop_condition = (block_col < scan_dim_size);
    }
  }
}

// exclusive block scan and store block sum for large scan
template <typename T>
__global__ void InnerMostDimExclusiveScan(const T* in, T* out, T* sum_data,
                                          int inner_dim_size,
                                          int outer_dim_size, int scan_dim_size,
                                          int two_power, bool reverse) {
  // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
  extern __shared__ __align__(sizeof(T)) unsigned char raw_tmp[];
  T* share_tmp = reinterpret_cast<T*>(raw_tmp);
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  int block_scan_size = blockDim.x * 2;
  int remain = scan_dim_size % (2 * blockDim.x);
  if (block_id == gridDim.x - 1 && remain != 0) block_scan_size = remain;
  int col1 = thread_id;
  int col2 = thread_id + (block_scan_size) / 2;
  int index1 = blockIdx.y * (scan_dim_size) + block_id * blockDim.x * 2 + col1;
  int index2 = blockIdx.y * (scan_dim_size) + block_id * blockDim.x * 2 + col2;
  if (reverse) {
    index1 = blockIdx.y * (scan_dim_size) + scan_dim_size - 1 -
             (block_id * blockDim.x * 2 + col1);
    index2 = blockIdx.y * (scan_dim_size) + scan_dim_size - 1 -
             (block_id * blockDim.x * 2 + col2);
  }
  int sum_index = blockIdx.y * gridDim.x + block_id;
  if (thread_id < block_scan_size) {
    share_tmp[col1 + (col1 >> 5)] = in[index1];
    share_tmp[col2 + (col2 >> 5)] = in[index2];
  } else {
    share_tmp[col1 + (col1 >> 5)] = 0;
    share_tmp[col2 + (col2 >> 5)] = 0;
  }

  // Up-Sweep
  int offset = 1;
  for (int d = (two_power / 2); d > 0; d >>= 1) {
    __syncthreads();
    if (thread_id < d) {
      int tmp_index1 = offset * (2 * thread_id + 1) - 1;
      int tmp_index2 = offset * (2 * thread_id + 2) - 1;
      tmp_index1 = tmp_index1 + (tmp_index1 >> 5);
      tmp_index2 = tmp_index2 + (tmp_index2 >> 5);

      share_tmp[tmp_index2] += share_tmp[tmp_index1];
    }
    offset *= 2;
  }
  __syncthreads();

  if (thread_id == 0) {
    int tmp_index = (two_power - 1) + ((two_power - 1) >> 5);
    sum_data[sum_index] = share_tmp[tmp_index];
    share_tmp[tmp_index] = 0;
  }

  // Down Sweep
  for (int d = 1; d < two_power; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thread_id < d) {
      int tmp_index1 = offset * (2 * thread_id + 1) - 1;
      int tmp_index2 = offset * (2 * thread_id + 2) - 1;
      tmp_index1 = tmp_index1 + (tmp_index1 >> 5);
      tmp_index2 = tmp_index2 + (tmp_index2 >> 5);

      T tmp = share_tmp[tmp_index1];
      share_tmp[tmp_index1] = share_tmp[tmp_index2];
      share_tmp[tmp_index2] += tmp;
    }
  }

  __syncthreads();

  if (col1 < block_scan_size) out[index1] = share_tmp[col1 + (col1 >> 5)];
  if (col2 < block_scan_size) out[index2] = share_tmp[col2 + (col2 >> 5)];
}

// for large scan_dim_size array we need to add for correct result
template <typename T>
__global__ void AddBlockScan(T* result, T* sum, int size, int scan_dim_size,
                             int sum_size, bool reverse) {
  int idx = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
  int block_id_start = blockIdx.y * sum_size;
  int block_id_end = blockIdx.x + blockIdx.y * sum_size;
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  int col = block_id * blockDim.x + thread_id + size;
  int index = blockIdx.y * (scan_dim_size) + col;
  if (reverse) {
    index = blockIdx.y * (scan_dim_size) + scan_dim_size - 1 - col;
  }

  if (col >= scan_dim_size || col < 0) return;
  for (int i = block_id_start; i <= block_id_end; i++) {
    result[index] += sum[i];
  }
}

template <typename DeviceContext, typename T>
class CumCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    int axis = context.Attr<int>("axis");
    bool exclusive = context.Attr<bool>("exclusive");
    bool reverse = context.Attr<bool>("reverse");
    auto in_dims = in->dims();
    auto size = in->numel();

    if (axis == -1) {
      axis = in_dims.size() - 1;
    }
    PADDLE_ENFORCE_LT(
        axis, in_dims.size(),
        platform::errors::InvalidArgument("axis(%d) should be less than the "
                                          "dimension(%d) of the input tensor.",
                                          axis, in_dims.size()));

    int scan_dim_size = in_dims[axis];
    bool optimize_condition = (axis == (in_dims.size() - 1)) ? true : false;
    int outer_dim_size = 1;
    int inner_dim_size = 1;
    // treat all dim index < axis as outer_dim_size
    for (size_t i = 0; i < axis; i++) {
      outer_dim_size *= in_dims[i];
    }
    // treat all dim index > axis as innner_dim_size
    for (size_t i = axis + 1; i < in_dims.size(); i++) {
      inner_dim_size *= in_dims[i];
    }

    T* out_data = out->mutable_data<T>(context.GetPlace());
    const T* in_data = in->data<T>();

    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (optimize_condition) {
      auto nextPowerOfTwo = [](int x) -> int {
        int ret = 1;
        while (ret < x) ret = ret * 2;
        return ret;
      };
      if (exclusive) {
        int element_per_block = nextPowerOfTwo(scan_dim_size) / 2;
        if (element_per_block > 512 || element_per_block < 32) {
          element_per_block = 64;
        }
        int two_power = element_per_block * 2;
        dim3 block(element_per_block);
        dim3 grid(((scan_dim_size + 1) / 2 + block.x - 1) / block.x,
                  outer_dim_size);
        int offset_size = (element_per_block * 2) >> 5;
        int share_mem_size = (element_per_block * 2 + offset_size) * sizeof(T);
        Tensor scan_sum;
        paddle::framework::DDim dims{
            ((scan_dim_size + 1) / 2 + block.x - 1) / block.x, outer_dim_size};
        scan_sum.Resize(dims);
        T* sum_data = scan_sum.mutable_data<T>(context.GetPlace());
        InnerMostDimExclusiveScan<
            T><<<grid, block, share_mem_size, dev_ctx.stream()>>>(
            in_data, out_data, sum_data, inner_dim_size, outer_dim_size,
            scan_dim_size, two_power, reverse);
        // for large scan array we need to do add for correct result
        int element_size = element_per_block * 2;
        if (scan_dim_size > element_size) {
          dim3 sum_block(element_per_block * 2);
          dim3 sum_grid((scan_dim_size - element_size + block.x - 1) / block.x,
                        outer_dim_size);
          int sum_size = ((scan_dim_size + 1) / 2 + block.x - 1) / block.x;
          AddBlockScan<T><<<sum_grid, sum_block, 0, dev_ctx.stream()>>>(
              out_data, sum_data, element_size, scan_dim_size, sum_size,
              reverse);
        }

      } else {
        dim3 block(32, 16);
        dim3 grid((outer_dim_size + block.y - 1) / block.y);
        InnerMostDimInclusiveScan<T, 32,
                                  16><<<grid, block, 0, dev_ctx.stream()>>>(
            in_data, out_data, inner_dim_size, outer_dim_size, scan_dim_size,
            reverse);
      }
    } else {
      dim3 block(std::min(512, inner_dim_size));
      dim3 grid(outer_dim_size, (inner_dim_size + block.x - 1) / block.x);
      OuterScan<T><<<grid, block, 0, dev_ctx.stream()>>>(
          in_data, out_data, inner_dim_size, outer_dim_size, scan_dim_size,
          exclusive, reverse);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cumsum, ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
