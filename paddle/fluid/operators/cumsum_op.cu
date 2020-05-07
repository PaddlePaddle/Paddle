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

// exclusive scan
template <typename T, int num_threads_x, int num_threads_y>
__global__ void InnerMostDimScan(const T* in, T* out, int inner_dim_size,
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
    optimize_condition = optimize_condition && !exclusive;
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
      dim3 block(32, 16);
      dim3 grid((outer_dim_size + block.y - 1) / block.y);
      InnerMostDimScan<T, 32, 16><<<grid, block, 0, dev_ctx.stream()>>>(
          in_data, out_data, inner_dim_size, outer_dim_size, scan_dim_size,
          reverse);
    } else {
      dim3 block(std::min(512, inner_dim_size));
      dim3 grid(outer_dim_size, (inner_dim_size + block.x - 1) / block.x);
      OuterScan<T><<<block, grid, 0, dev_ctx.stream()>>>(
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
