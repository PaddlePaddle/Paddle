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

#include <glog/logging.h>
#include "paddle/fluid/inference/tensorrt/plugin/elementwise_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

namespace details {

template <typename T>
struct Add {
  __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct Mul {
  __device__ T operator()(const T& a, const T& b) const { return a * b; }
};

template <typename T, typename Operator>
__global__ void ColumnWiseKernel(Operator op, const T* x, const T* y, T* out,
                                 int batch_size, int num_rows, int num_cols) {
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    int row = blockIdx.x;
    for (; row < num_rows; row += gridDim.x) {
      T value_y = y[batch_id * num_rows + row];
      int col = threadIdx.x;
      int offset = (batch_id * num_rows + row) * num_cols;
      for (; col < num_cols; col += blockDim.x) {
        T value_x = x[offset + col];
        out[offset + col] = op(value_x, value_y);
      }
    }
  }
}

template <typename T, typename Operator>
__global__ void RowWiseKernel(Operator op, const T* x, const T* y, T* out,
                              int num_ele, int num_cols) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_ele) {
    int offset = index % num_cols;
    out[index] = op(x[index], y[offset]);
  }
}

template <typename T, typename Operator>
static void ElementWise(Operator op, const T* x, const T* y, T* out,
                        int batch_size, int prev, int midd, int post,
                        cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int kMaximumBlocks = 65535;
  if (prev == 1) {
    int num_threads = (post > kThreadsPerBlock) ? kThreadsPerBlock
                                                : (((post + 31) >> 5) << 5);
    int num_blocks = (midd < kMaximumBlocks) ? midd : kMaximumBlocks;
    ColumnWiseKernel<<<num_blocks, num_threads, 0, stream>>>(
        op, x, y, out, batch_size, midd, post);
  } else if (post == 1) {
    int sum_ele = batch_size * prev * midd;
    int num_blocks = (sum_ele + 1023) / 1024;
    PADDLE_ENFORCE_LT(num_blocks, kMaximumBlocks);
    RowWiseKernel<<<num_blocks, 1024, 0, stream>>>(op, x, y, out, sum_ele,
                                                   midd);
  } else {
    PADDLE_THROW("Not implemented.");
  }
}

}  // namespace details

nvinfer1::Dims ElementWisePlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* input_dims, int num_inputs) {
  PADDLE_ENFORCE_EQ(index, 0);
  // PADDLE_ENFORCE_EQ(num_inputs, 2);
  PADDLE_ENFORCE_NOT_NULL(input_dims);
  return input_dims[0];
}

int ElementWisePlugin::initialize() {
  PADDLE_ENFORCE_GT(shape_y_.size(), 0);

  axis_ = (axis_ == -1) ? shape_x_.size() - shape_y_.size() : axis_;
  int trimed_nb_dims = shape_y_.size();
  for (; trimed_nb_dims > 0; --trimed_nb_dims) {
    if (shape_y_[trimed_nb_dims - 1] != 1) {
      break;
    }
  }

  PADDLE_ENFORCE_GE(shape_x_.size(), trimed_nb_dims + axis_);
  PADDLE_ENFORCE_LT(axis_, shape_x_.size());

  prev_size_ = 1;
  midd_size_ = 1;
  post_size_ = 1;
  for (int i = 0; i < axis_; ++i) {
    prev_size_ *= shape_x_[i];
  }

  for (int i = 0; i < trimed_nb_dims; ++i) {
    PADDLE_ENFORCE_EQ(shape_x_[i + axis_], shape_y_[i],
                      "Broadcast dimension mismatch.");
    midd_size_ *= shape_y_[i];
  }

  for (int i = axis_ + trimed_nb_dims; i < shape_x_.size(); ++i) {
    post_size_ *= shape_x_[i];
  }
  return 0;
}

int ElementWisePlugin::enqueue(int batch_size, const void* const* inputs,
                               void** outputs, void* workspace,
                               cudaStream_t stream) {
  const float* x = reinterpret_cast<const float*>(inputs[0]);
  const float* y = nullptr;
  if (!with_weights_) {
    y = reinterpret_cast<const float*>(inputs[1]);
  } else {
    y = reinterpret_cast<const float*>(weights_.get().values);
  }

  float* out = reinterpret_cast<float*>(outputs[0]);

  if (type_ == "add") {
    details::ElementWise(details::Add<float>(), x, y, out, batch_size,
                         prev_size_, midd_size_, post_size_, stream);
  } else if (type_ == "mul") {
    details::ElementWise(details::Mul<float>(), x, y, out, batch_size,
                         prev_size_, midd_size_, post_size_, stream);

  } else {
    PADDLE_THROW("Not implemented.");
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
