// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_fp16.h>
#include <algorithm>
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

SplitPlugin* CreateSplitPluginDeserialize(const void* buffer, size_t length) {
  return new SplitPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("split_plugin", CreateSplitPluginDeserialize);

template <typename T>
__device__ int upper_bound(T const* vals, int n, T const& key) {
  int i = 0;
  while (n > 0) {
    int m = n / 2;
    int j = i + m;
    if (!(key < vals[j])) {
      i = j + 1;
      n -= m + 1;
    } else {
      n = m;
    }
  }
  return i;
}

nvinfer1::Dims SplitPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* input_dims, int num_inputs) {
  PADDLE_ENFORCE_EQ(num_inputs, 1);
  PADDLE_ENFORCE_LT(index, this->getNbOutputs());

  nvinfer1::Dims output_dims = input_dims[0];
  output_dims.d[axis_] = output_length_.at(index);
  return output_dims;
}

int SplitPlugin::initialize() {
  PADDLE_ENFORCE_LE(axis_, nvinfer1::Dims::MAX_DIMS);
  // notice input dims is [C, H, W]
  nvinfer1::Dims dims = this->getInputDims(0);
  outer_rows_ = 1;
  inner_cols_ = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_rows_ *= dims.d[i];
  }
  for (int i = axis_ + 1; i < dims.nbDims; ++i) {
    inner_cols_ *= dims.d[i];
  }
  same_shape_ = true;
  std::vector<int> segment_offsets(1, 0);
  for (int i = 0; i < this->getNbOutputs(); ++i) {
    if (output_length_[i] != output_length_[0]) {
      same_shape_ = false;
    }
    segment_offsets.push_back(segment_offsets.back() + output_length_[i]);
  }
  axis_shape_ = dims.d[axis_];
  d_segment_offsets_ = segment_offsets;
  segment_offsets_ = std::move(segment_offsets);
  d_output_ptrs_.resize(this->getNbOutputs(), nullptr);
  return 0;
}

// The following part of the code refers to onnx-tensorrt
// https://github.com/onnx/onnx-tensorrt/blob/master/Split.cu
template <typename T>
__global__ void split_kernel(int nsegment,
                             int const* __restrict__ segment_offsets,
                             T const* __restrict__ idata, T* const* odatas,
                             int inner_cols, int axis_shape, int outer_rows) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int src_y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = threadIdx.z + blockIdx.z * blockDim.z;
  for (int z = z0; z < outer_rows; z += blockDim.z * gridDim.z) {
    for (int src_y = src_y0; src_y < axis_shape;
         src_y += blockDim.y * gridDim.y) {
      for (int x = x0; x < inner_cols; x += blockDim.x * gridDim.x) {
        int segment = upper_bound(segment_offsets, nsegment, src_y) - 1;
        int dst_y = src_y - segment_offsets[segment];
        int dst_ny = segment_offsets[segment + 1] - segment_offsets[segment];
        odatas[segment][x + inner_cols * (dst_y + dst_ny * z)] =
            idata[x + inner_cols * (src_y + axis_shape * z)];
      }
    }
  }
}

int SplitPlugin::enqueue(int batchSize, const void* const* inputs,
                         void** outputs, void* workspace, cudaStream_t stream) {
  const int* d_segment_offsets_ptr =
      thrust::raw_pointer_cast(&d_segment_offsets_[0]);
  float const* input_ptr = reinterpret_cast<float const*>(inputs[0]);
  float* const* h_odatas = reinterpret_cast<float* const*>(outputs);
  float** output_ptrs = thrust::raw_pointer_cast(&d_output_ptrs_[0]);
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpyAsync(
      output_ptrs, h_odatas, d_output_ptrs_.size() * sizeof(float*),
      cudaMemcpyHostToDevice, stream));

  int outer_rows = outer_rows_ * batchSize;

  dim3 block(32, 16);
  dim3 grid(std::min((inner_cols_ - 1) / block.x + 1, 65535u),
            std::min((axis_shape_ - 1) / block.y + 1, 65535u),
            std::min((outer_rows_ - 1) / block.z + 1, 65535u));

  split_kernel<<<grid, block, 0, stream>>>(
      d_segment_offsets_.size(), d_segment_offsets_ptr, input_ptr, output_ptrs,
      inner_cols_, axis_shape_, outer_rows);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
