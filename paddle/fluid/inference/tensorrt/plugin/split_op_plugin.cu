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

// copied from operators::math::SplitFunctor
template <typename T>
__global__ void SplitKernel(const T* input_data, const int in_row,
                            const int in_col, const int* out_cols,
                            int out_cols_size, T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_segment = 0;
  int curr_offset = out_cols[0];
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset = out_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = out_cols[curr_segment + 1];
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs_data[curr_segment];
    if (output_ptr != nullptr) {
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * segment_width + local_col] =
            input_data[tid_y * in_col + tid_x];
    }
  }
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int in_row,
                            const int in_col, const int fixed_out_col,
                            T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x / fixed_out_col;
    int in_offset = tid_x - split * fixed_out_col;
    T* output_ptr = outputs_data[split];
    if (output_ptr != nullptr) {
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * fixed_out_col + in_offset] =
            input_data[tid_y * in_col + tid_x];
    }
  }
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
    segment_offsets.push_back(segment_offsets.back() +
                              output_length_[i] * inner_cols_);
  }
  inner_cols_ *= dims.d[axis_];
  d_segment_offsets_ = segment_offsets;
  segment_offsets_ = std::move(segment_offsets);
  d_output_ptrs_.resize(this->getNbOutputs(), nullptr);
  return 0;
}

template <typename T>
inline void Split(cudaStream_t stream, const bool same_shape,
                  const int outer_rows, const int inner_cols,
                  const std::vector<int>& segment_offsets,
                  const int* d_segment_offsets, const T* input, T** outputs) {
  const int kThreadsPerBlock = 1024;
  const int kMaxBlocks = 65535;
  int block_cols = kThreadsPerBlock;
  if (inner_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
    block_cols = ((inner_cols + 31) >> 5) << 5;
  }
  int block_rows = kThreadsPerBlock / block_cols;
  dim3 block_size = dim3(block_cols, block_rows, 1);

  int grid_cols =
      std::min((inner_cols + block_cols - 1) / block_cols, kMaxBlocks);
  int grid_rows =
      std::min(kMaxBlocks / grid_cols, std::max(outer_rows / block_rows, 1));
  dim3 grid_size = dim3(grid_cols, grid_rows, 1);

  if (same_shape) {
    SplitKernel<<<grid_size, block_size, 0, stream>>>(
        input, outer_rows, inner_cols, segment_offsets[1], outputs);
  } else {
    SplitKernel<<<grid_size, block_size, 0, stream>>>(
        input, outer_rows, inner_cols, d_segment_offsets,
        static_cast<int>(segment_offsets.size()), outputs);
  }
}

int SplitPlugin::enqueue(int batchSize, const void* const* inputs,
                         void** outputs, void* workspace, cudaStream_t stream) {
  float const* input_ptr = reinterpret_cast<float const*>(inputs[0]);
  if (((batchSize == 1 && axis_ == 0) || axis_ == -1) &&
      this->getNbOutputs() < 10) {
    float** output_ptrs = reinterpret_cast<float**>(outputs);
    int data_type_size = (this->getDataType() == nvinfer1::DataType::kFLOAT)
                             ? sizeof(float)
                             : sizeof(__half);
    for (int i = 0; i < this->getNbOutputs(); ++i) {
      PADDLE_ENFORCE(
          cudaMemcpyAsync(
              output_ptrs[i], input_ptr + segment_offsets_[i],
              (segment_offsets_[i + 1] - segment_offsets_[i]) * data_type_size,
              cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
    }
  } else {
    outer_rows_ *= batchSize;
    const int* d_segment_offsets_ptr =
        thrust::raw_pointer_cast(&d_segment_offsets_[0]);
    float** output_ptrs = thrust::raw_pointer_cast(&d_output_ptrs_[0]);
    PADDLE_ENFORCE(cudaMemcpyAsync(output_ptrs, outputs,
                                   this->getNbOutputs() * sizeof(float*),
                                   cudaMemcpyHostToDevice,
                                   stream) == cudaSuccess);
    if (this->getDataType() == nvinfer1::DataType::kFLOAT) {
      Split(stream, same_shape_, outer_rows_, inner_cols_, segment_offsets_,
            d_segment_offsets_ptr, input_ptr, output_ptrs);
    } else {
      Split(stream, same_shape_, outer_rows_, inner_cols_, segment_offsets_,
            d_segment_offsets_ptr, (__half*)input_ptr,  // NOLINT
            (__half**)output_ptrs);                     // NOLINT
    }
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
