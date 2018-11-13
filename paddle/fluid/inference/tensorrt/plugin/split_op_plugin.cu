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

#include <stdio.h>
#include <cassert>
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

nvinfer1::Dims SplitPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims* inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const& input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  output_dims.d[axis_] = output_length_.at(index);
  return output_dims;
}

int SplitPlugin::initialize() {
  std::vector<int> segment_offsets(1, 0);
  for (int i = 0; i < this->getNbOutputs(); ++i) {
    segment_offsets.push_back(segment_offsets.back() + output_length_[i]);
  }
  segment_offsets_ = segment_offsets;
  d_segment_offsets_ = segment_offsets;
  nvinfer1::Dims dims = this->getInputDims(0);
  nx_ = 1;
  for (int i = dims.nbDims - 1; i > axis_; --i) {
    nx_ *= dims.d[i];
  }
  ny_ = dims.d[axis_];
  nz_ = 1;
  for (int i = axis_ - 1; i >= 0; --i) {
    nz_ *= dims.d[i];
  }
  return 0;
}

int SplitPlugin::enqueue(int batchSize, const void* const* inputs,
                         void** outputs, void* workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int input_size = 0;
  int const* d_segment_offsets_ptr =
      thrust::raw_pointer_cast(&d_segment_offsets_[0]);
  float const* idata = reinterpret_cast<float const*>(inputs[0]);
  float** odatas = reinterpret_cast<float**>(outputs);

  // kernel impl here.
  int inputBatchOffset = nx_ * ny_ * nz_;
  for (size_t i = 0; i < this->getNbOutputs(); i++) {
    for (size_t j = 0; j < batchSize; j++) {
      cudaMemcpyAsync(
          odatas[i] +
              j * (segment_offsets_[i + 1] - segment_offsets_[i]) * nx_ *
                  sizeof(float),
          inputs[0] +
              (inputBatchOffset * j + segment_offsets_[i] * nx_) *
                  sizeof(float),
          (segment_offsets_[i + 1] - segment_offsets_[i]) * nx_ * sizeof(float),
          cudaMemcpyDeviceToDevice, stream);
    }
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // tensorrt
}  // inference
}  // paddle
