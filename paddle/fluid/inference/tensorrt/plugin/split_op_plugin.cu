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

#include <cassert>
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

SplitPlugin* CreateSplitPlugin() { return new SplitPlugin(); };

nvinfer1::Dims SplitPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims* inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const& input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  output_dims.d[axis_] = output_lenght_.at(index);
  return output_dims;
}

int SplitPlugin::initialize() {
  std::vector<int> segment_offsets(1, 0);
  for (int i = 0; i < this->getNbOutputs(); ++i) {
    segment_offsets.push_back(segment_offsets.back() + output_lenght_[i]);
  }
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

template <typename T>
__global__ void split_kernel(int nsegment,
                             int const* __restrict__ segment_offsets,
                             T const* __restrict__ idata, T* const* odatas,
                             int nx, int srcny_, int nz) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int src_y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = threadIdx.z + blockIdx.z * blockDim.z;
  for (int z = z0; z < nz; z += blockDim.z * gridDim.z) {
    for (int src_y = src_y0; src_y < srcny_; src_y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nx; x += blockDim.x * gridDim.x) {
        int segment = upper_bound(segment_offsets, nsegment, src_y) - 1;
        int dst_y = src_y - segment_offsets[segment];
        int dstny_ = segment_offsets[segment + 1] - segment_offsets[segment];
        odatas[segment][x + nx * (dst_y + dstny_ * z)] =
            idata[x + nx * (src_y + srcny_ * z)];
      }
    }
  }
}

int SplitPlugin::enqueue(int batchSize, const void* const* inputs,
                         void** outputs, void* workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int const* d_segment_offsets_ptr =
      thrust::raw_pointer_cast(&d_segment_offsets_[0]);
  float const* idata = reinterpret_cast<float const*>(inputs[0]);
  float** odatas = reinterpret_cast<float**>(outputs);

  int nz = nz_ * batchSize;
  dim3 block(32, 16);
  dim3 grid(std::min((nx_ - 1) / block.x + 1, 65535u),
            std::min((ny_ - 1) / block.y + 1, 65535u),
            std::min((nz_ - 1) / block.z + 1, 65535u));

  split_kernel<<<grid, block, 0, stream>>>(d_segment_offsets_.size(),
                                           d_segment_offsets_ptr, idata, odatas,
                                           nx_, ny_, nz);

  return cudaGetLastError() != cudaSuccess;
}

}  // tensorrt
}  // inference
}  // paddle
