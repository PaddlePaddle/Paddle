/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "ScaleSubRegionOp.h"
#include "hl_base.h"

namespace paddle {

__global__ void KeScaleSubRegion(real* outputs,
                                 const real* inputs,
                                 const real* indices,
                                 real value,
                                 int channel,
                                 int height,
                                 int width,
                                 int nthreads) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nthreads) {
    const int w = idx % width;
    const int h = (idx / width) % height;
    const int c = (idx / width / height) % channel;
    const int n = idx / width / height / channel;

    const int offset = n * 6;
    if (c >= (indices[offset] - 1) && c <= (indices[offset + 1] - 1) &&
        h >= (indices[offset + 2] - 1) && h <= (indices[offset + 3] - 1) &&
        w >= (indices[offset + 4] - 1) && w <= (indices[offset + 5] - 1)) {
      outputs[idx] = inputs[idx] * value;
    } else {
      outputs[idx] = inputs[idx];
    }
  }
}

template <>
void ScaleSubRegion<DEVICE_TYPE_GPU>(real* outputs,
                                     const real* inputs,
                                     const real* indices,
                                     const TensorShape shape,
                                     const FuncConfig& conf) {
  real value = conf.get<real>("value");

  int number = shape[0];
  int channel = shape[1];
  int height = shape[2];
  int width = shape[3];

  size_t nth = number * channel * height * width;
  int blockSize = 1024;
  int gridSize = (nth + blockSize - 1) / blockSize;

  KeScaleSubRegion<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
      outputs, inputs, indices, value, channel, height, width, nth);
  CHECK_SYNC("ScaleSubRegion");
}

__global__ void KeScaleSubRegionDiff(const real* inGrad,
                                     real* outGrad,
                                     const real* indices,
                                     real value,
                                     int channel,
                                     int height,
                                     int width,
                                     int nthreads) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nthreads) {
    const int w = idx % width;
    const int h = (idx / width) % height;
    const int c = (idx / width / height) % channel;
    const int n = idx / width / height / channel;

    const int offset = n * 6;
    if (c >= (indices[offset] - 1) && c <= (indices[offset + 1] - 1) &&
        h >= (indices[offset + 2] - 1) && h <= (indices[offset + 3] - 1) &&
        w >= (indices[offset + 4] - 1) && w <= (indices[offset + 5] - 1)) {
      outGrad[idx] += inGrad[idx] * value;
    } else {
      outGrad[idx] += inGrad[idx];
    }
  }
}

template <>
void ScaleSubRegionGrad<DEVICE_TYPE_GPU>(const real* inGrad,
                                         real* outGrad,
                                         const real* indices,
                                         const TensorShape shape,
                                         const FuncConfig& conf) {
  real value = conf.get<real>("value");

  int number = shape[0];
  int channel = shape[1];
  int height = shape[2];
  int width = shape[3];

  size_t nth = number * channel * height * width;
  int blockSize = 1024;
  int gridSize = (nth + blockSize - 1) / blockSize;

  KeScaleSubRegionDiff<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
      inGrad, outGrad, indices, value, channel, height, width, nth);
  CHECK_SYNC("ScaleSubRegionGrad");
}

}  // namespace paddle
