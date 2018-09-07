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

#include "CrossMapNormalOp.h"
#include "hl_base.h"

namespace paddle {

__global__ void KeCMRNormFillScale(size_t imageSize,
                                   const real* in,
                                   real* scale,
                                   size_t channels,
                                   size_t height,
                                   size_t width,
                                   size_t size,
                                   real alpha) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < imageSize) {
    const int w = idx % width;
    const int h = (idx / width) % height;
    const int n = idx / width / height;
    const int offset = (n * channels * height + h) * width + w;

    in += offset;
    scale += offset;
    const int step = height * width;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;

    real accum = 0;
    int index = 0;
    while (index < channels + post_pad) {
      if (index < channels) {
        accum += in[index * step] * in[index * step];
      }
      if (index >= size) {
        accum -= in[(index - size) * step] * in[(index - size) * step];
      }
      if (index >= post_pad) {
        scale[(index - post_pad) * step] = 1. + accum * alpha;
      }
      ++index;
    }
  }
}

__global__ void KeCMRNormOutput(size_t inputSize,
                                const real* in,
                                const real* scale,
                                real negative_beta,
                                real* out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < inputSize) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <>
void CrossMapNormal<DEVICE_TYPE_GPU>(real* outputs,
                                     real* denoms,
                                     const real* inputs,
                                     size_t numSamples,
                                     size_t channels,
                                     size_t height,
                                     size_t width,
                                     size_t size,
                                     real scale,
                                     real pow) {
  size_t imageSize = numSamples * height * width;
  int blockSize = 1024;
  int gridSize = (imageSize + 1024 - 1) / 1024;
  KeCMRNormFillScale<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
      imageSize, inputs, denoms, channels, height, width, size, scale);

  size_t inputSize = numSamples * height * width * channels;
  blockSize = 1024;
  gridSize = (inputSize + 1024 - 1) / 1024;
  KeCMRNormOutput<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
      inputSize, inputs, denoms, -pow, outputs);

  CHECK_SYNC("CrossMapNormal");
}

__global__ void KeCMRNormDiff(size_t imageSize,
                              const real* bottom_data,
                              const real* top_data,
                              const real* scale,
                              const real* top_diff,
                              size_t channels,
                              size_t height,
                              size_t width,
                              size_t size,
                              real negative_beta,
                              real cache_ratio,
                              real* bottom_diff) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < imageSize) {
    const int w = idx % width;
    const int h = (idx / width) % height;
    const int n = idx / width / height;
    const int offset = (n * channels * height + h) * width + w;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;

    const int step = height * width;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;

    int index = 0;
    real accum = 0;
    while (index < channels + post_pad) {
      if (index < channels) {
        accum += top_diff[index * step] * top_data[index * step] /
                 scale[index * step];
      }
      if (index >= size) {
        accum -= top_diff[(index - size) * step] *
                 top_data[(index - size) * step] / scale[(index - size) * step];
      }
      if (index >= post_pad) {
        bottom_diff[(index - post_pad) * step] +=
            top_diff[(index - post_pad) * step] *
                pow(scale[(index - post_pad) * step], negative_beta) -
            cache_ratio * bottom_data[(index - post_pad) * step] * accum;
      }
      ++index;
    }
  }
}

template <>
void CrossMapNormalGrad<DEVICE_TYPE_GPU>(real* inputsGrad,
                                         const real* inputsValue,
                                         const real* outputsValue,
                                         const real* outputsGrad,
                                         const real* denoms,
                                         size_t numSamples,
                                         size_t channels,
                                         size_t height,
                                         size_t width,
                                         size_t size,
                                         real scale,
                                         real pow) {
  size_t imageSize = numSamples * height * width;

  int blockSize = 1024;
  int gridSize = (imageSize + 1024 - 1) / 1024;
  KeCMRNormDiff<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(imageSize,
                                                            inputsValue,
                                                            outputsValue,
                                                            denoms,
                                                            outputsGrad,
                                                            channels,
                                                            height,
                                                            width,
                                                            size,
                                                            -pow,
                                                            2.0f * pow * scale,
                                                            inputsGrad);
  CHECK_SYNC("CrossMapNormalGrad");
}

}  // namespace paddle
