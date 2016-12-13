/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hl_base.h"
#include "cross_map_normal_op.h"

namespace paddle {

__global__ void KeCMRNormFillScale(size_t imageSize, const real* in,
                                   real* scale, size_t channels,
                                   size_t height, size_t width, size_t size,
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

__global__ void KeCMRNormOutput(size_t inputSize, const real* in,
                                const real* scale, real negative_beta,
                                real* out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < inputSize) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <>
void CrossMapNormal<DEVICE_TYPE_GPU>::operator()(GpuMatrix& outputs,
                                                 GpuMatrix& denoms,
                                                 GpuMatrix& inputs,
                                                 size_t channels,
                                                 size_t imgSizeH,
                                                 size_t imgSizeW,
                                                 size_t sizeX,
                                                 real scale,
                                                 real pow) {
  CHECK(outputs.isContiguous());
  CHECK(inputs.isContiguous());
  CHECK(denoms.isContiguous());
  CHECK_EQ(outputs.getHeight(), inputs.getHeight());
  CHECK_EQ(outputs.getWidth(), inputs.getWidth());
  CHECK_EQ(outputs.getHeight(), denoms.getHeight());
  CHECK_EQ(outputs.getWidth(), denoms.getWidth());

  size_t numSample = inputs.getHeight();
  size_t numCols = inputs.getWidth();
  CHECK(imgSizeH * imgSizeW * channels == numCols);

  real* inputsData = inputs.getData();
  real* denomsData = denoms.getData();
  real* outputsData = outputs.getData();

  size_t imageSize = numSample * imgSizeH * imgSizeW;
  int blockSize = 1024;
  int gridSize = (imageSize + 1024 - 1) / 1024;
  KeCMRNormFillScale<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
    (imageSize, inputsData, denomsData,
    channels, imgSizeH, imgSizeW, sizeX, scale);

  size_t inputSize = numSample * imgSizeH * imgSizeW *channels;
  blockSize = 1024;
  gridSize = (inputSize + 1024 - 1) / 1024;
  KeCMRNormOutput<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
    (inputSize, inputsData, denomsData, -pow, outputsData);

  CHECK_SYNC("CrossMapNormalFwd");
}

__global__ void KeCMRNormDiff(size_t imageSize, const real* bottom_data,
                              const real* top_data, const real* scale,
                              const real* top_diff, size_t channels,
                              size_t height, size_t width, size_t size,
                              real negative_beta, real cache_ratio,
                              real* bottom_diff ) {
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
          pow(scale[(index - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(index - post_pad) * step] * accum;
      }
      ++index;
    }
  }
}

template <>
void CrossMapNormalGrad<DEVICE_TYPE_GPU>::operator()(GpuMatrix& inputsGrad,
                                                     GpuMatrix& inputsValue,
                                                     GpuMatrix& outputsGrad,
                                                     GpuMatrix& outputsValue,
                                                     GpuMatrix& denoms,
                                                     size_t channels,
                                                     size_t imgSizeH,
                                                     size_t imgSizeW,
                                                     size_t sizeX,
                                                     real scale,
                                                     real pow) {
  CHECK(inputsGrad.isContiguous());
  CHECK(outputsGrad.isContiguous());
  CHECK(denoms.isContiguous());
  CHECK(inputsValue.isContiguous());
  CHECK(outputsValue.isContiguous());
  CHECK_EQ(inputsGrad.getHeight(), outputsGrad.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), outputsGrad.getWidth());
  CHECK_EQ(inputsGrad.getHeight(), denoms.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), denoms.getWidth());
  CHECK_EQ(inputsGrad.getHeight(), inputsValue.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), inputsValue.getWidth());
  CHECK_EQ(inputsGrad.getHeight(), outputsValue.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), outputsValue.getWidth());

  size_t numSample = inputsGrad.getHeight();
  size_t numCols = inputsGrad.getWidth();
  CHECK(imgSizeH * imgSizeW * channels == numCols);

  size_t imageSize = numSample * imgSizeH * imgSizeW;
  real* inputsGradData = inputsGrad.getData();
  real* inputsData = inputsValue.getData();
  real* denomsData = denoms.getData();
  real* outputsGradData = outputsGrad.getData();
  real* outputsData = outputsValue.getData();

  int blockSize = 1024;
  int gridSize = (imageSize + 1024 - 1) / 1024;
  KeCMRNormDiff <<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
    (imageSize, inputsData, outputsData, denomsData, outputsGradData, channels,
      imgSizeH, imgSizeW, sizeX, -pow, 2.0f * pow * scale, inputsGradData);
  CHECK_SYNC("KeCMRNormDiff");
}

}  // namespace paddle
