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

#include "DepthwiseConvOp.h"
#include "paddle/math/BaseMatrix.h"

namespace paddle {

// CUDA kernel to compute the depthwise convolution forward pass
template <class T>
__global__ void ConvolutionDepthwiseForward(const int nthreads,
                                            const T* const inputData,
                                            const T* const filterData,
                                            const int batchSize,
                                            const int outputChannels,
                                            const int outputHeight,
                                            const int outputWidth,
                                            const int inputChannels,
                                            const int inputHeight,
                                            const int inputWidth,
                                            const int filterMultiplier,
                                            const int filterHeight,
                                            const int filterWidth,
                                            const int strideH,
                                            const int strideW,
                                            const int paddingH,
                                            const int paddingW,
                                            T* const outputData) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

  if (index < nthreads) {
    const int batch = index / outputChannels / outputHeight / outputWidth;
    const int c_out = (index / outputHeight / outputWidth) % outputChannels;
    const int h_out = (index / outputWidth) % outputHeight;
    const int w_out = index % outputWidth;

    const int c_in = c_out / filterMultiplier;
    const T* weight = filterData + c_out * filterHeight * filterWidth;
    T value = 0;
    const int h_in_start = -paddingH + h_out * strideH;
    const int w_in_start = -paddingW + w_out * strideW;
    const int h_in_end = -paddingH + h_out * strideH + filterHeight - 1;
    const int w_in_end = -paddingW + w_out * strideW + filterWidth - 1;
    if ((h_in_start >= 0) && (h_in_end < inputHeight) && (w_in_start >= 0) &&
        (w_in_end < inputWidth)) {
      for (int kh = 0; kh < filterHeight; ++kh) {
        for (int kw = 0; kw < filterWidth; ++kw) {
          const int h_in = -paddingH + h_out * strideH + kh;
          const int w_in = -paddingW + w_out * strideW + kw;
          const int offset =
              ((batch * inputChannels + c_in) * inputHeight + h_in) *
                  inputWidth +
              w_in;
          value += (*weight) * inputData[offset];
          ++weight;
        }
      }
    } else {
      for (int kh = 0; kh < filterHeight; ++kh) {
        for (int kw = 0; kw < filterWidth; ++kw) {
          const int h_in = -paddingH + h_out * strideH + kh;
          const int w_in = -paddingW + w_out * strideW + kw;
          if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) &&
              (w_in < inputWidth)) {
            const int offset =
                ((batch * inputChannels + c_in) * inputHeight + h_in) *
                    inputWidth +
                w_in;
            value += (*weight) * inputData[offset];
          }
          ++weight;
        }
      }
    }
    outputData[index] = value;
  }
}

// CUDA kernel to compute the depthwise convolution backprop w.r.t input.
template <class T>
__global__ void ConvolutionDepthwiseInputBackward(const int nthreads,
                                                  const T* const top_diff,
                                                  const T* const weight_data,
                                                  const int num,
                                                  const int outputChannels,
                                                  const int outputHeight,
                                                  const int outputWidth,
                                                  const int inputChannels,
                                                  const int inputHeight,
                                                  const int inputWidth,
                                                  const int filterMultiplier,
                                                  const int filterHeight,
                                                  const int filterWidth,
                                                  const int strideH,
                                                  const int strideW,
                                                  const int paddingH,
                                                  const int paddingW,
                                                  T* const bottom_diff) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    const int batch = index / inputChannels / inputHeight / inputWidth;
    const int c_in = (index / inputHeight / inputWidth) % inputChannels;
    const int h_in = (index / inputWidth) % inputHeight;
    const int w_in = index % inputWidth;

    const int c_out_start = c_in * filterMultiplier;

    int h_out_start = (h_in - filterHeight + paddingH + strideH) / strideH;
    h_out_start = 0 > h_out_start ? 0 : h_out_start;
    int h_out_end = (h_in + paddingH) / strideH;
    h_out_end = outputHeight - 1 < h_out_end ? outputHeight - 1 : h_out_end;
    int w_out_start = (w_in - filterWidth + paddingW + strideW) / strideW;
    w_out_start = 0 > w_out_start ? 0 : w_out_start;
    int w_out_end = (w_in + paddingW) / strideW;
    w_out_end = outputWidth - 1 < w_out_end ? outputWidth - 1 : w_out_end;

    T value = 0;

    for (int c_out = c_out_start; c_out < c_out_start + filterMultiplier;
         c_out++) {
      for (int h_out = h_out_start; h_out <= h_out_end; ++h_out) {
        const int filter_h = h_in + paddingH - h_out * strideH;
        for (int w_out = w_out_start; w_out <= w_out_end; ++w_out) {
          const int filter_w = w_in + paddingW - w_out * strideW;
          const int filter_offset = c_out * filterHeight * filterWidth +
                                    filter_h * filterWidth + filter_w;
          const int top_diff_offset =
              ((batch * outputChannels + c_out) * outputHeight + h_out) *
                  outputWidth +
              w_out;
          value += top_diff[top_diff_offset] * weight_data[filter_offset];
        }
      }
    }
    bottom_diff[index] += value;
  }
}

// CUDA kernel to compute the depthwise convolution backprop w.r.t filter.
template <class T>
__global__ void ConvolutionDepthwiseFilterBackward(const int num_i,
                                                   const int nthreads,
                                                   const T* const top_diff,
                                                   const T* const inputData,
                                                   const int num,
                                                   const int outputChannels,
                                                   const int outputHeight,
                                                   const int outputWidth,
                                                   const int inputChannels,
                                                   const int inputHeight,
                                                   const int inputWidth,
                                                   const int filterMultiplier,
                                                   const int filterHeight,
                                                   const int filterWidth,
                                                   const int strideH,
                                                   const int strideW,
                                                   const int paddingH,
                                                   const int paddingW,
                                                   T* const buffer_data) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    const int h_out = (index / outputWidth) % outputHeight;
    const int w_out = index % outputWidth;
    const int kh =
        (index / filterWidth / outputHeight / outputWidth) % filterHeight;
    const int kw = (index / outputHeight / outputWidth) % filterWidth;
    const int h_in = -paddingH + h_out * strideH + kh;
    const int w_in = -paddingW + w_out * strideW + kw;
    if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) &&
        (w_in < inputWidth)) {
      const int c_out =
          index / (filterHeight * filterWidth * outputHeight * outputWidth);
      const int c_in = c_out / filterMultiplier;
      const int batch = num_i;
      const int top_offset =
          ((batch * outputChannels + c_out) * outputHeight + h_out) *
              outputWidth +
          w_out;
      const int bottom_offset =
          ((batch * inputChannels + c_in) * inputHeight + h_in) * inputWidth +
          w_in;
      buffer_data[index] = top_diff[top_offset] * inputData[bottom_offset];
    } else {
      buffer_data[index] = 0;
    }
  }
}

template <class T>
class DepthwiseConvFunctor<DEVICE_TYPE_GPU, T> {
 public:
  void operator()(const T* inputData,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* outputData) {
    int outputSize = batchSize * outputChannels * outputHeight * outputWidth;

    size_t blocks = (outputSize + 1024 - 1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);

    ConvolutionDepthwiseForward<T><<<grid, threads, 0, STREAM_DEFAULT>>>(
        outputSize,
        inputData,
        filterData,
        batchSize,
        outputChannels,
        outputHeight,
        outputWidth,
        inputChannels,
        inputHeight,
        inputWidth,
        filterMultiplier,
        filterHeight,
        filterWidth,
        strideH,
        strideW,
        paddingH,
        paddingW,
        outputData);
  }
};

template <class T>
class DepthwiseConvGradInputFunctor<DEVICE_TYPE_GPU, T> {
 public:
  void operator()(const T* outputGrad,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* inputGrad) {
    int inputSize = batchSize * inputChannels * inputHeight * inputWidth;

    size_t blocks = (inputSize + 1024 - 1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);

    ConvolutionDepthwiseInputBackward<T>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<grid, threads, 0, STREAM_DEFAULT>>>(inputSize,
                                               outputGrad,
                                               filterData,
                                               batchSize,
                                               outputChannels,
                                               outputHeight,
                                               outputWidth,
                                               inputChannels,
                                               inputHeight,
                                               inputWidth,
                                               filterMultiplier,
                                               filterHeight,
                                               filterWidth,
                                               strideH,
                                               strideW,
                                               paddingH,
                                               paddingW,
                                               inputGrad);
  }
};

template <class T>
class DepthwiseConvGradFilterFunctor<DEVICE_TYPE_GPU, T> {
 public:
  void operator()(const T* outputGrad,
                  const T* inputData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* colData,
                  T* filterGrad) {
    int colDataSize = outputChannels * filterHeight * filterWidth *
                      outputHeight * outputWidth;

    size_t blocks = (colDataSize + 1024 - 1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);
    BaseMatrix filterGradMatrix(outputChannels * filterHeight * filterWidth,
                                1,
                                filterGrad,
                                false,
                                true);

    for (int i = 0; i < batchSize; i++) {
      ConvolutionDepthwiseFilterBackward<
          T><<<grid, threads, 0, STREAM_DEFAULT>>>(i,
                                                   colDataSize,
                                                   outputGrad,
                                                   inputData,
                                                   batchSize,
                                                   outputChannels,
                                                   outputHeight,
                                                   outputWidth,
                                                   inputChannels,
                                                   inputHeight,
                                                   inputWidth,
                                                   filterMultiplier,
                                                   filterHeight,
                                                   filterWidth,
                                                   strideH,
                                                   strideW,
                                                   paddingH,
                                                   paddingW,
                                                   colData);
      int K = outputHeight * outputWidth;
      int M = colDataSize / K;

      BaseMatrix colMatrix(M, K, colData, false, true);
      filterGradMatrix.sumRows(colMatrix, (T)1.0, (T)1.0);
    }
  }
};

#ifdef PADDLE_TYPE_DOUBLE
template class DepthwiseConvGradInputFunctor<DEVICE_TYPE_GPU, double>;
template class DepthwiseConvFunctor<DEVICE_TYPE_GPU, double>;
template class DepthwiseConvGradFilterFunctor<DEVICE_TYPE_GPU, double>;
#else
template class DepthwiseConvGradInputFunctor<DEVICE_TYPE_GPU, float>;
template class DepthwiseConvFunctor<DEVICE_TYPE_GPU, float>;
template class DepthwiseConvGradFilterFunctor<DEVICE_TYPE_GPU, float>;
#endif

}  // namespace paddle
