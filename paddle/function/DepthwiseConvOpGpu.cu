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

#include "ConvOp.h"
#include "DepthwiseConvOp.h"

namespace paddle {
template <class T>
__global__ void ConvolutionDepthwiseWeightForward(const int nthreads,
    const T* const bottom_data, const T* const weight_data,
    const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, T* const top_data) {

  int index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  
  if(index < nthreads) {
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const T* weight = weight_data + c * kernel_h * kernel_w;
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height)
              && (w_in >= 0) && (w_in < bottom_width)) {
          const int offset = ((n * channels + c) * bottom_height + h_in)
                * bottom_width + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}

template <class T>
__global__ void ConvolutionDepthwiseBottomBackward(const int nthreads,
    const T* const top_diff, const T* const weight_data,
    const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, T* const bottom_diff) {
  int index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if(index < nthreads) {
    const int n = index / channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    const T* weight = weight_data + c * kernel_h * kernel_w;
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_out_s = h + pad_h - kh * dilation_h;
        const int w_out_s = w + pad_w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
	  //it affect the effectives
          if ((h_out >= 0) && (h_out < top_height)
                && (w_out >= 0) && (w_out < top_width)) {
            const int offset = ((n * channels + c) * top_height + h_out)
                  * top_width + w_out;
            value += (*weight) * top_diff[offset];
          }
        }
        ++weight;
      }
    }
    bottom_diff[index] += value;
  }
}

template <class T>
__global__ void ConvolutionDepthwiseWeightBackward(const int num_i, const int nthreads,
    const T* const top_diff, const T* const bottom_data,
    const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, T* const buffer_data) {
  int index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int kh = (index / kernel_w / top_height / top_width)
          % kernel_h;
    const int kw = (index / top_height / top_width) % kernel_w;
    const int h_in = -pad_h + h * stride_h + kh * dilation_h;
    const int w_in = -pad_w + w * stride_w + kw * dilation_w;
    if ((h_in >= 0) && (h_in < bottom_height)
          && (w_in >= 0) && (w_in < bottom_width)) {
      const int c = index / kernel_h / kernel_w / top_height / top_width;
      const int n = num_i;
      const int top_offset = ((n * channels + c) * top_height + h)
            * top_width + w;
      const int bottom_offset = ((n * channels + c) * bottom_height + h_in)
            * bottom_width + w_in;
      buffer_data[index] = top_diff[top_offset] * bottom_data[bottom_offset];
    } else {
      buffer_data[index] = 0;
    }
  }
}

template <class T>
class DepthwiseConvFunctor<DEVICE_TYPE_GPU, T>{
public:
  void operator()(int outputSize, 
            const T* inputData, 
            const T* filterData,
            int batchSize,
            int outputChannels,
            int outputHeight,
            int outputWidth,
            int filterHeight,
            int filterWidth,
            int strideH,
            int strideW,
            int paddingH,
            int paddingW,
            T* outputData){

    size_t blocks = (outputSize + 1024 -1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks+512-1)/512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);
    
    ConvolutionDepthwiseWeightForward<T>
        <<< grid, threads, 0, STREAM_DEFAULT >>>(
            outputSize, 
            inputData, 
            filterData,
            batchSize,
            outputChannels,
            outputHeight,
            outputWidth,
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
class DepthwiseConvGradInputFunctor<DEVICE_TYPE_GPU, T>{
public:
  void operator()(int inputSize,
            const T* outputGrad,
            const T* filterData,
            int batchSize,
            int outputChannels,
            int outputHeight,
            int outputWidth,
            int inputHeight,
            int inputWidth,
            int filterHeight,
            int filterWidth,
            int strideH,
            int strideW,
            int paddingH,
            int paddingW,
                T* inputGrad){

    size_t blocks = (inputSize + 1024 -1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks+512-1)/512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);

    ConvolutionDepthwiseBottomBackward<T>
          // NOLINT_NEXT_LINE(whitespace/operators)
        <<< grid, threads, 0, STREAM_DEFAULT >>>(
            inputSize,
            outputGrad,
            filterData,
            batchSize,
            outputChannels,
            outputHeight,
            outputWidth,
            inputHeight,
            inputWidth,
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
  void operator()(int num_i,
                int colDataSize,
                const T* outputGrad,
                const T* inputData,
                int batchSize,
                int outputChannels,
                int outputHeight,
                int outputWidth,
                int inputHeight,
                int inputWidth,
                int filterHeight,
                int filterWidth,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                T* colData,
                T* multiplierData,
                T* filterGrad){

        size_t blocks = (colDataSize + 1024 -1) / 1024;
        size_t blockX = 512;
        size_t blockY = (blocks+512-1)/512;
        dim3 threads(1024, 1);
        dim3 grid(blockX, blockY);

	    ConvolutionDepthwiseWeightBackward<T>
            <<< grid, threads, 0, STREAM_DEFAULT >>>(
                i,
                size,
                outputGrad,
                inputData,
                batchSize,
                outputChannels,
                outputHeight,
                outputWidth,
                inputHeight,
                inputWidth,
                filterHeight,
                filterWidth,
                strideH,
                strideW,
                paddingH,
                paddingW,
                colData
            );
        GemmFunctor<Device, real> gemm;
        int M = size / outputHeight / outputWidth;
        int N = 1;
        int K = outputHeight * outputWidth;
        gemm(CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            1.0f,
            colData,
            K,
            multiplierData,
            N,
            1.0f,
            filterGrad,
            N);
        //gemv
    }
};

template class DepthwiseConvGradInputFunctor<DEVICE_TYPE_GPU, float>;
template class DepthwiseConvGradInputFunctor<DEVICE_TYPE_GPU, double>;
template class DepthwiseConvFunctor<DEVICE_TYPE_GPU, float>;
template class DepthwiseConvFunctor<DEVICE_TYPE_GPU, double>;
template class DepthwiseConvGradFilterFunctor<DEVICE_TYPE_GPU, float>;
template class DepthwiseConvGradFilterFunctor<DEVICE_TYPE_GPU, double>;

}  // namespace paddle
