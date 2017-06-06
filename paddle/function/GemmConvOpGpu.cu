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
#include "GemmConvOp.h"

namespace paddle {

template<class T>
__global__
void im2col(const T* data_im, int numOuts, int height, int width,
            int blockH, int blockW,
            int strideH, int strideW,
            int paddingH, int paddingW,
            int height_col, int width_col,
            T* data_col) {
  int index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < numOuts) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * blockH * blockW;
    int h_in = h_out * strideH;
    int w_in = w_out * strideW;

    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    for (int i = 0; i < blockH; ++i) {
      for (int j = 0; j < blockW; ++j) {
        int rIdx = int(h_in+i);
        int cIdx = int(w_in+j);
        if ((rIdx-(int)paddingH) >= (int)height ||
            (rIdx-(int)paddingH) < 0 ||
            (cIdx-(int)paddingW) >= (int)width ||
            (cIdx-(int)paddingW) < 0) {
          *data_col = 0;
        } else {
          rIdx = rIdx + channel_in*height - paddingH;
          cIdx = cIdx - paddingW;
          *data_col = data_im[rIdx* width + cIdx];
        }
        data_col += height_col * width_col;
      }
    }
  }
}

template <class T>
class Im2ColFunctor<DEVICE_TYPE_GPU, T> {
public:
  void operator()(const T* imData,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int outputHeight,
                  int outputWidth,
                  T* colData) {
    int numKernels = inputChannels * outputHeight * outputWidth;
    int blocks = (numKernels + 1024 -1) / 1024;
    int blockX = 512;
    int blockY = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);
    im2col<T><<< grid, threads, 0, STREAM_DEFAULT >>>
        (imData, numKernels, inputHeight, inputWidth, filterHeight, filterWidth,
         strideHeight, strideWidth, paddingHeight, paddingWidth,
         outputHeight, outputWidth, colData);
    CHECK_SYNC("Im2ColFunctor GPU failed");
  }
};

template<class T>
__global__
void col2im(size_t n, const T* data_col, size_t height,
            size_t width, size_t channels,
            size_t blockH, size_t blockW,
            size_t strideH, size_t strideW,
            size_t paddingH, size_t paddingW,
            size_t height_col, size_t width_col,
            T* data_im) {
  size_t index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < n) {
    T val = 0;
    int w = int(index % width);
    int h = int((index / width) % height);
    int c = int(index / (width * height));
    if ((w - (int)paddingW) >= 0 &&
        (w - (int)paddingW) < (width-2 * paddingW) &&
        (h - (int)paddingH) >= 0 &&
        (h - paddingH) < (height - 2 * paddingH)) {
      // compute the start and end of the output
      int w_col_start =
        (w < (int)blockW) ? 0 : (w - int(blockW)) / (int)strideW + 1;
      int w_col_end =
        min((int)(w / (int)strideW + 1), (int)(width_col));
      int h_col_start =
        (h < (int)blockH) ? 0 : (h - (int)blockH) / (int)strideH + 1;
      int h_col_end = min(int(h / strideH + 1), int(height_col));
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          // the col location: [c * width * height + h_out, w_out]
          int c_col = int(c * blockH* blockW) + \
            (h - h_col * (int)strideH) * (int)blockW +
            (w - w_col * (int)strideW);
          val += data_col[(c_col * height_col + h_col) * width_col + w_col];
        }
      }
      h -= paddingH;
      w -= paddingW;
      data_im[c*((width-2*paddingW) * (height-2*paddingH)) +
              h*(width-2*paddingW) + w] += val;
    }
  }
}

template <class T>
class Col2ImFunctor<DEVICE_TYPE_GPU, T> {
public:
  void operator()(const T* colData,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int outputHeight,
                  int outputWidth,
                  T* imData) {
    size_t numKernels = inputChannels * (inputHeight + 2*paddingHeight)
        * (inputWidth + 2*paddingWidth);

    size_t blocks = (numKernels + 1024 -1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks+512-1)/512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    col2im<T><<< grid, threads, 0, STREAM_DEFAULT >>>
             (numKernels,
              colData,
              inputHeight + 2*paddingHeight,
              inputWidth + 2*paddingWidth,
              inputChannels,
              filterHeight,
              filterWidth,
              strideHeight,
              strideWidth,
              paddingHeight,
              paddingWidth,
              outputHeight,
              outputWidth,
              imData);
    CHECK_SYNC("Col2ImFunctor GPU failed");
  }
};

template class Im2ColFunctor<DEVICE_TYPE_GPU, float>;
template class Im2ColFunctor<DEVICE_TYPE_GPU, double>;
template class Col2ImFunctor<DEVICE_TYPE_GPU, float>;
template class Col2ImFunctor<DEVICE_TYPE_GPU, double>;

}  // namespace paddle
