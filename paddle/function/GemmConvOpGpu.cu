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

template class Im2ColFunctor<DEVICE_TYPE_GPU, float>;
template class Im2ColFunctor<DEVICE_TYPE_GPU, double>;

}  // namespace paddle
