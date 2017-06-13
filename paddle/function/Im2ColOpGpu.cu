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

#include "Im2Col.h"

namespace paddle {

template<class T>
__global__
void im2colOCF(const T* imData, T* colData,
               int inputChannels,
               int inputHeight, int inputWidth,
               int filterHeight, int filterWidth,
               int strideHeight, int strideWidth,
               int paddingHeight, int paddingWidth,
               int outputHeight, int outputWidth) {
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int swId = blockIdx.x;
  int shId = blockIdx.y;

  for (int channelId = threadIdx.z;
       channelId < inputChannels;
       channelId += blockDim.z) {
    int widthOffset = idx + swId * strideWidth - paddingWidth;
    int heightOffset = idy + shId * strideHeight - paddingHeight;
    int imOffset = widthOffset + heightOffset * inputWidth
       + channelId * inputHeight * inputWidth;

    int colOffset = idx + idy * filterWidth
      + channelId * filterHeight * filterWidth
      + (shId * outputWidth + swId)
      * (inputChannels * filterHeight * filterWidth);

    if (idx < filterWidth && idy < filterHeight) {
      if (heightOffset >= inputHeight || heightOffset < 0 ||
          widthOffset >= inputWidth || widthOffset < 0) {
        colData[colOffset] = T(0);
      } else {
        colData[colOffset] = imData[imOffset];
      }
    }
  }
}

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 */
template <class T>
class Im2ColFunctor<kOCF, DEVICE_TYPE_GPU, T> {
public:
  void operator()(const T* imData,
                  const TensorShape& imShape,
                  T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[3];
    int filterWidth = colShape[4];
    int outputHeight = colShape[0];
    int outputWidth = colShape[1];

    int blockDimX = 0;
    int blockDimY = 0;
    if (filterHeight <= 4 && filterWidth <= 4) {
      blockDimX = 4;
      blockDimY = 4;
    } else if (filterHeight <= 8 && filterWidth <= 8) {
      blockDimX = 8;
      blockDimY = 8;
    } else if (filterHeight <= 16 && filterWidth <= 16) {
      blockDimX = 16;
      blockDimY = 16;
    } else {
      blockDimX = 32;
      blockDimY = 32;
    }

    int blockDimZ = 1024 / blockDimX / blockDimY;
    dim3 threads(blockDimX, blockDimY, std::min(blockDimZ, inputChannels));
    dim3 grid(outputWidth, outputHeight);
    im2colOCF<T><<< grid, threads, 0, STREAM_DEFAULT >>>
        (imData, colData, inputChannels, inputHeight, inputWidth,
         filterHeight, filterWidth, strideHeight, strideWidth,
         paddingHeight, paddingWidth, outputHeight, outputWidth);
    CHECK_SYNC("Im2ColFunctor GPU failed");
  }
};

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 */
template <class T>
class Col2ImFunctor<kOCF, DEVICE_TYPE_GPU, T> {
public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth) {
  }
};

template class Im2ColFunctor<kOCF, DEVICE_TYPE_GPU, float>;
template class Im2ColFunctor<kOCF, DEVICE_TYPE_GPU, double>;

}  // namespace paddle
