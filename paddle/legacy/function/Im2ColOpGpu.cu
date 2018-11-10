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

#include "Im2Col.h"
#include "hl_device_functions.cuh"

namespace paddle {

template <class T>
__global__ void im2col(const T* data_im,
                       int numOuts,
                       int height,
                       int width,
                       int blockH,
                       int blockW,
                       int strideH,
                       int strideW,
                       int paddingH,
                       int paddingW,
                       int dilationH,
                       int dilationW,
                       int height_col,
                       int width_col,
                       T* data_col) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
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
        int rIdx = int(h_in + i * dilationH);
        int cIdx = int(w_in + j * dilationW);
        if ((rIdx - (int)paddingH) >= (int)height ||
            (rIdx - (int)paddingH) < 0 ||
            (cIdx - (int)paddingW) >= (int)width ||
            (cIdx - (int)paddingW) < 0) {
          *data_col = 0;
        } else {
          rIdx = rIdx + channel_in * height - paddingH;
          cIdx = cIdx - paddingW;
          *data_col = data_im[rIdx * width + cIdx];
        }
        data_col += height_col * width_col;
      }
    }
  }
}

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [inputChannels, filterHeight, filterWidth, outputHeight, outputWidth]
 */
template <class T>
class Im2ColFunctor<kCFO, DEVICE_TYPE_GPU, T> {
 public:
  void operator()(const T* imData,
                  const TensorShape& imShape,
                  T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight,
                  int dilationWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[1];
    int filterWidth = colShape[2];
    int outputHeight = colShape[3];
    int outputWidth = colShape[4];

    int numKernels = inputChannels * outputHeight * outputWidth;
    int blocks = (numKernels + 1024 - 1) / 1024;
    int blockX = 512;
    int blockY = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);
    im2col<T><<<grid, threads, 0, STREAM_DEFAULT>>>(imData,
                                                    numKernels,
                                                    inputHeight,
                                                    inputWidth,
                                                    filterHeight,
                                                    filterWidth,
                                                    strideHeight,
                                                    strideWidth,
                                                    paddingHeight,
                                                    paddingWidth,
                                                    dilationHeight,
                                                    dilationWidth,
                                                    outputHeight,
                                                    outputWidth,
                                                    colData);
    CHECK_SYNC("Im2ColFunctor GPU failed");
  }
};

template <class T>
__global__ void col2im(size_t n,
                       const T* data_col,
                       size_t height,
                       size_t width,
                       size_t channels,
                       size_t blockH,
                       size_t blockW,
                       size_t strideH,
                       size_t strideW,
                       size_t paddingH,
                       size_t paddingW,
                       size_t dilationH,
                       size_t dilationW,
                       size_t height_col,
                       size_t width_col,
                       T* data_im) {
  size_t index =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < n) {
    T val = 0;
    int w = int(index % width);
    int h = int((index / width) % height);
    int c = int(index / (width * height));
    int filterH = (blockH - 1) * dilationH + 1;
    int filterW = (blockW - 1) * dilationW + 1;

    if ((w - (int)paddingW) >= 0 &&
        (w - (int)paddingW) < (width - 2 * paddingW) &&
        (h - (int)paddingH) >= 0 && (h - paddingH) < (height - 2 * paddingH)) {
      // compute the start and end of the output
      int w_col_start =
          (w < (int)filterW) ? 0 : (w - int(filterW)) / (int)strideW + 1;
      int w_col_end = min((int)(w / (int)strideW + 1), (int)(width_col));
      int h_col_start =
          (h < (int)filterH) ? 0 : (h - (int)filterH) / (int)strideH + 1;
      int h_col_end = min(int(h / strideH + 1), int(height_col));

      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          // the col location: [c * width * height + h_out, w_out]
          int h_k = (h - h_col * strideH);
          int w_k = (w - w_col * strideW);
          if (h_k % dilationH == 0 && w_k % dilationW == 0) {
            h_k /= dilationH;
            w_k /= dilationW;
            int c_col =
                (((c * blockH + h_k) * blockW + w_k) * height_col + h_col) *
                    width_col +
                w_col;
            val += data_col[c_col];
          }
        }
      }
      h -= paddingH;
      w -= paddingW;
      data_im[c * ((width - 2 * paddingW) * (height - 2 * paddingH)) +
              h * (width - 2 * paddingW) + w] += val;
    }
  }
}

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [inputChannels, filterHeight, filterWidth, outputHeight, outputWidth]
 */
template <class T>
class Col2ImFunctor<kCFO, DEVICE_TYPE_GPU, T> {
 public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight,
                  int dilationWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[1];
    int filterWidth = colShape[2];
    int outputHeight = colShape[3];
    int outputWidth = colShape[4];

    size_t numKernels = inputChannels * (inputHeight + 2 * paddingHeight) *
                        (inputWidth + 2 * paddingWidth);

    size_t blocks = (numKernels + 1024 - 1) / 1024;
    size_t blockX = 512;
    size_t blockY = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(blockX, blockY);

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    col2im<T><<<grid, threads, 0, STREAM_DEFAULT>>>(
        numKernels,
        colData,
        inputHeight + 2 * paddingHeight,
        inputWidth + 2 * paddingWidth,
        inputChannels,
        filterHeight,
        filterWidth,
        strideHeight,
        strideWidth,
        paddingHeight,
        paddingWidth,
        dilationHeight,
        dilationWidth,
        outputHeight,
        outputWidth,
        imData);
    CHECK_SYNC("Col2ImFunctor GPU failed");
  }
};

template class Im2ColFunctor<kCFO, DEVICE_TYPE_GPU, float>;
template class Im2ColFunctor<kCFO, DEVICE_TYPE_GPU, double>;
template class Col2ImFunctor<kCFO, DEVICE_TYPE_GPU, float>;
template class Col2ImFunctor<kCFO, DEVICE_TYPE_GPU, double>;

template <class T>
__global__ void im2colOCF(const T* imData,
                          T* colData,
                          int inputChannels,
                          int inputHeight,
                          int inputWidth,
                          int filterHeight,
                          int filterWidth,
                          int strideHeight,
                          int strideWidth,
                          int paddingHeight,
                          int paddingWidth,
                          int dilationHeight,
                          int dilationWidth,
                          int outputHeight,
                          int outputWidth) {
  int swId = blockIdx.x;
  int shId = blockIdx.y;
  for (int channelId = threadIdx.z; channelId < inputChannels;
       channelId += blockDim.z) {
    for (int idy = threadIdx.y; idy < filterHeight; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filterWidth; idx += blockDim.x) {
        int widthOffset =
            idx * dilationHeight + swId * strideWidth - paddingWidth;
        int heightOffset =
            idy * dilationWidth + shId * strideHeight - paddingHeight;
        int imOffset = widthOffset + heightOffset * inputWidth +
                       channelId * inputHeight * inputWidth;

        int colOffset = idx + idy * filterWidth +
                        channelId * filterHeight * filterWidth +
                        (shId * outputWidth + swId) *
                            (inputChannels * filterHeight * filterWidth);

        if (heightOffset >= inputHeight || heightOffset < 0 ||
            widthOffset >= inputWidth || widthOffset < 0) {
          colData[colOffset] = T(0);
        } else {
          colData[colOffset] = imData[imOffset];
        }
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
                  int paddingWidth,
                  int dilationHeight,
                  int dilationWidth) {
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
    im2colOCF<T><<<grid, threads, 0, STREAM_DEFAULT>>>(imData,
                                                       colData,
                                                       inputChannels,
                                                       inputHeight,
                                                       inputWidth,
                                                       filterHeight,
                                                       filterWidth,
                                                       strideHeight,
                                                       strideWidth,
                                                       paddingHeight,
                                                       paddingWidth,
                                                       dilationHeight,
                                                       dilationWidth,
                                                       outputHeight,
                                                       outputWidth);
    CHECK_SYNC("Im2ColFunctor GPU failed");
  }
};

template <class T>
__global__ void col2imOCF(T* imData,
                          const T* colData,
                          int inputChannels,
                          int inputHeight,
                          int inputWidth,
                          int filterHeight,
                          int filterWidth,
                          int strideHeight,
                          int strideWidth,
                          int paddingHeight,
                          int paddingWidth,
                          int dilationHeight,
                          int dilationWidth,
                          int outputHeight,
                          int outputWidth) {
  int swId = blockIdx.x;
  int shId = blockIdx.y;
  for (int channelId = threadIdx.z; channelId < inputChannels;
       channelId += blockDim.z) {
    for (int idy = threadIdx.y; idy < filterHeight; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filterWidth; idx += blockDim.x) {
        int widthOffset =
            idx * dilationWidth + swId * strideWidth - paddingWidth;
        int heightOffset =
            idy * dilationHeight + shId * strideHeight - paddingHeight;
        int imOffset = widthOffset + heightOffset * inputWidth +
                       channelId * inputHeight * inputWidth;

        int colOffset = idx + idy * filterWidth +
                        channelId * filterHeight * filterWidth +
                        (shId * outputWidth + swId) *
                            (inputChannels * filterHeight * filterWidth);

        if (heightOffset >= 0 && heightOffset < inputHeight &&
            widthOffset >= 0 && widthOffset < inputWidth) {
          paddle::paddleAtomicAdd(imData + imOffset, colData[colOffset]);
        }
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
class Col2ImFunctor<kOCF, DEVICE_TYPE_GPU, T> {
 public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight,
                  int dilationWidth) {
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
    col2imOCF<T><<<grid, threads, 0, STREAM_DEFAULT>>>(imData,
                                                       colData,
                                                       inputChannels,
                                                       inputHeight,
                                                       inputWidth,
                                                       filterHeight,
                                                       filterWidth,
                                                       strideHeight,
                                                       strideWidth,
                                                       paddingHeight,
                                                       paddingWidth,
                                                       dilationHeight,
                                                       dilationWidth,
                                                       outputHeight,
                                                       outputWidth);
    CHECK_SYNC("Col2ImFunctor GPU failed");
  }
};

template class Im2ColFunctor<kOCF, DEVICE_TYPE_GPU, float>;
template class Im2ColFunctor<kOCF, DEVICE_TYPE_GPU, double>;
template class Col2ImFunctor<kOCF, DEVICE_TYPE_GPU, float>;
template class Col2ImFunctor<kOCF, DEVICE_TYPE_GPU, double>;

}  // namespace paddle
