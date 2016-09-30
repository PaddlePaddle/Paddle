/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include <float.h>
#include "hl_base.h"
#include "hl_cnn.h"

__global__ void KeFeature2col(size_t n, size_t height, const real* data_im,
                              size_t blockH, size_t blockW, size_t width,
                              size_t strideH, size_t strideW,
                              size_t paddingH, size_t paddingW,
                              size_t height_col, size_t width_col,
                              real* data_col) {
  size_t index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < n) {
    size_t w_out = index % width_col;
    index /= width_col;
    size_t h_out = index % height_col;
    size_t channel_in = index / height_col;
    size_t channel_out = channel_in * blockH * blockW;
    size_t h_in = h_out * strideH;
    size_t w_in = w_out * strideW;

    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    for (size_t i = 0; i < blockH; ++i) {
      for (size_t j = 0; j < blockW; ++j) {
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

void hl_expand_feature2col(const real* dataIm, size_t channels,
                           size_t height, size_t width,
                           size_t blockH, size_t blockW,
                           size_t strideH, size_t strideW,
                           size_t paddingH, size_t paddingW,
                           size_t outputH, size_t outputW,
                           real* dataCol) {
  size_t numKernels = channels * outputH * outputW;

  size_t blocks = (numKernels + 1024 -1) / 1024;
  size_t blockX = 512;
  size_t blockY = (blocks+512-1)/512;
  dim3 threads(1024, 1);
  dim3 grid(blockX, blockY);
  KeFeature2col<<< grid, threads, 0, STREAM_DEFAULT >>>
           (numKernels, height, dataIm, blockH, blockW, width,
           strideH, strideW, paddingH, paddingW,
           outputH, outputW, dataCol);
  CHECK_SYNC("hl_expand_feature2col failed");
}

__global__ void KeCol2Feature(size_t n, const real* data_col, size_t height,
                              size_t width, size_t channels,
                              size_t blockH, size_t blockW,
                              size_t strideH, size_t strideW,
                              size_t paddingH, size_t paddingW,
                              size_t height_col, size_t width_col,
                              real* data_im, real alpha, real beta) {
  size_t index =
    (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < n) {
    real val = 0;
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
      real tD = data_im[c*((width-2*paddingW) * (height-2*paddingH)) +
                          h*(width-2*paddingW) + w];
      data_im[c*((width-2*paddingW) * (height-2*paddingH)) +
              h*(width-2*paddingW) + w] = alpha * val + beta*tD;
    }
  }
}

void hl_shrink_col2feature(const real * dataCol, size_t channels,
                           size_t height, size_t width,
                           size_t blockH, size_t blockW,
                           size_t strideH, size_t strideW,
                           size_t paddingH, size_t paddingW,
                           size_t outputH, size_t outputW,
                           real* dataIm, real alpha, real beta) {
  size_t numKernels = channels * (height + 2*paddingH) * (width + 2*paddingW);

  size_t blocks = (numKernels + 1024 -1) / 1024;
  size_t blockX = 512;
  size_t blockY = (blocks+512-1)/512;
  dim3 threads(1024, 1);
  dim3 grid(blockX, blockY);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  KeCol2Feature<<< grid, threads, 0, STREAM_DEFAULT >>>
           (numKernels, dataCol, height + 2*paddingH, width + 2*paddingW,
           channels, blockH, blockW, strideH, strideW, paddingH, paddingW,
           outputH, outputW, dataIm, alpha, beta);
  CHECK_SYNC("hl_shrink_col2feature failed");
}

__global__ void KeMaxPoolForward(const int nthreads, const real* inputData,
                                 const int channels, const int height,
                                 const int width,
                                 const int pooledH, const int pooledW,
                                 const int ksizeW, const int ksizeH,
                                 const int strideH, const int strideW,
                                 const int offsetH, const int offsetW,
                                 real* tgtData) {
  int index =  blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    int pw = index % pooledW;
    int ph = (index / pooledW) % pooledH;
    int c = (index / pooledW / pooledH) % channels;
    int frameNum = index / pooledW / pooledH / channels;
    int hstart = ph * strideH - offsetH;
    int wstart = pw * strideW - offsetW;
    int hend = min(hstart + ksizeH, height);
    int wend = min(wstart + ksizeW, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    real maxval = -FLT_MAX;
    inputData += (frameNum * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (maxval < inputData[h * width + w])
          maxval = inputData[h * width + w];
      }
    }
    tgtData[index] = maxval;
  }
}

void hl_maxpool_forward(const int frameCnt, const real* inputData,
                        const int channels,
                        const int height, const int width,
                        const int pooledH, const int pooledW,
                        const int sizeX, const int sizeY,
                        const int strideH, const int strideW,
                        const int paddingH, const int paddingW,
                        real* tgtData) {

  int num_kernels = pooledH * pooledW * channels * frameCnt;
  int blocks = (num_kernels + 1024 - 1) / 1024;
  dim3 threads(1024, 1);
  dim3 grid(blocks, 1);

  KeMaxPoolForward<<< grid, threads, 0, STREAM_DEFAULT >>>
           (num_kernels, inputData, channels, height, width,
           pooledH, pooledW, sizeX, sizeY, strideH, strideW,
           paddingH, paddingW, tgtData);
  CHECK_SYNC("hl_maxpool_forward failed");
}

__global__ void KeMaxPoolBackward(const int nthreads, const real* inputData,
                                  const real* outData, const real* outGrad,
                                  const int channels, const int height,
                                  const int width,
                                  const int pooledH, const int pooledW,
                                  const int sizeX, const int sizeY,
                                  const int strideH, const int strideW,
                                  const int padH, const int padW,
                                  real scaleA, real scaleB,
                                  real* targetGrad) {
  int index = blockIdx.x  * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int offsetW = index % width + padW;
    int offsetH = (index / width) % height + padH;
    int offsetC = (index / width / height) % channels;

    int frameNum = index / width / height / channels;
    int phstart = (offsetH < sizeY) ? 0 : (offsetH - sizeY) / strideH + 1;
    int pwstart = (offsetW < sizeX) ? 0 : (offsetW - sizeX) / strideW + 1;
    int phend = offsetH >= 0 ? min(offsetH / strideH + 1, pooledH) : 0;
    int pwend = offsetW >= 0 ? min(offsetW / strideW + 1, pooledW) : 0;
    real gradient = 0;
    real input = inputData[index];
    outData += (frameNum * channels + offsetC) * pooledH * pooledW;
    outGrad += (frameNum * channels + offsetC) * pooledH * pooledW;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (input == outData[ph * pooledW + pw]) {
          gradient += outGrad[ph * pooledW + pw];
        }
      }
    }
    targetGrad[index] =
      scaleB * targetGrad[index] + scaleA * gradient;
  }
}

void hl_maxpool_backward(const int frameCnt, const real* inputData,
                        const real* outData, const real* outGrad,
                        const int channels, const int height,
                        const int width,
                        const int pooledH, const int pooledW,
                        const int sizeX, const int sizeY,
                        const int strideH, const int strideW,
                        const int paddingH, const int paddingW,
                        real scaleA, real scaleB,
                        real* targetGrad) {

  int num_kernels = height * width * channels * frameCnt;
  int blocks = (num_kernels + 1024 - 1) / 1024;

  KeMaxPoolBackward<<< blocks, 1024, 0, STREAM_DEFAULT >>>
           (num_kernels, inputData, outData, outGrad, channels,
           height, width, pooledH, pooledW, sizeX, sizeY,
           strideH, strideW,
           paddingH, paddingW,
           scaleA, scaleB,
           targetGrad);
  CHECK_SYNC("hl_maxpool_backward");
}

__global__ void KeAvgPoolForward(const int nthreads, const real* inputData,
                                 const int channels,
                                 const int height, const int width,
                                 const int pooledH, const int pooledW,
                                 const int sizeX, const int sizeY,
                                 const int strideH, const int strideW,
                                 const int padH, const int padW,
                                 real* tgtData) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    int pw = index % pooledW;
    int ph = (index / pooledW) % pooledH;
    int c = (index / pooledW / pooledH) % channels;
    int frameNum = index / pooledW / pooledH / channels;

    int hstart = ph * strideH - padH;
    int wstart = pw * strideW - padW;
    int hend = min(hstart + sizeY, height + padH);
    int wend = min(wstart + sizeX, width + padW);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);

    real aveval = 0;
    inputData += (frameNum * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += inputData[h * width + w];
      }
    }
    tgtData[index] = aveval / pool_size;
  }
}

void hl_avgpool_forward(const int frameCnt, const real* inputData,
                        const int channels,
                        const int height, const int width,
                        const int pooledH, const int pooledW,
                        const int sizeX, const int sizeY,
                        const int strideH, const int strideW,
                        const int paddingH, const int paddingW, real* tgtData) {
  int num_kernels = pooledH * pooledW * channels * frameCnt;
  int blocks = (num_kernels + 1024 - 1) / 1024;
  KeAvgPoolForward<<< blocks, 1024, 0, STREAM_DEFAULT >>>
           (num_kernels, inputData, channels,
           height, width, pooledH, pooledW,
           sizeX, sizeY, strideH, strideW,
           paddingH, paddingW, tgtData);
  CHECK_SYNC("hl_avgpool_forward failed");
}

__global__ void KeAvgPoolBackward(const int nthreads, const real* outGrad,
                                  const int channels, const int height,
                                  const int width,
                                  const int pooledH, const int pooledW,
                                  const int sizeX, const int sizeY,
                                  const int strideH, const int strideW,
                                  const int padH, const int padW,
                                  real scaleA, real scaleB,
                                  real* tgtGrad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    int offsetW = index % width + padW;
    int offsetH = (index / width) % height + padH;
    int offsetC = (index / width / height) % channels;
    int frameNum = index / width / height / channels;

    int phstart = (offsetH < sizeY) ? 0 : (offsetH - sizeY) / strideH + 1;
    int pwstart = (offsetW < sizeX) ? 0 : (offsetW - sizeX) / strideW + 1;
    int phend = offsetH >= 0 ? min(offsetH / strideH + 1, pooledH) : 0;
    int pwend = offsetW >= 0 ? min(offsetW / strideW + 1, pooledW) : 0;
    real gradient = 0;
    outGrad += (frameNum * channels + offsetC) * pooledH * pooledW;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * strideH - padH;
        int wstart = pw * strideW - padW;
        int hend = min(hstart + sizeY, height + padH);
        int wend = min(wstart + sizeX, width + padW);
        int poolsize = (hend - hstart) * (wend - wstart);
        gradient += outGrad[ph * pooledW + pw]/poolsize;
      }
    }
    tgtGrad[index] = scaleB * tgtGrad[index] + scaleA * gradient;
  }
}

void hl_avgpool_backward(const int frameCnt, const real* outGrad,
                         const int channels,
                         const int height, const int width,
                         const int pooledH, const int pooledW,
                         const int sizeX, const int sizeY,
                         const int strideH, const int strideW,
                         const int paddingH, const int paddingW,
                         real scaleA, real scaleB,
                         real* backGrad) {
  int num_kernels = height * width * channels * frameCnt;
  int blocks = (num_kernels + 1024 - 1) / 1024;

  KeAvgPoolBackward <<< blocks, 1024, 0, STREAM_DEFAULT >>>
           (num_kernels, outGrad, channels, height, width,
           pooledH, pooledW, sizeX, sizeY,
           strideH, strideW,
           paddingH, paddingW,
           scaleA, scaleB,
           backGrad);
  CHECK_SYNC("hl_avgpool_backward failed");
}

__global__ void KeCMRNormFillScale(size_t nthreads, const real* in,
                                   real* scale, size_t channels,
                                   size_t height, size_t width, size_t size,
                                   real alpha) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    size_t w = index % width;
    size_t h = (index / width) % height;
    size_t n = index / width / height;
    size_t offset = (n * channels * height + h) * width + w;
    size_t step = height * width;
    in += offset;
    scale += offset;
    size_t head = 0;
    size_t pre_pad = (size - 1) / 2;
    size_t post_pad = size - pre_pad - 1;
    real accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha;
      ++head;
    }
  }
}

 __global__ void KeCMRNormOutput(size_t nthreads, const real* in,
                                 const real* scale, real negative_beta,
                                 real* out) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

void hl_CMRNorm_forward(size_t frameCnt, const real* in, real* scale,
                        real* out, size_t channels,
                        size_t height, size_t width, size_t sizeX,
                        real alpha, real beta) {
  size_t threadsNum = frameCnt * height * width;
  size_t blocksX = (threadsNum + 1024 - 1) / 1024;
  size_t blocksY = 1;
  dim3 threads(1024, 1);
  dim3 grid(blocksX, blocksY);

  KeCMRNormFillScale<<<grid, threads, 0, STREAM_DEFAULT>>>
      (threadsNum, in, scale, channels, height, width, sizeX, alpha);

  threadsNum = frameCnt * height * width *channels;
  blocksX = (threadsNum + 1024 -1) / 1024;
  dim3 threads2(1024, 1);
  dim3 grid2(blocksX, blocksY);
  KeCMRNormOutput<<<grid2, threads2, 0, STREAM_DEFAULT>>>
           (threadsNum, in, scale, beta, out);
  CHECK_SYNC("hl_CMRNorm_forward");
}

__global__ void KeCMRNormDiff(size_t nthreads, const real* bottom_data,
                              const real* top_data, const real* scale,
                              const real* top_diff, size_t channels,
                              size_t height, size_t width, size_t size,
                              real negative_beta, real cache_ratio,
                              real* bottom_diff ) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    size_t w = index % width;
    size_t h = (index / width) % height;
    size_t n = index / width / height;
    size_t offset = (n * channels * height + h) * width + w;
    size_t step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    real accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += top_diff[head * step] *
        top_data[head * step] / scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += top_diff[head * step] *
        top_data[head * step] / scale[head * step];
      bottom_diff[(head - post_pad) * step] +=
        top_diff[(head - post_pad) * step] *
        pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
        bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] +=
        top_diff[(head - post_pad) * step] *
        pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
        bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] +=
        top_diff[(head - post_pad) * step] *
        pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
        bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

void hl_CMRNorm_backward(size_t frameCnt, const real* inV,
                         const real* scale,
                         const real* outV, const real* outDiff,
                         real *inDiff, size_t channels,
                         size_t height, size_t width, size_t sizeX,
                         real alpha, real beta) {
  size_t threadsNum = frameCnt * height * width;
  size_t blocksX = (threadsNum + 1024 -1) / 1024;
  size_t blocksY = 1;
  dim3 threads(1024, 1);
  dim3 grid(blocksX, blocksY);
  KeCMRNormDiff <<<grid, threads, 0, STREAM_DEFAULT>>>
           (threadsNum, inV, outV, scale, outDiff, channels,
           height, width, sizeX, alpha, beta, inDiff);
  CHECK_SYNC("hl_CMRNorm_backward");
}
