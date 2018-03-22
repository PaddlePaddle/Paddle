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

#pragma once
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cuda_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeBilinearInterpFw(const T* in, const size_t inImgH,
                                   const size_t inImgW, const size_t inputH,
                                   const size_t inputW, T* out,
                                   const size_t outImgH, const size_t outImgW,
                                   const size_t outputH, const size_t outputW,
                                   const size_t numChannels, const T ratioH,
                                   const T ratioW) {
  int nthreads = outputH * outputW;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nthreads) {
    int outIdH = tid / outputW;
    int outIdW = tid % outputW;
    int inImgSize = inputW / numChannels;
    int outImgSize = outputW / numChannels;
    int channelId = outIdW / outImgSize;

    int outImgIdy = (outIdW % outImgSize) / outImgW;
    int inImgIdy = ratioH * outImgIdy;
    int hId = (inImgIdy < inImgH - 1) ? 1 : 0;
    T h1lambda = ratioH * outImgIdy - inImgIdy;
    T h2lambda = 1.f - h1lambda;

    int outImgIdx = tid % outImgW;
    int inImgIdx = ratioW * outImgIdx;
    int wId = (inImgIdx < inImgW - 1) ? 1 : 0;
    T w1lambda = ratioW * outImgIdx - inImgIdx;
    T w2lambda = 1.f - w1lambda;

    const T* inPos = &in[outIdH * inputW + channelId * inImgSize +
                         inImgIdy * inImgW + inImgIdx];

    // bilinear interpolation
    out[outIdH * outputW + outIdW] =
        h2lambda * (w2lambda * inPos[0] + w1lambda * inPos[wId]) +
        h1lambda * (w2lambda * inPos[hId * inImgW] +
                    w1lambda * inPos[hId * inImgW + wId]);
  }
}

template <typename T>
__global__ void KeBilinearInterpBw(T* in, const size_t inImgH,
                                   const size_t inImgW, const size_t inputH,
                                   const size_t inputW, const T* out,
                                   const size_t outImgH, const size_t outImgW,
                                   const size_t outputH, const size_t outputW,
                                   const size_t numChannels, const T ratioH,
                                   const T ratioW) {
  int nthreads = outputH * outputW;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nthreads) {
    int outIdH = tid / outputW;
    int outIdW = tid % outputW;
    int inImgSize = inputW / numChannels;
    int outImgSize = outputW / numChannels;
    int channelId = outIdW / outImgSize;

    int outImgIdy = (outIdW % outImgSize) / outImgW;
    int inImgIdy = ratioH * outImgIdy;
    int hId = (inImgIdy < inImgH - 1) ? 1 : 0;
    T h1lambda = ratioH * outImgIdy - inImgIdy;
    T h2lambda = 1.f - h1lambda;

    int outImgIdx = tid % outImgW;
    int inImgIdx = ratioW * outImgIdx;
    int wId = (inImgIdx < inImgW - 1) ? 1 : 0;
    T w1lambda = ratioW * outImgIdx - inImgIdx;
    T w2lambda = 1.f - w1lambda;

    T* inPos = &in[outIdH * inputW + channelId * inImgSize + inImgIdy * inImgW +
                   inImgIdx];
    const T* outPos = &out[outIdH * outputW + outIdW];
    atomicAdd(&inPos[0], h2lambda * w2lambda * outPos[0]);
    atomicAdd(&inPos[wId], h2lambda * w1lambda * outPos[0]);
    atomicAdd(&inPos[hId * inImgW], h1lambda * w2lambda * outPos[0]);
    atomicAdd(&inPos[hId * inImgW + wId], h1lambda * w1lambda * outPos[0]);
  }
}

}  // namespace operators
}  // namespace paddle
