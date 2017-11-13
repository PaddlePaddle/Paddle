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

#pragma once
#include <type_traits>
#include "paddle/operators/math/detail/activation_functions.h"
#include "paddle/operators/math/gru_compute.h"
#include "paddle/platform/cuda_helper.h"
#include "paddle/platform/device_context.h"

#include <glog/logging.h>

namespace paddle {
namespace operators {
namespace math {
namespace detail {

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template <class OpResetOutput, bool isBatch, typename T>
__global__ void KeGruForwardResetOutput(OpResetOutput opResetOutput,
                                        T *gateValue, T *resetOutputValue,
                                        T *prevOutputValue, int frameSize,
                                        int batchSize,
                                        activation_mode_t active_gate) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;

  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    resetOutputValue += batchIdx * frameSize;
  }

  T rPrevOut = 0;
  T rValueResetOutput;
  T rValueUpdateGate = gateValue[frameIdx + frameSize * 0];
  T rValueResetGate = gateValue[frameIdx + frameSize * 1];

  if (prevOutputValue) {
    if (isBatch) prevOutputValue += batchIdx * frameSize;
    rPrevOut = prevOutputValue[frameIdx];
  }

  opResetOutput(rValueUpdateGate, rValueResetGate, rPrevOut, rValueResetOutput,
                active_gate);

  gateValue[frameIdx + frameSize * 0] = rValueUpdateGate;
  gateValue[frameIdx + frameSize * 1] = rValueResetGate;
  resetOutputValue[frameIdx] = rValueResetOutput;
}

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template <class OpFinalOutput, bool isBatch, typename T>
__global__ void KeGruForwardFinalOutput(OpFinalOutput opFinalOutput,
                                        T *gateValue, T *prevOutputValue,
                                        T *outputValue, int frameSize,
                                        int batchSize,
                                        activation_mode_t active_node) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;
  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    outputValue += batchIdx * frameSize;
  }

  T rOutput;
  T rPrevOut = 0;
  T rValueUpdateGate = gateValue[frameIdx + frameSize * 0];
  T rValueFrameState = gateValue[frameIdx + frameSize * 2];

  if (prevOutputValue) {
    if (isBatch) prevOutputValue += batchIdx * frameSize;
    rPrevOut = prevOutputValue[frameIdx];
  }

  opFinalOutput(rValueUpdateGate, rValueFrameState, rPrevOut, rOutput,
                active_node);

  gateValue[frameIdx + frameSize * 2] = rValueFrameState;
  outputValue[frameIdx] = rOutput;
}

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template <class OpStateGrad, bool isBatch, typename T>
__global__ void KeGruBackwardStateGrad(OpStateGrad opStateGrad, T *gateValue,
                                       T *gateGrad, T *prevOutValue,
                                       T *prevOutGrad, T *outputGrad,
                                       int frameSize, int batchSize,
                                       activation_mode_t active_node) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;
  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    gateGrad += batchIdx * 3 * frameSize;
    outputGrad += batchIdx * frameSize;
  }

  T rUpdateGateGrad;
  T rFrameStateGrad;
  T rPrevOutValue = 0;
  T rPrevOutGrad = 0;
  T rUpdateGateValue = gateValue[frameIdx + frameSize * 0];
  T rFrameStateValue = gateValue[frameIdx + frameSize * 2];
  T rOutGrad = outputGrad[frameIdx];

  if (prevOutValue && prevOutGrad) {
    if (isBatch) prevOutValue += batchIdx * frameSize;
    rPrevOutValue = prevOutValue[frameIdx];

    if (isBatch) prevOutGrad += batchIdx * frameSize;
    rPrevOutGrad = prevOutGrad[frameIdx];
  }

  opStateGrad(rUpdateGateValue, rUpdateGateGrad, rFrameStateValue,
              rFrameStateGrad, rPrevOutValue, rPrevOutGrad, rOutGrad,
              active_node);

  gateGrad[frameIdx + frameSize * 0] = rUpdateGateGrad;
  gateGrad[frameIdx + frameSize * 2] = rFrameStateGrad;
  if (prevOutGrad) {
    prevOutGrad[frameIdx] = rPrevOutGrad;
  }
}

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template <class OpResetGrad, bool isBatch, typename T>
__global__ void KeGruBackwardResetGrad(OpResetGrad opResetGrad, T *gateValue,
                                       T *gateGrad, T *prevOutValue,
                                       T *prevOutGrad, T *resetOutputGrad,
                                       int frameSize, int batchSize,
                                       activation_mode_t active_gate) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;
  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    gateGrad += batchIdx * 3 * frameSize;
    resetOutputGrad += batchIdx * frameSize;
  }

  T rResetGateGrad;
  T rPrevOutValue = 0;
  T rPrevOutGrad = 0;
  T rResetOutputGrad = 0;
  T rUpdateGateValue = gateValue[frameIdx + frameSize * 0];
  T rUpdateGateGrad = gateGrad[frameIdx + frameSize * 0];
  T rResetGateValue = gateValue[frameIdx + frameSize * 1];

  if (prevOutValue && prevOutGrad) {
    if (isBatch) prevOutValue += batchIdx * frameSize;
    if (isBatch) prevOutGrad += batchIdx * frameSize;
    rPrevOutValue = prevOutValue[frameIdx];
    rPrevOutGrad = prevOutGrad[frameIdx];
    rResetOutputGrad = resetOutputGrad[frameIdx];
  }

  opResetGrad(rUpdateGateValue, rUpdateGateGrad, rResetGateValue,
              rResetGateGrad, rPrevOutValue, rPrevOutGrad, rResetOutputGrad,
              active_gate);

  gateGrad[frameIdx + frameSize * 0] = rUpdateGateGrad;
  gateGrad[frameIdx + frameSize * 1] = rResetGateGrad;
  if (prevOutGrad) {
    prevOutGrad[frameIdx] = rPrevOutGrad;
  }
}
}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
