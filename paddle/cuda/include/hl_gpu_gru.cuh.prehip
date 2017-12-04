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


#ifndef HL_GPU_GRU_CUH_
#define HL_GPU_GRU_CUH_

#ifdef __NVCC__

#include "paddle/utils/Logging.h"

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template<class OpResetOutput, bool isBatch>
__global__ void KeGruForwardResetOutput(OpResetOutput opResetOutput,
                                        real *gateValue,
                                        real *resetOutputValue,
                                        real *prevOutputValue,
                                        int frameSize,
                                        int batchSize,
                                        hl_activation_mode_t active_gate) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;

  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    resetOutputValue += batchIdx * frameSize;
  }

  real rPrevOut = 0;
  real rValueResetOutput;
  real rValueUpdateGate = gateValue[frameIdx + frameSize * 0];
  real rValueResetGate  = gateValue[frameIdx + frameSize * 1];

  if (prevOutputValue) {
    if (isBatch) prevOutputValue += batchIdx * frameSize;
    rPrevOut = prevOutputValue[frameIdx];
  }

  opResetOutput(rValueUpdateGate,
                rValueResetGate,
                rPrevOut,
                rValueResetOutput,
                hppl::gpu::forward[active_gate]);

  gateValue[frameIdx + frameSize * 0] = rValueUpdateGate;
  gateValue[frameIdx + frameSize * 1] = rValueResetGate;
  resetOutputValue[frameIdx] = rValueResetOutput;
}

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template<class OpFinalOutput, bool isBatch>
__global__ void KeGruForwardFinalOutput(OpFinalOutput opFinalOutput,
                                        real *gateValue,
                                        real *prevOutputValue,
                                        real *outputValue,
                                        int frameSize,
                                        int batchSize,
                                        hl_activation_mode_t active_node) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;
  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    outputValue += batchIdx * frameSize;
  }

  real rOutput;
  real rPrevOut = 0;
  real rValueUpdateGate = gateValue[frameIdx + frameSize * 0];
  real rValueFrameState = gateValue[frameIdx + frameSize * 2];

  if (prevOutputValue) {
    if (isBatch) prevOutputValue += batchIdx * frameSize;
    rPrevOut = prevOutputValue[frameIdx];
  }

  opFinalOutput(rValueUpdateGate,
                rValueFrameState,
                rPrevOut,
                rOutput,
                hppl::gpu::forward[active_node]);

  gateValue[frameIdx + frameSize * 2] = rValueFrameState;
  outputValue[frameIdx] = rOutput;
}

template<class OpResetOutput, class OpFinalOutput>
void hl_gpu_gru_forward(OpResetOutput opResetOutput,
                        OpFinalOutput opFinalOutput,
                        hl_gru_value value,
                        int frameSize,
                        int batchSize,
                        hl_activation_mode_t active_node,
                        hl_activation_mode_t active_gate) {
  dim3 threads;
  dim3 grid;
  if (batchSize == 1) {
    int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
    int frameBlocks = (frameSize + 1024 - 1) / 1024;
    threads = dim3(framePerBlock, 1);
    grid = dim3(frameBlocks, 1);
  } else {
    threads = dim3(32, 32);
    grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 32 - 1) / 32);
  }

  if (value.prevOutValue) {
    hl_matrix_mul(value.prevOutValue, HPPL_OP_N,
                  value.gateWeight, HPPL_OP_N,
                  value.gateValue,
                  batchSize, 2*frameSize, frameSize,
                  /*alpha = */ 1, /*beta = */ 1,
                  frameSize, 2* frameSize, 3*frameSize);
  }

  if (batchSize == 1) {
    KeGruForwardResetOutput<OpResetOutput, /* isBatch= */false>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opResetOutput,
        value.gateValue, value.resetOutputValue, value.prevOutValue,
        frameSize, batchSize, active_gate);
  } else {
    KeGruForwardResetOutput<OpResetOutput, /* isBatch= */true>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opResetOutput,
        value.gateValue, value.resetOutputValue, value.prevOutValue,
        frameSize, batchSize, active_gate);
  }

  if (value.prevOutValue) {
    hl_matrix_mul(value.resetOutputValue, HPPL_OP_N,
                  value.stateWeight, HPPL_OP_N,
                  value.gateValue + 2*frameSize,
                  batchSize, frameSize, frameSize,
                  /*alpha = */ 1, /*beta = */ 1,
                  frameSize, frameSize, 3*frameSize);
  }

  if (batchSize == 1) {
    KeGruForwardFinalOutput<OpFinalOutput, /* isBatch= */false>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opFinalOutput,
        value.gateValue, value.prevOutValue, value.outputValue,
        frameSize, batchSize, active_node);
  } else {
    KeGruForwardFinalOutput<OpFinalOutput, /* isBatch= */true>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opFinalOutput,
        value.gateValue, value.prevOutValue, value.outputValue,
        frameSize, batchSize, active_node);
  }

  CHECK_SYNC("hl_gpu_gru_forward failed");
}

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template<class OpStateGrad, bool isBatch>
__global__ void KeGruBackwardStateGrad(OpStateGrad opStateGrad,
                                       real *gateValue,
                                       real *gateGrad,
                                       real *prevOutValue,
                                       real *prevOutGrad,
                                       real *outputGrad,
                                       int frameSize,
                                       int batchSize,
                                       hl_activation_mode_t active_node) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;
  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    gateGrad  += batchIdx * 3 * frameSize;
    outputGrad += batchIdx * frameSize;
  }

  real rUpdateGateGrad;
  real rFrameStateGrad;
  real rPrevOutValue = 0;
  real rPrevOutGrad  = 0;
  real rUpdateGateValue = gateValue[frameIdx + frameSize * 0];
  real rFrameStateValue = gateValue[frameIdx + frameSize * 2];
  real rOutGrad  = outputGrad[frameIdx];

  if (prevOutValue && prevOutGrad) {
    if (isBatch) prevOutValue += batchIdx * frameSize;
    rPrevOutValue = prevOutValue[frameIdx];

    if (isBatch) prevOutGrad  += batchIdx * frameSize;
    rPrevOutGrad  = prevOutGrad[frameIdx];
  }

  opStateGrad(rUpdateGateValue,
              rUpdateGateGrad,
              rFrameStateValue,
              rFrameStateGrad,
              rPrevOutValue,
              rPrevOutGrad,
              rOutGrad,
              hppl::gpu::backward[active_node]);

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
template<class OpResetGrad, bool isBatch>
__global__ void KeGruBackwardResetGrad(OpResetGrad opResetGrad,
                                       real *gateValue,
                                       real *gateGrad,
                                       real *prevOutValue,
                                       real *prevOutGrad,
                                       real *resetOutputGrad,
                                       int frameSize,
                                       int batchSize,
                                       hl_activation_mode_t active_gate) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;
  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    gateValue += batchIdx * 3 * frameSize;
    gateGrad  += batchIdx * 3 * frameSize;
    resetOutputGrad += batchIdx * frameSize;
  }

  real rResetGateGrad;
  real rPrevOutValue = 0;
  real rPrevOutGrad  = 0;
  real rResetOutputGrad = 0;
  real rUpdateGateValue = gateValue[frameIdx + frameSize * 0];
  real rUpdateGateGrad  = gateGrad[frameIdx + frameSize * 0];
  real rResetGateValue  = gateValue[frameIdx + frameSize * 1];

  if (prevOutValue && prevOutGrad) {
    if (isBatch) prevOutValue += batchIdx * frameSize;
    if (isBatch) prevOutGrad  += batchIdx * frameSize;
    rPrevOutValue = prevOutValue[frameIdx];
    rPrevOutGrad  = prevOutGrad[frameIdx];
    rResetOutputGrad = resetOutputGrad[frameIdx];
  }

  opResetGrad(rUpdateGateValue,
              rUpdateGateGrad,
              rResetGateValue,
              rResetGateGrad,
              rPrevOutValue,
              rPrevOutGrad,
              rResetOutputGrad,
              hppl::gpu::backward[active_gate]);

  gateGrad[frameIdx + frameSize * 0] = rUpdateGateGrad;
  gateGrad[frameIdx + frameSize * 1] = rResetGateGrad;
  if (prevOutGrad) {
    prevOutGrad[frameIdx] = rPrevOutGrad;
  }
}

template<class OpStateGrad, class OpResetGrad>
void hl_gpu_gru_backward(OpStateGrad opStateGrad,
                         OpResetGrad opResetGrad,
                         hl_gru_value value,
                         hl_gru_grad  grad,
                         int frameSize,
                         int batchSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate) {
  dim3 threads;
  dim3 grid;
  if (batchSize == 1) {
    int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
    int frameBlocks = (frameSize + 1024 - 1) / 1024;
    threads = dim3(framePerBlock, 1);
    grid = dim3(frameBlocks, 1);
  } else {
    threads = dim3(32, 32);
    grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 32 - 1) / 32);
  }

  if (batchSize == 1) {
    KeGruBackwardStateGrad<OpStateGrad, /* isBatch= */false>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opStateGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.outputGrad, frameSize, batchSize, active_node);
  } else {
    KeGruBackwardStateGrad<OpStateGrad, /* isBatch= */true>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opStateGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.outputGrad, frameSize, batchSize, active_node);
  }

  if (value.prevOutValue && grad.prevOutGrad) {
    hl_matrix_mul(grad.gateGrad + 2*frameSize, HPPL_OP_N,
                  value.stateWeight, HPPL_OP_T,
                  grad.resetOutputGrad,
                  batchSize, frameSize, frameSize,
                  /*alpha = */ 1, /*beta = */ 0,
                  3*frameSize, frameSize, frameSize);
    if (grad.stateWeightGrad) {
      hl_matrix_mul(value.resetOutputValue, HPPL_OP_T,
                    grad.gateGrad + 2*frameSize, HPPL_OP_N,
                    grad.stateWeightGrad,
                    frameSize, frameSize, batchSize,
                    /*alpha = */ 1, /*beta = */ 1,
                    frameSize, 3*frameSize, frameSize);
    }
  }

  if (batchSize == 1) {
    KeGruBackwardResetGrad<OpResetGrad, /* isBatch= */false>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opResetGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.resetOutputGrad, frameSize, batchSize, active_gate);
  } else {
    KeGruBackwardResetGrad<OpResetGrad, /* isBatch= */true>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(opResetGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.resetOutputGrad, frameSize, batchSize, active_gate);
  }

  if (grad.prevOutGrad && value.prevOutValue) {
    hl_matrix_mul(grad.gateGrad, HPPL_OP_N,
                  value.gateWeight, HPPL_OP_T,
                  grad.prevOutGrad,
                  batchSize, frameSize, 2*frameSize,
                  /*alpha = */ 1, /*beta = */ 1,
                  3*frameSize, 2*frameSize, frameSize);
    if (grad.gateWeightGrad) {
      hl_matrix_mul(value.prevOutValue, HPPL_OP_T,
                    grad.gateGrad, HPPL_OP_N,
                    grad.gateWeightGrad,
                    frameSize, 2*frameSize, batchSize,
                    /*alpha = */ 1, /*beta = */ 1,
                    frameSize, 3*frameSize, 2*frameSize);
    }
  }

  CHECK_SYNC("hl_gpu_gru_backward failed");
}

#else

template<class OpResetOutput, class OpFinalOutput>
void hl_gpu_gru_forward(OpResetOutput opResetOutput,
                        OpFinalOutput opFinalOutput,
                        hl_gru_value value,
                        int frameSize,
                        int batchSize,
                        hl_activation_mode_t active_node,
                        hl_activation_mode_t active_gate) {}

template<class OpStateGrad, class OpResetGrad>
void hl_gpu_gru_backward(OpStateGrad opStateGrad,
                         OpResetGrad opResetGrad,
                         hl_gru_value value,
                         hl_gru_grad  grad,
                         int frameSize,
                         int batchSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate) {}

#endif

#endif /* HL_GPU_GRU_CUH_ */
