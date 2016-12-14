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


#include "hl_base.h"
#include "hl_cuda_cublas.h"
#include "hl_device_functions.cuh"
#include "hl_activation_functions.h"
#include "paddle/utils/Logging.h"

typedef hppl::Active<real>::forward  t_forward;
typedef hppl::Active<real>::backward t_backward;

bool hl_lstm_sequence_parallel(int frameSize) {
  if (frameSize == 32 || frameSize == 64) {
    return true;
  } else {
    return false;
  }
}

class frameValue {
public:
  real *value_;
  __device__ frameValue(real *value) : value_(value) {}
  template <int reversed, int frameSize>
  __device__ inline void init(int start, int length, int idx) {
    if (reversed == 0) {
      value_ += start * frameSize + idx;
    } else {
      value_ += (start + length - 1) * frameSize + idx;
    }
  }
  __device__ inline real *getPtr() const {return value_;}
  __device__ inline real getValue() {return *value_;}
  __device__ inline void setValue(real value) {*value_ = value;}
  template <int reversed, int frameSize>
  __device__ inline void nextFrame() {
    if (reversed == 0) {
      value_ += frameSize;
    } else {
      value_ -= frameSize;
    }
  }
};

__device__ __forceinline__
void ptx_sync(const int id, const int barriers) {
  asm volatile("bar.sync %0, %1;" : : "r"(id), "r"(barriers) : "memory");
}

__device__ __forceinline__
void ptx_arrive(const int id, const int barriers) {
  asm volatile("bar.arrive %0, %1;" : : "r"(id), "r"(barriers) : "memory");
}

template<int valueSize, int frameSize>
__device__ __forceinline__ real
forward_sequence(real value,
                 real *shValue,
                 real *state,
                 real *preOutput,
                 real *output,
                 real check,
                 int index,
                 t_forward activeNode,
                 t_forward activeGate,
                 t_forward activeState) {
  real out;
  real prevOut;
  real state_r;
  const int idx = index % frameSize;
  const int idy = index / frameSize;
  // assert(index < valueSize);

  if (idy == 0) {
    value = activeNode(value);
    shValue[index] = value;
  }
  if (idy == 1 || idy == 2) {
    state_r = state[idx];
    value += state_r * check;
    value = activeGate(value);
    shValue[index] = value;
  }
  ptx_sync(1, valueSize);
  if (idy == 3) {
    state_r = state[idx];
    state_r = state_r * shValue[idx + frameSize * 2];
    state_r += shValue[idx] * shValue[idx + frameSize];
    state[idx] = state_r;
    ptx_arrive(2, frameSize * 2);
    value += state_r * check;
    value = activeGate(value);
    shValue[index] = value;
    ptx_sync(3, frameSize * 2);
    prevOut = preOutput[idx];
    out = prevOut * value;
    output[idx] = out;
  }
  if (idy == 0) {
    ptx_sync(2, frameSize * 2);
    prevOut = state[idx];
     prevOut = activeState(prevOut);
    preOutput[idx] = prevOut;
    ptx_arrive(3, frameSize * 2);
  }
  return value;
}

#define     OUTPUT_BARRIER_ID               10
#define     OUTPUT_BARRIER_ID2              11
template<int valueSize, int frameSize, int reversed,
         int computeThreads, int blockSize>
__global__ void KeLstmForward(real *gateValue,
                              real *state,
                              real *output,
                              real *preOutput,
                              real *checkIg,
                              real *checkFg,
                              real *checkOg,
                              real *weight,
                              const int *starts,
                              hl_activation_mode_t active_node,
                              hl_activation_mode_t active_gate,
                              hl_activation_mode_t active_state) {
  __shared__ real shValue[valueSize];
  __shared__ real shState[frameSize];
  __shared__ real shPrevOutput[frameSize];
  __shared__ real shOutput[frameSize];

  const int index = threadIdx.x;
  int start = starts[blockIdx.x];
  int length = starts[blockIdx.x + 1] - start;

  /* init */
  real check;
  real value;
  frameValue frameGate(gateValue);
  frameValue frameState(state);
  frameValue frameOutput(output);
  frameValue framePreOutput(preOutput);
  if (index < valueSize) {
    const int idx = index % frameSize;
    const int idy = index / frameSize;
    frameGate.init<reversed, valueSize>(start, length, index);
    value = frameGate.getValue();
    if (idy == 0) {
      shState[idx] = 0.0;
    } else if (idy == 1) {
      check = checkIg[idx];
    } else if (idy == 2) {
      check = checkFg[idx];
    } else if (idy == 3) {
      check = checkOg[idx];
    }

    if (idy == 3) {
      frameState.init<reversed, frameSize>(start, length, idx);
      frameOutput.init<reversed, frameSize>(start, length, idx);
      framePreOutput.init<reversed, frameSize>(start, length, idx);
    }

    ptx_sync(1, valueSize);
  }

  for (int i = 0; i < length; ++i) {
    if (index < valueSize) {
      if (valueSize == 128) {
        if (i != 0) {
          ptx_sync(OUTPUT_BARRIER_ID2, blockSize);
          value += shValue[index];
        }
      }
      value = forward_sequence<valueSize, frameSize>(
        value, shValue, shState, shPrevOutput, shOutput, check, index,
        hppl::gpu::forward[active_node],
        hppl::gpu::forward[active_gate],
        hppl::gpu::forward[active_state]);
      const int idx = index % frameSize;
      const int idy = index / frameSize;
      if (valueSize == 128) {
        if (idy == 3) {
          ptx_arrive(OUTPUT_BARRIER_ID, frameSize + 128);
        }
      }
      if (valueSize == 256) {
        ptx_sync(OUTPUT_BARRIER_ID, valueSize);
      }
      frameGate.setValue(value);
      if (idy == 3) {
        frameState.setValue(shState[idx]);
        frameOutput.setValue(shOutput[idx]);
        framePreOutput.setValue(shPrevOutput[idx]);
        frameState.nextFrame<reversed, frameSize>();
        frameOutput.nextFrame<reversed, frameSize>();
        framePreOutput.nextFrame<reversed, frameSize>();
      }
      if (i != length - 1) {
        frameGate.nextFrame<reversed, valueSize>();
        value = frameGate.getValue();
      }
    }
    if (i != length - 1) {
      if (valueSize == 128) {
        if (valueSize <= index) {
          real B_r[frameSize];
          const int computeIdx = index - valueSize;
          if (i == 0) {
            #pragma unroll
            for (int n = 0; n < frameSize; n++) {
              B_r[n] = weight[n * valueSize + computeIdx];
            }
          }
          ptx_sync(OUTPUT_BARRIER_ID, frameSize + 128);
          real A_r[frameSize];
          for (int n = 0; n < frameSize; n++) {
            A_r[n] = shOutput[n];
          }
          real sum = 0.0f;
          for (int n = 0; n < frameSize; n++) {
            sum += A_r[n]*B_r[n];
          }
          shValue[computeIdx] = sum;
          ptx_arrive(OUTPUT_BARRIER_ID2, blockSize);
        }
      }
      if (valueSize == 256) {
        real B_r[frameSize];
        if (i == 0) {
          #pragma unroll
          for (int n = 0; n < frameSize; n++) {
            B_r[n] = weight[n * valueSize + index];
          }
        }
        real sum = 0.0f;
        for (int n = 0; n < frameSize; n++) {
          sum += shOutput[n]*B_r[n];
        }
        value += sum;
      }
    }
  }
}

void hl_lstm_parallel_forward(real *gateValue,
                              real *stateValue,
                              real *preOutputValue,
                              real *outputValue,
                              real *checkIg,
                              real *checkFg,
                              real *checkOg,
                              real *weight,
                              const int *sequence,
                              int frameSize,
                              int numSequences,
                              bool reversed,
                              hl_activation_mode_t active_node,
                              hl_activation_mode_t active_gate,
                              hl_activation_mode_t active_state) {
  CHECK(frameSize == 32 || frameSize == 64);
  dim3 grid(numSequences, 1);
  if (!reversed) {
    if (frameSize == 32) {
      KeLstmForward<128, 32, 0, 128, 256>
               <<<grid, 256, 0, STREAM_DEFAULT>>>
               (gateValue, stateValue, outputValue, preOutputValue,
               checkIg, checkFg, checkOg, weight, sequence,
               active_node, active_gate, active_state);
    } else if (frameSize == 64) {
      KeLstmForward<256, 64, 0, 256, 256>
               <<<grid, 256, 0, STREAM_DEFAULT>>>
               (gateValue, stateValue, outputValue, preOutputValue,
               checkIg, checkFg, checkOg, weight, sequence,
               active_node, active_gate, active_state);
    }
  } else {
    if (frameSize == 32) {
      KeLstmForward<128, 32, 1, 128, 256>
               <<<grid, 256, 0, STREAM_DEFAULT>>>
               (gateValue, stateValue, outputValue, preOutputValue,
               checkIg, checkFg, checkOg, weight, sequence,
               active_node, active_gate, active_state);
    } else if (frameSize == 64) {
      KeLstmForward<256, 64, 1, 256, 256>
               <<<grid, 256, 0, STREAM_DEFAULT>>>
               (gateValue, stateValue, outputValue, preOutputValue,
               checkIg, checkFg, checkOg, weight, sequence,
               active_node, active_gate, active_state);
    }
  }
  CHECK_SYNC("hl_lstm_parallel_forward failed");
}

__device__ __forceinline__
void transpose_32x32(real a[], const int idx) {
  int addr = idx % 32;
  #pragma unroll
  for (int k = 1; k < 32; k++) {
    // rSrc[k] = __shfl(rSrc[k], (threadIdx.x + k) % 32, 32);
    addr = __shfl(addr, (idx + 1) % 32, 32);
    a[k] = __shfl(a[k], addr, 32);
  }

  #pragma unroll
  for (int tid = 0; tid < 31; tid++) {
    real tmp = (idx > tid) ? a[0] : a[1];
    #pragma unroll
    for (int k = 31; k > 0; k--) {
      a[(k + 1) % 32] = (idx > tid) ? a[k] : a[(k + 1) % 32];
    }
    a[1] = tmp;
  }

  addr = (32 - idx) % 32;
  #pragma unroll
  for (int k = 0; k < 32; k++) {
    a[k] = __shfl(a[k], addr, 32);
    addr = __shfl(addr, (idx + 31) % 32, 32);
  }
}

template<int valueSize, int frameSize>
__device__ void
backward_sequence(real rGateValue,
                  real rOutputGrad,
                  real rPreOutputValue,
                  real &rGateGrad,
                  real &rStateGrad,
                  real *shStateGrad,
                  real *shStateValue,
                  real *shGateValue,
                  real rCheck,
                  real &rGateValuePrev,
                  int index,
                  t_backward activeNode,
                  t_backward activeGate,
                  t_backward activeState) {
  const int frameIdx = index % frameSize;
  const int frameIdy = index / frameSize;
  if (frameIdy == 3) {
    real rPrevOutputGrad;
    rPrevOutputGrad = rOutputGrad * rGateValue;
    rStateGrad = activeState(rPrevOutputGrad, rPreOutputValue);
    rGateGrad = rOutputGrad * rPreOutputValue;
    rGateGrad = activeGate(rGateGrad, rGateValue);
    rStateGrad += rGateGrad * rCheck;
    shStateGrad[index] = rStateGrad;
    ptx_arrive(3, valueSize);
  } else if (frameIdy == 1) {
    shGateValue[frameIdx + frameSize] = rGateValue;
    rStateGrad = rGateGrad * rCheck;
    shStateGrad[index] = rStateGrad;
    ptx_sync(3, valueSize);
    rStateGrad += shStateGrad[frameIdx + frameSize *2];
    rStateGrad += shStateGrad[frameIdx + frameSize *3];
    rGateGrad = rStateGrad * shGateValue[frameIdx];
    rGateGrad = activeGate(rGateGrad, rGateValue);
  } else if (frameIdy == 2) {
    rStateGrad = rStateGrad * rGateValuePrev;
    rStateGrad += rGateGrad * rCheck;
    shStateGrad[index] = rStateGrad;
    ptx_sync(3, valueSize);
    rStateGrad += shStateGrad[frameIdx + frameSize];
    rStateGrad += shStateGrad[frameIdx + frameSize *3];
    rGateValuePrev = rGateValue;
    rGateGrad = rStateGrad * shStateValue[frameIdx];
    rGateGrad = activeGate(rGateGrad, rGateValue);
  } else if (frameIdy == 0) {
    shGateValue[frameIdx] = rGateValue;
    ptx_sync(3, valueSize);
    rStateGrad = shStateGrad[frameIdx + frameSize];
    rStateGrad += shStateGrad[frameIdx + frameSize *2];
    rStateGrad += shStateGrad[frameIdx + frameSize *3];
    rGateGrad = rStateGrad * shGateValue[frameIdx + frameSize];
    rGateGrad = activeNode(rGateGrad, rGateValue);
  }
}

template<int valueSize, int frameSize>
__device__ void load_weight(real rWeight[], real *weight, const int index) {
  if (valueSize == 128) {
    weight += index;
    #pragma unroll
    for (int n = 0; n < frameSize; n++) {
      rWeight[n] = weight[n*valueSize];
    }
    transpose_32x32(rWeight, index % 32);
  }
  if (valueSize == 256) {
    int id = (index / 32) % 2;
    weight += index - id * 32 + id * 32 * valueSize;
    #pragma unroll
    for (int n = 0; n < 32; n++) {
      rWeight[n] = weight[n*valueSize];
      rWeight[n + 32] = weight[n*valueSize + 32];
    }
    transpose_32x32(rWeight, index % 32);
    transpose_32x32(&rWeight[32], index % 32);
  }
}

template<int valueSize, int frameSize, int reversed>
__global__ void KeLstmBackward(real *gateValue,
                               real *gateGrad,
                               real *stateValue,
                               real *stateGrad,       /* do not need save */
                               real *preOutputValue,
                               real *preOutputGrad,   /* do not need save */
                               real *checkIg,
                               real *checkIgGrad,
                               real *checkFg,
                               real *checkFgGrad,
                               real *checkOg,
                               real *checkOgGrad,
                               real *outputGrad,
                               real *weightValue,
                               const int *starts,
                               hl_activation_mode_t active_node,
                               hl_activation_mode_t active_gate,
                               hl_activation_mode_t active_state) {
  __shared__ real shGateValue[valueSize];
  __shared__ real shStateGrad[valueSize];
  __shared__ real shStateValue[frameSize];
  __shared__ real shGateGrad[4][frameSize];
  __shared__ real shOutputGrad[4][frameSize];
  const int index = threadIdx.x;
  int start = starts[blockIdx.x];
  int length = starts[blockIdx.x + 1] - start;

  const int frameIdx = index % frameSize;
  const int frameIdy = index / frameSize;
  real rCheck;
  real rCheckGrad;
  real rGateGrad;
  real rStateGrad;
  real rGateValuePrev;
  real rPreOutputValue;
  real rOutputGrad;
  real rGateValue;
  real rStateValue;

  frameValue frameGateValue(gateValue);
  frameValue frameGateGrad(gateGrad);
  frameValue framePreOutputValue(preOutputValue);
  frameValue frameStateValue(stateValue);
  frameValue frameOutputGrad(outputGrad);
  if (frameIdy == 0) {
  } else if (frameIdy == 1) {
    rCheck = checkIg[frameIdx];
  } else if (frameIdy == 2) {
    rCheck = checkFg[frameIdx];
    rGateValuePrev = 0.0;
    rStateGrad = 0.0;
  } else if (frameIdy == 3) {
    rCheck = checkOg[frameIdx];
    framePreOutputValue.init<!reversed, frameSize>(start, length, frameIdx);
    frameOutputGrad.init<!reversed, frameSize>(start, length, frameIdx);
    rOutputGrad = frameOutputGrad.getValue();
    rPreOutputValue = framePreOutputValue.getValue();
    frameStateValue.init<!reversed, frameSize>(start, length, frameIdx);
    rStateValue = frameStateValue.getValue();
  }

  frameGateValue.init<!reversed, valueSize>(start, length, index);
  frameGateGrad.init<!reversed, valueSize>(start, length, index);
  rGateValue = frameGateValue.getValue();
  rGateGrad = 0.0;
  rCheckGrad = 0.0;

  real B_r[frameSize];
  load_weight<valueSize, frameSize>(B_r, weightValue, index);

  for (int i = 0; i < length; ++i) {
    if (frameIdy == 3) {
      if (i != length -1) {
        frameStateValue.nextFrame<!reversed, frameSize>();
        shStateValue[frameIdx] = frameStateValue.getValue();
      } else {
        shStateValue[frameIdx] = 0.0;
      }
    }
    backward_sequence<valueSize, frameSize>(
        rGateValue, rOutputGrad, rPreOutputValue, rGateGrad,
        rStateGrad, shStateGrad, shStateValue, shGateValue,
        rCheck, rGateValuePrev, index,
        hppl::gpu::backward[active_node],
        hppl::gpu::backward[active_gate],
        hppl::gpu::backward[active_state]);
    if (frameIdy == 3) {
      rCheckGrad += rGateGrad * rStateValue;
      rStateValue = shStateValue[frameIdx];
    }

    frameGateGrad.setValue(rGateGrad);
    frameGateGrad.nextFrame<!reversed, valueSize>();

    if (i != length - 1) {
      if (frameIdy == 3) {
        framePreOutputValue.nextFrame<!reversed, frameSize>();
        rPreOutputValue = framePreOutputValue.getValue();
        frameOutputGrad.nextFrame<!reversed, frameSize>();
        rOutputGrad = frameOutputGrad.getValue();
      } else if (frameIdy == 2) {
        rCheckGrad += rGateGrad * shStateValue[frameIdx];
      } else if (frameIdy == 1) {
        rCheckGrad += rGateGrad * shStateValue[frameIdx];
      }

      frameGateValue.nextFrame<!reversed, valueSize>();
      rGateValue = frameGateValue.getValue();
      shGateGrad[frameIdy][frameIdx] = rGateGrad;
      if (valueSize == 128) {
        real sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < frameSize; n++) {
          sum += shGateGrad[frameIdy][n]*B_r[n];
        }
        if (frameIdy == 3) {
          rOutputGrad += sum;
        } else {
          shOutputGrad[frameIdy][frameIdx] = sum;
        }
      }
      if (valueSize == 256) {
        ptx_sync(5, valueSize);
        real A_r[frameSize];
        for (int n = 0; n < frameSize; n++) {
          A_r[n] = shGateGrad[frameIdy][n];
        }
        real sum = 0.0f;
        for (int n = 0; n < frameSize; n++) {
          sum += A_r[n]*B_r[n];
        }
        if (frameIdy == 3) {
          rOutputGrad += sum;
        } else {
          shOutputGrad[frameIdy][frameIdx] = sum;
        }
      }

      if (frameIdy == 3) {
        ptx_sync(6, valueSize);
        #pragma unroll
        for (int i = 0; i < 3; i ++) {
          rOutputGrad += shOutputGrad[i][frameIdx];
        }
      } else {
        ptx_arrive(6, valueSize);
      }
    }
  }

  /* TODO: Temporary save & merger in another kernel */
  if (frameIdy == 1) {
    if (checkIgGrad) paddle::paddleAtomicAdd(checkIgGrad+frameIdx, rCheckGrad);
  } else if (frameIdy == 2) {
    if (checkFgGrad) paddle::paddleAtomicAdd(checkFgGrad+frameIdx, rCheckGrad);
  } else if (frameIdy == 3) {
    if (checkOgGrad) paddle::paddleAtomicAdd(checkOgGrad+frameIdx, rCheckGrad);
  }
}

void hl_lstm_parallel_backward_data(real *gateValue,
                                    real *gateGrad,
                                    real *stateValue,
                                    real *stateGrad,
                                    real *preOutputValue,
                                    real *preOutputGrad,
                                    real *outputGrad,
                                    real *checkIg,
                                    real *checkIgGrad,
                                    real *checkFg,
                                    real *checkFgGrad,
                                    real *checkOg,
                                    real *checkOgGrad,
                                    real *weight,
                                    const int *sequence,
                                    int frameSize,
                                    int numSequences,
                                    bool reversed,
                                    hl_activation_mode_t active_node,
                                    hl_activation_mode_t active_gate,
                                    hl_activation_mode_t active_state) {
  CHECK(frameSize == 32 || frameSize == 64 ||
        frameSize == 128 || frameSize == 256);
  dim3 grid(numSequences, 1);
  if (!reversed) {
    if (frameSize == 32) {
      KeLstmBackward<128, 32, 0><<<grid, 128, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    } else if (frameSize == 64) {
      KeLstmBackward<256, 64, 0><<<grid, 256, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    } else if (frameSize == 128) {
      KeLstmBackward<512, 128, 0><<<grid, 512, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    } else if (frameSize == 256) {
      KeLstmBackward<1024, 256, 0><<<grid, 1024, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    }
  } else {
    if (frameSize == 32) {
      KeLstmBackward<128, 32, 1><<<grid, 128, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    } else if (frameSize == 64) {
      KeLstmBackward<256, 64, 1><<<grid, 256, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    } else if (frameSize == 128) {
      KeLstmBackward<512, 128, 1><<<grid, 512, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    } else if (frameSize == 256) {
      KeLstmBackward<1024, 256, 1><<<grid, 1024, 0, STREAM_DEFAULT>>>
          (gateValue, gateGrad, stateValue, stateGrad, preOutputValue,
          preOutputGrad, checkIg, checkIgGrad, checkFg, checkFgGrad, checkOg,
          checkOgGrad, outputGrad, weight, sequence,
          active_node, active_gate, active_state);
    }
  }
  CHECK_SYNC("hl_lstm_parallel_backward_data");
}

template<int B_X, int B_Y>
__global__ void KeSetGradZero(real *gateGrad,
    const int *starts, int valueSize, int numSequences, bool reversed) {
  // const int tid = threadIdx.x;

  const int frameIdx = blockIdx.x * B_X + threadIdx.x;
  const int numSeqId = blockIdx.y * B_Y + threadIdx.y;

  if (numSeqId >= numSequences || frameIdx >= valueSize) return;

  if (!reversed) {
    int seqId = starts[numSeqId];
    gateGrad[seqId * valueSize + frameIdx] = 0.0;
  } else {
    int seqId = starts[numSeqId + 1] - 1;
    gateGrad[seqId * valueSize + frameIdx] = 0.0;
  }
}

void hl_lstm_parallel_backward_weight(real *weightGrad,
                                      real *outputValue,
                                      real *gateGrad,
                                      const int *sequence,
                                      int frameSize,
                                      int batchSize,
                                      int numSequences,
                                      bool reversed) {
  int valueSize = 4 * frameSize;
  dim3 threads(32, 32);
  dim3 grid((valueSize + 32 - 1) / 32, (numSequences + 32 - 1) / 32);
  KeSetGradZero<32, 32><<<grid, threads, 0, STREAM_DEFAULT>>>
           (gateGrad, sequence, valueSize, numSequences, reversed);

  if (!reversed) {
    hl_matrix_mul(outputValue,
      HPPL_OP_T, gateGrad + valueSize, HPPL_OP_N, weightGrad,
      frameSize, valueSize, batchSize - 1,
      1.0, 1.0);
  } else {
    hl_matrix_mul(outputValue + frameSize,
      HPPL_OP_T, gateGrad, HPPL_OP_N, weightGrad,
      frameSize, valueSize, batchSize - 1,
      1.0, 1.0);
  }
  CHECK_SYNC("hl_lstm_parallel_backward_weight");
}
