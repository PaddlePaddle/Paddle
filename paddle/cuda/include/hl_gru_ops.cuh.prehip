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


#ifndef HL_GRU_OPS_CUH_
#define HL_GRU_OPS_CUH_

#ifdef __CUDA_ARCH__
#define INLINE   __device__ inline
#else
#define INLINE   inline
#endif

namespace hppl {

namespace forward {
class gru_resetOutput {
public:
  /**
   * @param[in,out]   valueUpdateGate  update gate
   * @param[in,out]   valueResetGate   reset gate
   * @param[in]       prevOut          previous output
   * @param[out]      valueResetOutput intermediate value for frame state
   * @param[in]       actGate          forward function of gate
   */
  INLINE void operator()(real &valueUpdateGate,
                         real &valueResetGate,
                         real &prevOut,
                         real &valueResetOutput,
                         Active<real>::forward actGate) {
    valueUpdateGate  = actGate(valueUpdateGate);
    valueResetGate   = actGate(valueResetGate);
    valueResetOutput = prevOut * valueResetGate;
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  INLINE void operator()(__m256 &valueUpdateGate,
                         __m256 &valueResetGate,
                         __m256 &prevOut,
                         __m256 &valueResetOutput,
                         Active<__m256>::forward actGate) {
    valueUpdateGate  = actGate(valueUpdateGate);
    valueResetGate   = actGate(valueResetGate);
    valueResetOutput = _mm256_mul_ps(prevOut, valueResetGate);
  }
#endif
#endif
};

class gru_finalOutput {
public:
  /**
   * @param[in]     valueUpdateGate   update gate
   * @param[in,out] valueFrameState   frame state ({\tilde{h}_t})
   * @param[in]     prevOut           previous output
   * @param[out]    valueOutput       output
   * @param[in]     actInput          forward function of node
   */
  INLINE void operator()(real &valueUpdateGate,
                         real &valueFrameState,
                         real &prevOut,
                         real &valueOutput,
                         Active<real>::forward actInput ) {
    valueFrameState = actInput(valueFrameState);
    valueOutput = prevOut - (valueUpdateGate * prevOut) +
      (valueUpdateGate * valueFrameState);
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  INLINE void operator()(__m256 &valueUpdateGate,
                         __m256 &valueFrameState,
                         __m256 &prevOut,
                         __m256 &valueOutput,
                         Active<__m256>::forward actInput) {
    valueFrameState = actInput(valueFrameState);
    valueOutput = _mm256_add_ps(
      _mm256_sub_ps(prevOut, _mm256_mul_ps(valueUpdateGate, prevOut)),
      _mm256_mul_ps(valueUpdateGate, valueFrameState));
  }
#endif
#endif
};
}  // namespace forward

namespace backward {
class gru_stateGrad {
public:
  /**
   * @param[in]     valueUpdateGate   update gate value
   * @param[out]    gradUpdateGate    update gate grad
   * @param[in]     valueFrameState   frame state value
   * @param[out]    gradFrameState    frame state grad
   * @param[in]     valuePrevOut      previous output value
   * @param[in,out] gradPrevOut       previous output grad
   * @param[in]     gradOutput        output grad
   * @param[in]     actInput          backward function of frame state
   */
  INLINE void operator()(real &valueUpdateGate,
                         real &gradUpdateGate,
                         real &valueFrameState,
                         real &gradFrameState,
                         real &valuePrevOut,
                         real &gradPrevOut,
                         real &gradOutput,
                         Active<real>::backward actInput) {
    gradUpdateGate = (gradOutput * valueFrameState);
    gradUpdateGate -= (gradOutput * valuePrevOut);
    gradPrevOut -= (gradOutput * valueUpdateGate);
    gradPrevOut += gradOutput;
    gradFrameState = actInput(gradOutput * valueUpdateGate, valueFrameState);
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  INLINE void operator()(__m256 &valueUpdateGate,
                         __m256 &gradUpdateGate,
                         __m256 &valueFrameState,
                         __m256 &gradFrameState,
                         __m256 &valuePrevOut,
                         __m256 &gradPrevOut,
                         __m256 &gradOutput,
                         Active<__m256>::backward actInput) {
    gradUpdateGate = _mm256_mul_ps(gradOutput, valueFrameState);
    gradUpdateGate = _mm256_sub_ps(
      gradUpdateGate, _mm256_mul_ps(gradOutput, valuePrevOut));
    gradPrevOut = _mm256_add_ps(
      _mm256_sub_ps(gradPrevOut, _mm256_mul_ps(gradOutput, valueUpdateGate)),
      gradOutput);
    gradFrameState = actInput(
      _mm256_mul_ps(gradOutput, valueUpdateGate), valueFrameState);
  }
#endif
#endif
};

class gru_resetGrad {
public:
  /**
   * @param[in]     valueUpdateGate   update gate value
   * @param[in,out] gradUpdateGate    update gate grad
   * @param[in]     valueResetGate    reset gate value
   * @param[out]    gradResetGate     reset gate grad
   * @param[in]     valuePrevOut      previous output value
   * @param[in,out] gradPrevOut       previous output grad
   * @param[in]     gradResetOutput   reset output grad (temp val)
   * @param[in]     actGate           backward function of gate
   */
  INLINE void operator()(real &valueUpdateGate,
                         real &gradUpdateGate,
                         real &valueResetGate,
                         real &gradResetGate,
                         real &valuePrevOut,
                         real &gradPrevOut,
                         real &gradResetOutput,
                         Active<real>::backward actGate) {
    gradResetGate = (gradResetOutput * valuePrevOut);
    gradPrevOut += (gradResetOutput * valueResetGate);
    gradUpdateGate = actGate(gradUpdateGate, valueUpdateGate);
    gradResetGate  = actGate(gradResetGate , valueResetGate);
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  INLINE void operator()(__m256 &valueUpdateGate,
                         __m256 &gradUpdateGate,
                         __m256 &valueResetGate,
                         __m256 &gradResetGate,
                         __m256 &valuePrevOut,
                         __m256 &gradPrevOut,
                         __m256 &gradResetOutput,
                         Active<__m256>::backward actGate) {
    gradResetGate = _mm256_mul_ps(gradResetOutput, valuePrevOut);
    gradPrevOut = _mm256_add_ps(
      gradPrevOut, _mm256_mul_ps(gradResetOutput, valueResetGate));
    gradUpdateGate = actGate(gradUpdateGate, valueUpdateGate);
    gradResetGate  = actGate(gradResetGate , valueResetGate);
  }
#endif
#endif
};
}  // namespace backward
}  // namespace hppl

#endif /* HL_GRU_OPS_CUH_ */
