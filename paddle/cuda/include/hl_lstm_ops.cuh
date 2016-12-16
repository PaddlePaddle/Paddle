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


#ifndef HL_LSTM_OPS_CUH_
#define HL_LSTM_OPS_CUH_

#ifdef __CUDA_ARCH__
#define INLINE   __device__ inline
#else
#define INLINE   inline
#endif

namespace hppl {

namespace forward {
class lstm {
public:
  /**
   * @param   valueIn     input
   * @param   valueIg     input gate
   * @param   valueFg     forget gate
   * @param   valueOg     output gate
   * @param   prevState   previous state
   * @param   state       current state
   * @param   stateAtv    state active
   * @param   output      output
   * @param   checkI      check input gate
   * @param   checkF      check forget gate
   * @param   checkO      check output gate
   * @param   actInput    forward function of input
   * @param   actGate     forward function of gate
   * @param   actState    forward function of state
   */
  INLINE void operator()(real &valueIn,
                         real &valueIg,
                         real &valueFg,
                         real &valueOg,
                         real &prevState,
                         real &state,
                         real &stateAtv,
                         real &output,
                         real &checkI,
                         real &checkF,
                         real &checkO,
                         Active<real>::forward actInput,
                         Active<real>::forward actGate,
                         Active<real>::forward actState) {
    valueIn = actInput(valueIn);
    valueIg = actGate(valueIg + prevState * checkI);
    valueFg = actGate(valueFg + prevState * checkF);
    state = valueIn * valueIg + prevState * valueFg;
    valueOg = actGate(valueOg + state * checkO);
    stateAtv = actState(state);
    output = valueOg * stateAtv;
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  INLINE void operator()(__m256 &valueIn,
                         __m256 &valueIg,
                         __m256 &valueFg,
                         __m256 &valueOg,
                         __m256 &prevState,
                         __m256 &state,
                         __m256 &stateAtv,
                         __m256 &output,
                         __m256 &checkI,
                         __m256 &checkF,
                         __m256 &checkO,
                         Active<__m256>::forward actInput,
                         Active<__m256>::forward actGate,
                         Active<__m256>::forward actState) {
    valueIn = actInput(valueIn);
    valueIg = actGate(
      _mm256_add_ps(valueIg, _mm256_mul_ps(prevState, checkI)));
    valueFg = actGate(
      _mm256_add_ps(valueFg, _mm256_mul_ps(prevState, checkF)));
    state = _mm256_add_ps(_mm256_mul_ps(valueIn, valueIg)
        , _mm256_mul_ps(prevState, valueFg));
    valueOg = actGate(_mm256_add_ps(valueOg, _mm256_mul_ps(state, checkO)));
    stateAtv = actState(state);
    output = _mm256_mul_ps(valueOg, stateAtv);
  }
#endif
#endif
};
}  // namespace forward

namespace backward {
class lstm {
public:
  /**
   * @param   valueIn         input
   * @param   valueIg         input gate
   * @param   valueFg         forget gate
   * @param   valueOg         output gate
   * @param   gradIn          input grad
   * @param   gradIg          input gate grad
   * @param   gradFg          forget gate grad
   * @param   gradOg          output gate grad
   * @param   prevState       previous state value
   * @param   prevStateGrad   previous state grad
   * @param   state           current state value
   * @param   stateGrad       current state grad
   * @param   stateAtv        state active
   * @param   outputGrad      output grad
   * @param   checkI          check input gate
   * @param   checkF          check forget gate
   * @param   checkO          check output gate
   * @param   checkIGrad      check input gate grad
   * @param   checkFGrad      check forget gate grad
   * @param   checkOGrad      check output gate grad
   * @param   actInput        backward function of input
   * @param   actGate         backward function of gate
   * @param   actState        backward function of state
   */
  INLINE void operator()(real &valueIn,
                         real &valueIg,
                         real &valueFg,
                         real &valueOg,
                         real &gradIn,
                         real &gradIg,
                         real &gradFg,
                         real &gradOg,
                         real &prevState,
                         real &prevStateGrad,
                         real &state,
                         real &stateGrad,
                         real &stateAtv,
                         real &outputGrad,
                         real &checkI,
                         real &checkF,
                         real &checkO,
                         real &checkIGrad,
                         real &checkFGrad,
                         real &checkOGrad,
                         Active<real>::backward actInput,
                         Active<real>::backward actGate,
                         Active<real>::backward actState) {
    gradOg = actGate(outputGrad * stateAtv, valueOg);
    stateGrad += actState(outputGrad * valueOg, stateAtv) + gradOg * checkO;
    gradIn = actInput(stateGrad * valueIg, valueIn);
    gradIg = actGate(stateGrad * valueIn, valueIg);
    gradFg = actGate(stateGrad * prevState, valueFg);
    prevStateGrad = gradIg * checkI + gradFg * checkF + stateGrad * valueFg;
    checkIGrad = gradIg * prevState;
    checkFGrad = gradFg * prevState;
    checkOGrad = gradOg * state;
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  INLINE void operator()(__m256 &valueIn,
                         __m256 &valueIg,
                         __m256 &valueFg,
                         __m256 &valueOg,
                         __m256 &gradIn,
                         __m256 &gradIg,
                         __m256 &gradFg,
                         __m256 &gradOg,
                         __m256 &prevState,
                         __m256 &prevStateGrad,
                         __m256 &state,
                         __m256 &stateGrad,
                         __m256 &stateAtv,
                         __m256 &outputGrad,
                         __m256 &checkI,
                         __m256 &checkF,
                         __m256 &checkO,
                         __m256 &checkIGrad,
                         __m256 &checkFGrad,
                         __m256 &checkOGrad,
                         Active<__m256>::backward actInput,
                         Active<__m256>::backward actGate,
                         Active<__m256>::backward actState) {
    gradOg = actGate(_mm256_mul_ps(outputGrad, stateAtv), valueOg);
    stateGrad = _mm256_add_ps(
      actState(_mm256_mul_ps(outputGrad, valueOg), stateAtv), stateGrad);
    stateGrad = _mm256_add_ps(_mm256_mul_ps(gradOg, checkO), stateGrad);
    gradIn = actInput(_mm256_mul_ps(stateGrad, valueIg), valueIn);
    gradIg = actGate(_mm256_mul_ps(stateGrad, valueIn), valueIg);
    gradFg = actGate(_mm256_mul_ps(stateGrad, prevState), valueFg);
    prevStateGrad = _mm256_add_ps(
      _mm256_mul_ps(gradIg, checkI), _mm256_mul_ps(gradFg, checkF));
    prevStateGrad = _mm256_add_ps(
      _mm256_mul_ps(stateGrad, valueFg), prevStateGrad);
    checkIGrad = _mm256_mul_ps(gradIg, prevState);
    checkFGrad = _mm256_mul_ps(gradFg, prevState);
    checkOGrad = _mm256_mul_ps(gradOg, state);
  }
#endif
#endif
};
}  // namespace backward
}  // namespace hppl

#endif /* HL_LSTM_OPS_CUH_ */
