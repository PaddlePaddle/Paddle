/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct BaseCudaActiveFunctor {
  using ELEMENT_TYPE = T;
  using AttrPair = std::vector<std::pair<const char*, float*>>;
  AttrPair GetAttrs() { return AttrPair(); }
};

// For forward, args[0] means the input x;
// For backward, args[0] means the input dout, args[1] means the input x or out,
// which depends on the FwdDeps;
/********************Relu Begin********************/
template <typename T>
struct CudaReluFunctor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > zero ? args[0] : zero;
  }
};

template <typename T>
struct CudaReluGradFunctor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > zero ? args[0] : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Relu End********************/

/********************LeakyRelu Begin********************/
template <typename T>
struct CudaLeakyReluFunctor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > zero ? args[0] : static_cast<T>(alpha) * args[0];
  }
};

template <typename T>
struct CudaLeakyReluGradFunctor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > zero ? args[0] : static_cast<T>(alpha) * args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************LeakyRelu End********************/

/********************Sigmoid Begin********************/
template <typename T>
struct CudaSigmoidFunctor : public BaseCudaActiveFunctor<T> {
  // CT means Compute Type
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(one / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSigmoidGradFunctor : public BaseCudaActiveFunctor<T> {
  T one = static_cast<T>(1.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1] * (one - args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Sigmoid End********************/

/********************LogSigmoid Begin********************/
template <typename T>
struct CudaLogSigmoidFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    CT temp = x > zero ? zero : -x;
    return T(-temp - log(exp(-temp) + exp(-x - temp)));
  }
};

template <typename T>
struct CudaLogSigmoidGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    CT temp = x > zero ? zero : -x;
    return T(dout * (exp(-x - temp) / (exp(-temp) + exp(-x - temp))));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************LogSigmoid End********************/

/********************Atan Begin********************/
template <typename T>
struct CudaAtanFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(atan(x));
  }
};

template <typename T>
struct CudaAtanGradFunctor : public BaseCudaActiveFunctor<T> {
  T one = static_cast<T>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (one + args[1] * args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Atan End********************/

/********************SoftShrink Begin********************/
template <typename T>
struct CudaSoftShrinkFunctor : public BaseCudaActiveFunctor<T> {
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T l = static_cast<T>(lambda);
    T temp1 = static_cast<T>(x > l);
    T temp2 = static_cast<T>(x < -l);
    return temp1 * (x - l) + temp2 * (x + l);
  }
};

template <typename T>
struct CudaSoftShrinkGradFunctor : public BaseCudaActiveFunctor<T> {
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[1];
    T l = static_cast<T>(lambda);
    T temp1 = static_cast<T>(x > l);
    T temp2 = static_cast<T>(x < -l);
    return args[0] * (temp1 + temp2);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************SoftShrink End********************/

/********************Ceil Begin********************/
template <typename T>
struct CudaCeilFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(ceil(x));
  }
};
/********************Ceil End********************/

/********************Floor Begin********************/
template <typename T>
struct CudaFloorFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(floor(x));
  }
};
/********************Floor End********************/

/********************Round Begin********************/
template <typename T>
struct CudaRoundFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(round(x));
  }
};
/********************Floor End********************/

/********************Zero Begin********************/
template <typename T>
struct CudaZeroGradFunctor : public BaseCudaActiveFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return static_cast<T>(0.0f);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kNoDeps; }
};
/********************Zero End********************/

/********************Cos Begin********************/
template <typename T>
struct CudaCosFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(cos(x));
  }
};

template <typename T>
struct CudaCosGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(-dout * sin(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Cos End********************/

/********************Sin Begin********************/
template <typename T>
struct CudaSinFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(sin(x));
  }
};

template <typename T>
struct CudaSinGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout * cos(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Sin End********************/

/********************Tan Begin********************/
template <typename T>
struct CudaTanFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(tan(x));
  }
};

template <typename T>
struct CudaTanGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout / (cos(x) * cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Tan End********************/

/********************Asin Begin********************/
template <typename T>
struct CudaAsinFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(asin(x));
  }
};

template <typename T>
struct CudaAsinGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Asin End********************/

/********************Acos Begin********************/
template <typename T>
struct CudaAcosFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(acos(x));
  }
};

template <typename T>
struct CudaAcosGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(-dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Acos End********************/

/********************Cosh Begin********************/
template <typename T>
struct CudaCoshFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(cosh(x));
  }
};

template <typename T>
struct CudaCoshGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout * sinh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Cosh End********************/

/********************Sinh Begin********************/
template <typename T>
struct CudaSinhFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(sinh(x));
  }
};

template <typename T>
struct CudaSinhGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout * cosh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Sinh End********************/

/********************Reciprocal Begin********************/
template <typename T>
struct CudaReciprocalFunctor : public BaseCudaActiveFunctor<T> {
  T one = static_cast<T>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    return one / args[0];
  }
};

template <typename T>
struct CudaReciprocalGradFunctor : public BaseCudaActiveFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return -args[0] * args[1] * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Reciprocal End********************/

/********************Log1p Begin********************/
template <typename T>
struct CudaLog1pFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(log(one + x));
  }
};

template <typename T>
struct CudaLog1pGradFunctor : public BaseCudaActiveFunctor<T> {
  T one = static_cast<T>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (one + args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Log1p End********************/

/********************Log2 Begin********************/
template <typename T>
struct CudaLog2Functor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(log2(x));
  }
};

template <typename T>
struct CudaLog2GradFunctor : public BaseCudaActiveFunctor<T> {
  T log_two = static_cast<T>(log(2));
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (args[1] * log_two);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Log2 End********************/

/********************Log10 Begin********************/
template <typename T>
struct CudaLog10Functor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(log10(x));
  }
};

template <typename T>
struct CudaLog10GradFunctor : public BaseCudaActiveFunctor<T> {
  T log_ten = static_cast<T>(log(10));
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (args[1] * log_ten);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Log10 End********************/

/********************BRelu Begin********************/
template <typename T>
struct CudaBReluFunctor : public BaseCudaActiveFunctor<T> {
  float t_min;
  float t_max;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    return (x > t_min_cast && x < t_max_cast)
               ? x
               : (x <= t_min_cast ? t_min_cast : t_max_cast);
  }
};

template <typename T>
struct CudaBReluGradFunctor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float t_min;
  float t_max;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T dout = args[0];
    T x = args[1];
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    return (x <= t_min_cast || x >= t_max_cast) ? zero : dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************BRelu End********************/

/********************SoftRelu Begin********************/
template <typename T>
struct CudaSoftReluFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  float threshold;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    CT t = static_cast<CT>(threshold);
    CT temp = (x > -t && x < t) ? x : (x <= -t ? -t : t);
    return T(log(one + exp(temp)));
  }
};

template <typename T>
struct CudaSoftReluGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  float threshold;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT out = static_cast<CT>(args[1]);
    CT t = static_cast<CT>(threshold);
    return (out <= -t || out >= t) ? static_cast<T>(0.0f)
                                   : T(dout * (one - exp(-out)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************SoftRelu End********************/

/********************STanh Begin********************/
template <typename T>
struct CudaSTanhFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  float scale_a;
  float scale_b;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    CT a = static_cast<CT>(scale_a);
    CT b = static_cast<CT>(scale_b);
    return T(b * tanh(a * x));
  }
};

template <typename T>
struct CudaSTanhGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  float scale_a;
  float scale_b;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    CT a = static_cast<CT>(scale_a);
    CT b = static_cast<CT>(scale_b);
    CT temp = tanh(a * x) * tanh(a * x);
    return T(dout * a * b * (one - temp));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************STanh End********************/

/********************Softplus Begin********************/
template <typename T>
struct CudaSoftplusFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  float beta;
  float threshold;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    CT b = static_cast<CT>(beta);
    CT t = static_cast<CT>(threshold);
    CT x_beta = x * beta;
    return T(x_beta > t ? x : log(one + exp(x_beta)) / b);
  }
};

template <typename T>
struct CudaSoftplusGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  float beta;
  float threshold;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    CT b = static_cast<CT>(beta);
    CT t = static_cast<CT>(threshold);
    CT x_beta = x * beta;
    return x_beta > t ? args[0] : T(dout / (one + exp(-x_beta)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Softplus End********************/

/********************Softsign Begin********************/
template <typename T>
struct CudaSoftsignFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(x / (one + abs(x)));
  }
};

template <typename T>
struct CudaSoftsignGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout / ((one + abs(x)) * (one + abs(x))));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Softsign End********************/

/********************Relu6 Begin********************/
template <typename T>
struct CudaRelu6Functor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T t = static_cast<T>(threshold);
    return args[0] <= zero ? zero : (args[0] < t ? args[0] : t);
  }
};

template <typename T>
struct CudaRelu6GradFunctor : public BaseCudaActiveFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T t = static_cast<T>(threshold);
    return (args[1] > zero && args[1] < t) ? args[0] : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Relu6 End********************/

/********************TanhShrink Begin********************/
template <typename T>
struct CudaTanhShrinkFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(x - tanh(x));
  }
};

template <typename T>
struct CudaTanhShrinkGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return T(dout * tanh(x) * tanh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************TanhShrink End********************/

/********************HardShrink Begin********************/
template <typename T>
struct CudaHardShrinkFunctor : public BaseCudaActiveFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T t = static_cast<T>(threshold);
    T temp1 = static_cast<T>(x > t);
    T temp2 = static_cast<T>(x < -t);
    return x * (temp1 + temp2);
  }
};

template <typename T>
struct CudaHardShrinkGradFunctor : public BaseCudaActiveFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[1];
    T t = static_cast<T>(threshold);
    T temp1 = static_cast<T>(x > t);
    T temp2 = static_cast<T>(x < -t);
    return args[0] * (temp1 + temp2);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************HardShrink End********************/

template <typename DeviceContext, typename Functor>
class ActivationGPUKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* x = nullptr;
    framework::Tensor* out = nullptr;
    ExtractActivationTensor(ctx, &x, &out);
    out->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {out};
    auto functor = Functor();
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
    LaunchElementwiseCudaKernel<ElementwiseType::kUnary, T>(dev_ctx, ins, &outs,
                                                            functor);
  }
};

template <typename DeviceContext, typename Functor>
class ActivationGradGPUKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *x, *out, *d_out;
    framework::Tensor* d_x = nullptr;
    x = out = d_out = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(ctx, &x, &out, &d_out,
                                                    &d_x);
    d_x->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto functor = Functor();
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }

    std::vector<const framework::Tensor*> ins = {d_out};
    std::vector<framework::Tensor*> outs = {d_x};

    if (static_cast<int>(Functor::FwdDeps()) == static_cast<int>(kDepOut)) {
      // Only need forward output Out
      ins.push_back(out);
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T>(dev_ctx, ins,
                                                               &outs, functor);
    } else if (static_cast<int>(Functor::FwdDeps()) ==
               static_cast<int>(kDepX)) {
      // Only need forward input X
      ins.push_back(x);
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T>(dev_ctx, ins,
                                                               &outs, functor);
    } else {
      LaunchElementwiseCudaKernel<ElementwiseType::kUnary, T>(dev_ctx, ins,
                                                              &outs, functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, op_name, functor,         \
                                        grad_functor)                       \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type,                                                             \
      ops::ActivationKernel<plat::CUDADeviceContext, ops::functor<float>>,  \
      ops::ActivationKernel<plat::CUDADeviceContext, ops::functor<double>>, \
      ops::ActivationKernel<plat::CUDADeviceContext,                        \
                            ops::functor<plat::float16>>);                  \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type##_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,   \
                                                 ops::grad_functor<float>>, \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<double>>,                 \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<plat::float16>>);

#define REGISTER_ACTIVATION_GPU_KERNEL(act_type, op_name, functor,             \
                                       grad_functor)                           \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type, ops::ActivationGPUKernel<paddle::platform::CUDADeviceContext,  \
                                         ops::functor<float>>,                 \
      ops::ActivationGPUKernel<paddle::platform::CUDADeviceContext,            \
                               ops::functor<double>>,                          \
      ops::ActivationGPUKernel<plat::CUDADeviceContext,                        \
                               ops::functor<plat::float16>>);                  \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type##_grad, ops::ActivationGradGPUKernel<plat::CUDADeviceContext,   \
                                                    ops::grad_functor<float>>, \
      ops::ActivationGradGPUKernel<plat::CUDADeviceContext,                    \
                                   ops::grad_functor<double>>,                 \
      ops::ActivationGradGPUKernel<plat::CUDADeviceContext,                    \
                                   ops::grad_functor<plat::float16>>);

/* ======================== leaky relu register  ============================ */
REGISTER_ACTIVATION_GPU_KERNEL(leaky_relu, LeakyRelu, CudaLeakyReluFunctor,
                               CudaLeakyReluGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    leaky_relu_grad_grad,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::LeakyReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::LeakyReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<
        plat::CUDADeviceContext, ops::LeakyReluGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ======================== elu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(elu, ELU, ELUFunctor, ELUGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    elu_grad_grad, ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::ELUGradGradFunctor<float>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<double>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================    relu register  ============================ */
REGISTER_ACTIVATION_GPU_KERNEL(relu, Relu, CudaReluFunctor,
                               CudaReluGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    relu_grad_grad,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================    tanh register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(tanh, Tanh, TanhFunctor, TanhGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    tanh_grad_grad,
    ops::TanhDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::TanhGradGradFunctor<float>>,
    ops::TanhDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::TanhGradGradFunctor<double>>,
    ops::TanhDoubleGradKernel<plat::CUDADeviceContext,
                              ops::TanhGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================   sqrt register  ============================= */
REGISTER_ACTIVATION_CUDA_KERNEL(sqrt, Sqrt, SqrtFunctor, SqrtGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    sqrt_grad_grad,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<float>>,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<double>>,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================   rsqrt register  =============================
 */
REGISTER_ACTIVATION_CUDA_KERNEL(rsqrt, Rsqrt, RsqrtFunctor, RsqrtGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    rsqrt_grad_grad,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<float>>,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<double>>,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================  square register  ============================ */
REGISTER_OP_CUDA_KERNEL(
    square,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<float>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<double>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<int>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<int64_t>>,
    ops::ActivationKernel<plat::CUDADeviceContext,
                          ops::SquareFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    square_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,
                                           ops::SquareGradFunctor<float>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<double>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<int>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<int64_t>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<plat::float16>>);

REGISTER_OP_CUDA_KERNEL(
    square_grad_grad,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<float>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<double>>,
    ops::SquareDoubleGradKernel<plat::CUDADeviceContext,
                                ops::SquareGradGradFunctor<plat::float16>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<int>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<int64_t>>);
/* ========================================================================== */

/* ==========================   pow register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    pow, ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<float>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<double>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<int>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<int64_t>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    pow_grad,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<float>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<double>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<int>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<int64_t>>,
    ops::PowGradKernel<plat::CUDADeviceContext,
                       ops::PowGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================   exp register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    exp, ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<float>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<double>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int64_t>>,
    ops::ActivationKernel<plat::CUDADeviceContext,
                          ops::ExpFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    exp_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,
                                        ops::ExpGradFunctor<float>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<double>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<int>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<int64_t>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================  Log register ==================================*/
REGISTER_ACTIVATION_CUDA_KERNEL(log, Log, LogFunctor, LogGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    log_grad_grad, ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::LogGradGradFunctor<float>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<double>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<plat::float16>>);
/* ========================================================================== */
REGISTER_ACTIVATION_GPU_KERNEL(sigmoid, Sigmoid, CudaSigmoidFunctor,
                               CudaSigmoidGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(logsigmoid, LogSigmoid, CudaLogSigmoidFunctor,
                               CudaLogSigmoidGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(atan, Atan, CudaAtanFunctor,
                               CudaAtanGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(softshrink, SoftShrink, CudaSoftShrinkFunctor,
                               CudaSoftShrinkGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(ceil, Ceil, CudaCeilFunctor,
                               CudaZeroGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(floor, Floor, CudaFloorFunctor,
                               CudaZeroGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(cos, Cos, CudaCosFunctor, CudaCosGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(tan, Tan, CudaTanFunctor, CudaTanGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(acos, Acos, CudaAcosFunctor,
                               CudaAcosGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(sin, Sin, CudaSinFunctor, CudaSinGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(asin, Asin, CudaAsinFunctor,
                               CudaAsinGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(sinh, Sinh, CudaSinhFunctor,
                               CudaSinhGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(cosh, Cosh, CudaCoshFunctor,
                               CudaCoshGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(round, Round, CudaRoundFunctor,
                               CudaZeroGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(reciprocal, Reciprocal, CudaReciprocalFunctor,
                               CudaReciprocalGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(log1p, Log1p, CudaLog1pFunctor,
                               CudaLog1pGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(log2, Log2, CudaLog2Functor,
                               CudaLog2GradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(log10, Log10, CudaLog10Functor,
                               CudaLog10GradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(brelu, BRelu, CudaBReluFunctor,
                               CudaBReluGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(soft_relu, SoftRelu, CudaSoftReluFunctor,
                               CudaSoftReluGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(stanh, STanh, CudaSTanhFunctor,
                               CudaSTanhGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(softplus, Softplus, CudaSoftplusFunctor,
                               CudaSoftplusGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(softsign, Softsign, CudaSoftsignFunctor,
                               CudaSoftsignGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(relu6, Relu6, CudaRelu6Functor,
                               CudaRelu6GradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(tanh_shrink, TanhShrink, CudaTanhShrinkFunctor,
                               CudaTanhShrinkGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(hard_shrink, HardShrink, CudaHardShrinkFunctor,
                               CudaHardShrinkGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(sigmoid, Sigmoid, SigmoidFunctor,
// SigmoidGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(logsigmoid, LogSigmoid, LogSigmoidFunctor,
// LogSigmoidGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(atan, Atan, AtanFunctor, AtanGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(softshrink, SoftShrink, SoftShrinkFunctor,
// SoftShrinkGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(ceil, Ceil, CeilFunctor, ZeroGradFunctor);
//  REGISTER_ACTIVATION_CUDA_KERNEL(floor, Floor, FloorFunctor,
//  ZeroGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(cos, Cos, CosFunctor, CosGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(tan, Tan, TanFunctor, TanGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(acos, Acos, AcosFunctor, AcosGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(sin, Sin, SinFunctor, SinGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(asin, Asin, AsinFunctor, AsinGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(sinh, Sinh, SinhFunctor, SinhGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(cosh, Cosh, CoshFunctor, CoshGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(round, Round, RoundFunctor, ZeroGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(reciprocal, Reciprocal, ReciprocalFunctor,
//                                ReciprocalGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(log1p, Log1p, Log1pFunctor,
// Log1pGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(log2, Log2, Log2Functor, Log2GradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(log10, Log10, Log10Functor,
// Log10GradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(brelu, BRelu, BReluFunctor,
// BReluGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(soft_relu, SoftRelu, SoftReluFunctor,
//                                SoftReluGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(stanh, STanh, STanhFunctor,
// STanhGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(softplus, Softplus, SoftplusFunctor,
//                                SoftplusGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(softsign, Softsign, SoftsignFunctor,
//                                SoftsignGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(relu6, Relu6, Relu6Functor,
// Relu6GradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(tanh_shrink, TanhShrink, TanhShrinkFunctor,
//                                TanhShrinkGradFunctor);
// REGISTER_ACTIVATION_CUDA_KERNEL(hard_shrink, HardShrink, HardShrinkFunctor,
//                                HardShrinkGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(hard_sigmoid, HardSigmoid, HardSigmoidFunctor,
                                HardSigmoidGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(swish, Swish, SwishFunctor, SwishGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(thresholded_relu, ThresholdedRelu,
                                ThresholdedReluFunctor,
                                ThresholdedReluGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(hard_swish, HardSwish, HardSwishFunctor,
                                HardSwishGradFunctor);
