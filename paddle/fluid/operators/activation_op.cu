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

/********************Silu Begin********************/
template <typename T>
struct CudaSiluFunctor : public BaseCudaActiveFunctor<T> {
  // CT means Compute Type
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(x / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSiluGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT one = static_cast<CT>(1.0f);

  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    CT temp1 = one + exp(-x);
    CT temp2 = x * exp(-x);
    return T(dout * ((one / temp1) * (one + temp2 / temp1)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Silu End********************/

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

/********************Tanh Begin********************/
template <typename T>
struct CudaTanhFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(tanh(x));
  }
};

template <typename T>
struct CudaTanhGradFunctor : public BaseCudaActiveFunctor<T> {
  T one = static_cast<T>(1.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    T dout = static_cast<T>(args[0]);
    T out = static_cast<T>(args[1]);
    return dout * (one - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Tanh End********************/

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

/********************Exp Begin********************/
template <typename T>
struct CudaExpFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(exp(x));
  }
};

template <typename T>
struct CudaExpGradFunctor : public BaseCudaActiveFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Exp End********************/

/********************Log Begin********************/
template <typename T>
struct CudaLogFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(log(x));
  }
};

template <typename T>
struct CudaLogGradFunctor : public BaseCudaActiveFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Log End********************/

/********************ELU Begin********************/
template <typename T>
struct CudaELUFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);
  CT one = static_cast<CT>(1.0f);
  float alpha;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return x >= zero ? args[0] : T(static_cast<CT>(alpha) * (exp(x) - one));
  }
};

template <typename T>
struct CudaELUGradFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);
  CT one = static_cast<CT>(1.0f);
  float alpha;

  typename BaseCudaActiveFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  __device__ __forceinline__ T operator()(const T* args) const {
    CT dout = static_cast<CT>(args[0]);
    CT x = static_cast<CT>(args[1]);
    return x >= zero ? args[0] : T(dout * static_cast<CT>(alpha) * exp(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************ELU End********************/

/********************Square Begin********************/
template <typename T>
struct CudaSquareFunctor : public BaseCudaActiveFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[0];
  }
};

template <typename T>
struct CudaSquareGradFunctor : public BaseCudaActiveFunctor<T> {
  T two = static_cast<T>(2.0f);
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * two * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
/********************Square End********************/

/********************Sqrt Begin********************/
template <typename T>
struct CudaSqrtFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(sqrt(x));
  }
};

template <typename T>
struct CudaSqrtGradFunctor : public BaseCudaActiveFunctor<T> {
  T one_half = static_cast<T>(0.5f);
  __device__ __forceinline__ T operator()(const T* args) const {
    return one_half * args[0] / args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Sqrt End********************/

/********************Rsqrt Begin********************/
template <typename T>
struct CudaRsqrtFunctor : public BaseCudaActiveFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    return T(rsqrt(x));
  }
};

template <typename T>
struct CudaRsqrtGradFunctor : public BaseCudaActiveFunctor<T> {
  T minus_one_half = static_cast<T>(-0.5f);
  __device__ __forceinline__ T operator()(const T* args) const {
    T out = args[1];
    return minus_one_half * args[0] * out * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};
/********************Rsqrt End********************/

template <typename DeviceContext, typename Functor>
class ActivationCudaKernel
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
class ActivationGradCudaKernel
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

#define REGISTER_ACTIVATION_GPU_KERNEL(act_type, op_name, functor,          \
                                       grad_functor)                        \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type, ops::ActivationKernel<paddle::platform::CUDADeviceContext,  \
                                      ops::functor<float>>,                 \
      ops::ActivationKernel<paddle::platform::CUDADeviceContext,            \
                            ops::functor<double>>,                          \
      ops::ActivationKernel<plat::CUDADeviceContext,                        \
                            ops::functor<plat::float16>>);                  \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type##_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,   \
                                                 ops::grad_functor<float>>, \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<double>>,                 \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<plat::float16>>);

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, op_name, functor,            \
                                        grad_functor)                          \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type, ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext, \
                                          ops::functor<float>>,                \
      ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,           \
                                ops::functor<double>>,                         \
      ops::ActivationCudaKernel<plat::CUDADeviceContext,                       \
                                ops::functor<plat::float16>>);                 \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type##_grad,                                                         \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<float>>,                 \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<double>>,                \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<plat::float16>>);

/* ======================== leaky relu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(leaky_relu, LeakyRelu, CudaLeakyReluFunctor,
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
REGISTER_ACTIVATION_CUDA_KERNEL(elu, ELU, CudaELUFunctor, CudaELUGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    elu_grad_grad, ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::ELUGradGradFunctor<float>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<double>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================    relu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(relu, Relu, CudaReluFunctor,
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
REGISTER_ACTIVATION_CUDA_KERNEL(tanh, Tanh, CudaTanhFunctor,
                                CudaTanhGradFunctor);

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
REGISTER_ACTIVATION_CUDA_KERNEL(sqrt, Sqrt, CudaSqrtFunctor,
                                CudaSqrtGradFunctor);

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
REGISTER_ACTIVATION_CUDA_KERNEL(rsqrt, Rsqrt, CudaRsqrtFunctor,
                                CudaRsqrtGradFunctor);

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
    square, ops::ActivationCudaKernel<plat::CUDADeviceContext,
                                      ops::CudaSquareFunctor<float>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaSquareFunctor<double>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaSquareFunctor<int>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaSquareFunctor<int64_t>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaSquareFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    square_grad,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaSquareGradFunctor<float>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaSquareGradFunctor<double>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaSquareGradFunctor<int>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaSquareGradFunctor<int64_t>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaSquareGradFunctor<plat::float16>>);

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
    exp, ops::ActivationCudaKernel<plat::CUDADeviceContext,
                                   ops::CudaExpFunctor<float>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaExpFunctor<double>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int64_t>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaExpFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    exp_grad, ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                            ops::CudaExpGradFunctor<float>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<double>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<int>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<int64_t>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================  Log register ==================================*/
REGISTER_ACTIVATION_CUDA_KERNEL(log, Log, CudaLogFunctor, CudaLogGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    log_grad_grad, ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::LogGradGradFunctor<float>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<double>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<plat::float16>>);
/* ========================================================================== */

REGISTER_ACTIVATION_CUDA_KERNEL(sigmoid, Sigmoid, CudaSigmoidFunctor,
                                CudaSigmoidGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(silu, Silu, CudaSiluFunctor,
                                CudaSiluGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(logsigmoid, LogSigmoid, CudaLogSigmoidFunctor,
                                CudaLogSigmoidGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(atan, Atan, CudaAtanFunctor,
                                CudaAtanGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(softshrink, SoftShrink, CudaSoftShrinkFunctor,
                                CudaSoftShrinkGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(ceil, Ceil, CudaCeilFunctor,
                                CudaZeroGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(floor, Floor, CudaFloorFunctor,
                                CudaZeroGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(cos, Cos, CudaCosFunctor, CudaCosGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(tan, Tan, CudaTanFunctor, CudaTanGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(acos, Acos, CudaAcosFunctor,
                                CudaAcosGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(sin, Sin, CudaSinFunctor, CudaSinGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(asin, Asin, CudaAsinFunctor,
                                CudaAsinGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(sinh, Sinh, CudaSinhFunctor,
                                CudaSinhGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(cosh, Cosh, CudaCoshFunctor,
                                CudaCoshGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(round, Round, CudaRoundFunctor,
                                CudaZeroGradFunctor);
REGISTER_ACTIVATION_CUDA_KERNEL(reciprocal, Reciprocal, CudaReciprocalFunctor,
                                CudaReciprocalGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(log1p, Log1p, Log1pFunctor, Log1pGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(log2, Log2, Log2Functor, Log2GradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(log10, Log10, Log10Functor, Log10GradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(brelu, BRelu, BReluFunctor, BReluGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(soft_relu, SoftRelu, SoftReluFunctor,
                               SoftReluGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(stanh, STanh, STanhFunctor, STanhGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(softplus, Softplus, SoftplusFunctor,
                               SoftplusGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(softsign, Softsign, SoftsignFunctor,
                               SoftsignGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(relu6, Relu6, Relu6Functor, Relu6GradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(tanh_shrink, TanhShrink, TanhShrinkFunctor,
                               TanhShrinkGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(hard_shrink, HardShrink, HardShrinkFunctor,
                               HardShrinkGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(hard_sigmoid, HardSigmoid, HardSigmoidFunctor,
                               HardSigmoidGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(swish, Swish, SwishFunctor, SwishGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(thresholded_relu, ThresholdedRelu,
                               ThresholdedReluFunctor,
                               ThresholdedReluGradFunctor);
REGISTER_ACTIVATION_GPU_KERNEL(hard_swish, HardSwish, HardSwishFunctor,
                               HardSwishGradFunctor);
