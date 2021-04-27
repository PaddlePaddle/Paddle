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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct CudaReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // relu(x) = max(x, 0)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > zero ? args[0] : zero;
  }
};

template <typename T>
struct CudaReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // dx = dout * (out > 0)
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > zero ? args[0] : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaLeakyReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // leakyrelu(x) = x > 0 ? x : alpha * x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > zero ? args[0] : static_cast<T>(alpha) * args[0];
  }
};

template <typename T>
struct CudaLeakyReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // dx = dout * (x > 0 ? 1 : alpha)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > zero ? args[0] : static_cast<T>(alpha) * args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSigmoidFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // sigmoid(x) = 1 / (1 + exp(-x))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(one / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSigmoidGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * out * (1 - out)
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1] * (one - args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaSiluFunctor : public BaseActivationFunctor<T> {
  // MPType means Compute Type
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // silu(x) = x / (1 + exp(-x))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(x / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSiluGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout * (1 + exp(-x) + x * exp(-x) / (1 + exp(-x))^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType temp = one / (one + exp(-x));
    return static_cast<T>(dout * (temp * (one + x * (one - temp))));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaLogSigmoidFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);

  // logsigmoid(x) = log(1 / (1 + exp(-x)))
  // For numerical stability,
  // logsigmoid(x) =
  //          - (max(-x, 0) + log(exp(-max(-x, 0)) + exp(-x - max(-x, 0))))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    MPType temp = x > zero ? zero : -x;
    return static_cast<T>(-temp - log(exp(-temp) + exp(-x - temp)));
  }
};

template <typename T>
struct CudaLogSigmoidGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);

  // dx = dout * exp(-x) / (1 + exp(-x))
  // For numerical stability:
  // dx = dout * exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) + exp(-x - max(-x,
  // 0)))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType temp1 = x > zero ? zero : -x;
    MPType temp2 = exp(-x - temp1);
    return static_cast<T>(dout * (temp2 / (exp(-temp1) + temp2)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaAtanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // atan(x) = atan(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(atan(x));
  }
};

template <typename T>
struct CudaAtanGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + x^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (one + args[1] * args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSoftShrinkFunctor : public BaseActivationFunctor<T> {
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  // softshrink(x) = x - lambda, if x > lambda;
  //                 x + lambda, if x < -lambda;
  //                 0, otherwise.
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T l = static_cast<T>(lambda);
    T temp1 = static_cast<T>(x > l);
    T temp2 = static_cast<T>(x < -l);
    return temp1 * (x - l) + temp2 * (x + l);
  }
};

template <typename T>
struct CudaSoftShrinkGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  // dx = dout, if x > lambda or x < -lambda else 0
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[1];
    T l = static_cast<T>(lambda);
    return (x >= -l && x <= l) ? zero : args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaCeilFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // ceil(x) = ceil(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(ceil(x));
  }
};

template <typename T>
struct CudaFloorFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // floor(x) = floor(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(floor(x));
  }
};

template <typename T>
struct CudaRoundFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // round(x) = round(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(round(x));
  }
};

// grad functor for ceil, floor and round
template <typename T>
struct CudaZeroGradFunctor : public BaseActivationFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return static_cast<T>(0.0f);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kNoDeps; }
};

template <typename T>
struct CudaCosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // cos(x) = cos(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(cos(x));
  }
};

template <typename T>
struct CudaCosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * (-sin(x))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(-dout * sin(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // sin(x) = sin(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(sin(x));
  }
};

template <typename T>
struct CudaSinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * cos(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * cos(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaTanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // tan(x) = tan(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(tan(x));
  }
};

template <typename T>
struct CudaTanGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout / cos(x)^2
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout / (cos(x) * cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaAsinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // asin(x) = asin(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(asin(x));
  }
};

template <typename T>
struct CudaAsinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout / sqrt(1 - x^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaAcosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // acos(x) = acos(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(acos(x));
  }
};

template <typename T>
struct CudaAcosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = -dout / sqrt(1 - x^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(-dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaCoshFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // cosh(x) = cosh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(cosh(x));
  }
};

template <typename T>
struct CudaCoshGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * sinh(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * sinh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSinhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // sinh(x) = sinh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(sinh(x));
  }
};

template <typename T>
struct CudaSinhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * cosh(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * cosh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaTanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // tanh(x) = tanh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(tanh(x));
  }
};

template <typename T>
struct CudaTanhGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * (1 - out^2)
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    T dout = static_cast<T>(args[0]);
    T out = static_cast<T>(args[1]);
    return dout * (one - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaReciprocalFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // reciprocal(x) = 1 / x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return one / args[0];
  }
};

template <typename T>
struct CudaReciprocalGradFunctor : public BaseActivationFunctor<T> {
  // dx = -dout * out^2
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return -args[0] * args[1] * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaExpFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // exp(x) = exp(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(exp(x));
  }
};

template <typename T>
struct CudaExpGradFunctor : public BaseActivationFunctor<T> {
  // dx = dout * out
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaLogFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // log(x) = log(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(log(x));
  }
};

template <typename T>
struct CudaLogGradFunctor : public BaseActivationFunctor<T> {
  // dx = dout / x
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSquareFunctor : public BaseActivationFunctor<T> {
  // square(x) = x * x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[0];
  }
};

template <typename T>
struct CudaSquareGradFunctor : public BaseActivationFunctor<T> {
  T two = static_cast<T>(2.0f);

  // dx = dout * 2 * x
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * two * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSqrtFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // sqrt(x) = sqrt(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(sqrt(x));
  }
};

template <typename T>
struct CudaSqrtGradFunctor : public BaseActivationFunctor<T> {
  T one_half = static_cast<T>(0.5f);

  // dx = dout * 0.5 / out
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return one_half * args[0] / args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaRsqrtFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // rsqrt(x) = rsqrt(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(rsqrt(x));
  }
};

template <typename T>
struct CudaRsqrtGradFunctor : public BaseActivationFunctor<T> {
  T minus_one_half = static_cast<T>(-0.5f);

  // dx = dout * -0.5 / out^3
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    T out = args[1];
    return minus_one_half * args[0] * out * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

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
REGISTER_ACTIVATION_GPU_KERNEL(elu, ELU, ELUFunctor, ELUGradFunctor);

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
