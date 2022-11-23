// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

using phi::GPUContext;
using platform::ActivationDescriptor;
using platform::TensorDescriptor;

#ifdef PADDLE_WITH_HIP
#define GPUDNN_ACTIVATION_RELU miopenActivationRELU
#define GPUDNN_ACTIVATION_CLIPPED_RELU miopenActivationCLIPPEDRELU
#define GPUDNN_ACTIVATION_SIGMOID miopenActivationLOGISTIC
#define GPUDNN_ACTIVATION_TANH miopenActivationTANH
#else
#define GPUDNN_ACTIVATION_RELU CUDNN_ACTIVATION_RELU
#define GPUDNN_ACTIVATION_CLIPPED_RELU CUDNN_ACTIVATION_CLIPPED_RELU
#define GPUDNN_ACTIVATION_SIGMOID CUDNN_ACTIVATION_SIGMOID
#define GPUDNN_ACTIVATION_TANH CUDNN_ACTIVATION_TANH
#endif

template <typename T>
struct CudnnActivationFunctor {
  using ELEMENT_TYPE = T;
#ifdef PADDLE_WITH_HIP
  CudnnActivationFunctor(const phi::GPUContext& ctx,
                         const T& c,
                         const miopenActivationMode_t& m)
      : ctx_(ctx), coef_(c), mode_(m) {}
#else
  CudnnActivationFunctor(const phi::GPUContext& ctx,
                         const T& c,
                         const cudnnActivationMode_t& m)
      : ctx_(ctx), coef_(c), mode_(m) {}
#endif
  void operator()(const phi::DenseTensor& x, phi::DenseTensor* out) {
    ActivationDescriptor act_desc;
    act_desc.set(mode_, coef_);
    TensorDescriptor x_desc, out_desc;
    x_desc.set(x);
    out_desc.set(GET_DATA_SAFELY(out, "Output", "Out", "CudnnActivation"));
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenActivationForward(
        ctx_.cudnn_handle(),
        act_desc.desc(),
        platform::CudnnDataType<T>::kOne(),
        x_desc.desc(),
        x.data<T>(),
        platform::CudnnDataType<T>::kZero(),
        out_desc.desc(),
        out->mutable_data<T>(ctx_.GetPlace())));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnActivationForward(
        ctx_.cudnn_handle(),
        act_desc.desc(),
        platform::CudnnDataType<T>::kOne(),
        x_desc.desc(),
        x.data<T>(),
        platform::CudnnDataType<T>::kZero(),
        out_desc.desc(),
        out->mutable_data<T>(ctx_.GetPlace())));
#endif
  }
  const phi::GPUContext& ctx_;
  const T coef_;
#ifdef PADDLE_WITH_HIP
  const miopenActivationMode_t mode_;
#else
  const cudnnActivationMode_t mode_;
#endif
};

template <typename T>
struct CudnnActivationGradFunctor {
  using ELEMENT_TYPE = T;
#ifdef PADDLE_WITH_HIP
  CudnnActivationGradFunctor(const phi::GPUContext& ctx,
                             const T& c,
                             const miopenActivationMode_t& m)
      : ctx_(ctx), coef_(c), mode_(m) {}
#else
  CudnnActivationGradFunctor(const phi::GPUContext& ctx,
                             const T& c,
                             const cudnnActivationMode_t& m)
      : ctx_(ctx), coef_(c), mode_(m) {}
#endif
  void operator()(const phi::DenseTensor& x,
                  const phi::DenseTensor& out,
                  const phi::DenseTensor dout,
                  phi::DenseTensor* dx) {
    ActivationDescriptor act_desc;
    act_desc.set(mode_, coef_);
    TensorDescriptor x_desc, out_desc, dout_desc, dx_desc;
    x_desc.set(x);
    out_desc.set(out);
    dout_desc.set(dout);
    dx_desc.set(GET_DATA_SAFELY(dx, "Output", "X@GRAD", "CudnnActivationGrad"));
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenActivationBackward(
        ctx_.cudnn_handle(),
        act_desc.desc(),
        platform::CudnnDataType<T>::kOne(),
        out_desc.desc(),
        out.data<T>(),
        dout_desc.desc(),
        dout.data<T>(),
        x_desc.desc(),
        x.data<T>(),
        platform::CudnnDataType<T>::kZero(),
        dx_desc.desc(),
        dx->mutable_data<T>(ctx_.GetPlace())));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnActivationBackward(
        ctx_.cudnn_handle(),
        act_desc.desc(),
        platform::CudnnDataType<T>::kOne(),
        out_desc.desc(),
        out.data<T>(),
        dout_desc.desc(),
        dout.data<T>(),
        x_desc.desc(),
        x.data<T>(),
        platform::CudnnDataType<T>::kZero(),
        dx_desc.desc(),
        dx->mutable_data<T>(ctx_.GetPlace())));
#endif
  }
  const phi::GPUContext& ctx_;
  const T coef_;
#ifdef PADDLE_WITH_HIP
  const miopenActivationMode_t mode_;
#else
  const cudnnActivationMode_t mode_;
#endif
};

template <typename T>
struct CudnnReluFunctor : public CudnnActivationFunctor<T> {
  explicit CudnnReluFunctor(const phi::GPUContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 0.0, GPUDNN_ACTIVATION_RELU) {}
};
template <typename T>
struct CudnnReluGradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnReluGradFunctor(const phi::GPUContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 0.0, GPUDNN_ACTIVATION_RELU) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudnnRelu6Functor : public CudnnActivationFunctor<T> {
  explicit CudnnRelu6Functor(const phi::GPUContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 6.0, GPUDNN_ACTIVATION_CLIPPED_RELU) {}
};
template <typename T>
struct CudnnRelu6GradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnRelu6GradFunctor(const phi::GPUContext& ctx)
      : CudnnActivationGradFunctor<T>(
            ctx, 6.0, GPUDNN_ACTIVATION_CLIPPED_RELU) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudnnSigmoidFunctor : public CudnnActivationFunctor<T> {
  explicit CudnnSigmoidFunctor(const phi::GPUContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 0.0, GPUDNN_ACTIVATION_SIGMOID) {}
};
template <typename T>
struct CudnnSigmoidGradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnSigmoidGradFunctor(const phi::GPUContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 0.0, GPUDNN_ACTIVATION_SIGMOID) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudnnTanhFunctor : public CudnnActivationFunctor<T> {
  explicit CudnnTanhFunctor(const phi::GPUContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 0.0, GPUDNN_ACTIVATION_TANH) {}
};
template <typename T>
struct CudnnTanhGradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnTanhGradFunctor(const phi::GPUContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 0.0, GPUDNN_ACTIVATION_TANH) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename Functor>
class CudnnActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* X = nullptr;
    phi::DenseTensor* Out = nullptr;
    ExtractActivationTensor(context, &X, &Out);
    Out->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<phi::GPUContext>();
    Functor functor(dev_ctx);
    functor(GET_DATA_SAFELY(X, "Input", "X", "CudnnActivation"), Out);
  }
};

template <typename Functor>
class CudnnActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    static_assert(Functor::FwdDeps() == ActBwdOpFwdDeps::kDepOut,
                  "Forward deps must be Out.");

    const phi::DenseTensor *X, *Out, *dOut;
    X = Out = dOut = nullptr;
    phi::DenseTensor* dX = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(
        context, &X, &Out, &dOut, &dX);
    dX->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<phi::GPUContext>();
    Functor functor(dev_ctx);
    functor(GET_DATA_SAFELY(X, "Input", "X", "CudnnActivationGrad"),
            GET_DATA_SAFELY(Out, "Input", "Out", "CudnnActivationGrad"),
            GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "CudnnActivationGrad"),
            dX);
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

#define FOR_EACH_CUDNN_OP_FUNCTOR(__macro)                        \
  __macro(relu, CudnnReluFunctor, CudnnReluGradFunctor);          \
  __macro(relu6, CudnnRelu6Functor, CudnnRelu6GradFunctor);       \
  __macro(sigmoid, CudnnSigmoidFunctor, CudnnSigmoidGradFunctor); \
  __macro(tanh, CudnnTanhFunctor, CudnnTanhGradFunctor)

#ifdef PADDLE_WITH_HIP
#define REGISTER_ACTIVATION_CUDNN_KERNEL(act_type, functor, grad_functor) \
  REGISTER_OP_KERNEL(act_type,                                            \
                     CUDNN,                                               \
                     plat::CUDAPlace,                                     \
                     ops::CudnnActivationKernel<ops::functor<float>>);    \
  REGISTER_OP_KERNEL(                                                     \
      act_type##_grad,                                                    \
      CUDNN,                                                              \
      plat::CUDAPlace,                                                    \
      ops::CudnnActivationGradKernel<ops::grad_functor<float>>);
#else
#define REGISTER_ACTIVATION_CUDNN_KERNEL(act_type, functor, grad_functor) \
  REGISTER_OP_KERNEL(act_type,                                            \
                     CUDNN,                                               \
                     plat::CUDAPlace,                                     \
                     ops::CudnnActivationKernel<ops::functor<float>>,     \
                     ops::CudnnActivationKernel<ops::functor<double>>);   \
  REGISTER_OP_KERNEL(                                                     \
      act_type##_grad,                                                    \
      CUDNN,                                                              \
      plat::CUDAPlace,                                                    \
      ops::CudnnActivationGradKernel<ops::grad_functor<float>>,           \
      ops::CudnnActivationGradKernel<ops::grad_functor<double>>);
#endif

FOR_EACH_CUDNN_OP_FUNCTOR(REGISTER_ACTIVATION_CUDNN_KERNEL);
