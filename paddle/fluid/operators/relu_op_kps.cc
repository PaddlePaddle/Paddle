/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include <string>
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/xpu/xpu_header.h"

namespace paddle {
namespace operators {

template <typename T>
struct XPUReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // relu(x) = max(x, 0)
  T operator()(const T& x) const { return x > zero ? x : zero; }
};

template <typename T>
struct XPUReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // dx = dout * (out > 0)
  T operator()(const T& dout, const T& out) const {
    return out > zero ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

using paddle::framework::Tensor;

void ReluXPU2Compute(const framework::ExecutionContext& ctx,
                     const std::vector<const framework::Tensor*>& ins,
                     std::vector<framework::Tensor*>* outs, int axis);
void ReluGradXPU2Compute(const framework::ExecutionContext& ctx,
                         const std::vector<const framework::Tensor*>& ins,
                         std::vector<framework::Tensor*>* outs, int axis);

template <typename Functor>
class ActivationCudaKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* x = nullptr;
    framework::Tensor* out = nullptr;
    ExtractActivationTensor(ctx, &x, &out);
    out->mutable_data<T>(ctx.GetPlace());
    // auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {out};
    auto functor = Functor();
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
    ReluXPU2Compute(ctx, ins, &outs, 0);
  }
};

template <typename Functor>
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
    // auto& dev_ctx = ctx.template device_context<DeviceContext>();
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
      ReluGradXPU2Compute(ctx, ins, &outs, 2);
    } else if (static_cast<int>(Functor::FwdDeps()) ==
               static_cast<int>(kDepX)) {
      // Only need forward input X
      ins.push_back(x);
      ReluGradXPU2Compute(ctx, ins, &outs, 1);
    } else {
      ReluGradXPU2Compute(ctx, ins, &outs, 1);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_XPU_KERNEL(act_type, functor, grad_functor)   \
  REGISTER_OP_XPU_KERNEL(act_type,                                        \
                         ops::ActivationCudaKernel<ops::functor<float>>); \
  REGISTER_OP_XPU_KERNEL(                                                 \
      act_type##_grad,                                                    \
      ops::ActivationGradCudaKernel<ops::grad_functor<float>>);

REGISTER_ACTIVATION_XPU_KERNEL(relu, XPUReluFunctor, XPUReluGradFunctor)
#endif  // PADDLE_WITH_XPU
