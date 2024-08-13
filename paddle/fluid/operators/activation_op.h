/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <type_traits>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace paddle {
namespace operators {

using phi::To32BitIndex;

using ActBwdOpFwdDeps = phi::funcs::ActBwdOpFwdDeps;

/* The following operator can be used to process SelectedRows, because the
 * output of those operator for zero is zero too.
 */
static std::unordered_set<std::string> CanBeUsedBySelectedRows = {
    "abs", "abs_grad", "square", "square_grad", "sqrt", "sqrt_grad"};

inline void ExtractActivationTensor(const framework::ExecutionContext& context,
                                    const phi::DenseTensor** X,
                                    phi::DenseTensor** Out) {
  auto x_var = context.InputVar("X");
  auto out_var = context.OutputVar("Out");
  PADDLE_ENFORCE_NOT_NULL(x_var,
                          common::errors::NotFound(
                              "Cannot get input Variable X, variable name = %s",
                              context.InputName("X")));
  PADDLE_ENFORCE_NOT_NULL(
      out_var,
      common::errors::NotFound(
          "Cannot get output Variable Out, variable name = %s",
          context.OutputName("Out")));
  if (CanBeUsedBySelectedRows.count(context.Type())) {
    *X = paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_var);
    *Out = paddle::framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(
        out_var);
  } else {
    *X = context.Input<phi::DenseTensor>("X");
    *Out = context.Output<phi::DenseTensor>("Out");
  }

  PADDLE_ENFORCE_NOT_NULL(
      *Out,
      common::errors::NotFound("Cannot get the tensor from the Variable "
                               "Output(Out), variable name = %s",
                               context.OutputName("Out")));
}

template <ActBwdOpFwdDeps kDepValue>
inline void ExtractActivationGradTensor(
    const framework::ExecutionContext& context,
    const phi::DenseTensor** X,
    const phi::DenseTensor** Out,
    const phi::DenseTensor** dOut,
    phi::DenseTensor** dX) {
  auto out_grad_var = context.InputVar(framework::GradVarName("Out"));
  auto x_grad_var = context.OutputVar(framework::GradVarName("X"));
  const framework::Variable* out_var = nullptr;

  if (static_cast<int>(kDepValue) &
      static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
    out_var = context.InputVar("Out");
    PADDLE_ENFORCE_NOT_NULL(
        out_var,
        common::errors::NotFound(
            "Cannot get input Variable Out, variable name = %s",
            context.InputName("Out")));
  }

  PADDLE_ENFORCE_NOT_NULL(
      out_grad_var,
      common::errors::NotFound(
          "Cannot get input Variable %s, variable name = %s",
          framework::GradVarName("Out"),
          context.InputName(framework::GradVarName("Out"))));
  PADDLE_ENFORCE_NOT_NULL(
      x_grad_var,
      common::errors::NotFound(
          "Cannot get output Variable %s, variable name = %s",
          framework::GradVarName("X"),
          context.OutputName(framework::GradVarName("X"))));

  if (CanBeUsedBySelectedRows.count(context.Type())) {
    *dOut = paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(
        *out_grad_var);
    *dX = paddle::framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(
        x_grad_var);

    if (out_var) {
      *Out =
          paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(*out_var);
    } else {
      *Out = *dOut;  // fake out
    }

  } else {
    *Out = context.Input<phi::DenseTensor>("Out");
    *dOut = context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    *dX = context.Output<phi::DenseTensor>(framework::GradVarName("X"));

    if (out_var) {
      *Out = &(out_var->Get<phi::DenseTensor>());
    } else {
      *Out = *dOut;  // fake out
    }
  }

  PADDLE_ENFORCE_NOT_NULL(*dX,
                          common::errors::NotFound(
                              "Cannot get the tensor from the Variable "
                              "Output(Out), variable name = %s",
                              context.OutputName(framework::GradVarName("X"))));

  if (static_cast<int>(kDepValue) & static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
    auto x_var = context.InputVar("X");
    PADDLE_ENFORCE_NOT_NULL(
        x_var,
        common::errors::NotFound("Cannot get the tensor from the "
                                 "Variable Input(X), variable name = %s",
                                 context.InputName("X")));
    if (CanBeUsedBySelectedRows.count(context.Type())) {
      *X = paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_var);
    } else {
      *X = context.Input<phi::DenseTensor>("X");
    }
  } else {
    VLOG(10) << " Inplace activation of Op : " << context.Type();
    *X = *dX;
  }
}

template <typename DeviceContext, typename Functor>
class ActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;

  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* X = nullptr;
    phi::DenseTensor* Out = nullptr;
    ExtractActivationTensor(context, &X, &Out);
    Out->mutable_data<T>(context.GetPlace());

    auto x = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "Activation"));
    auto out = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "Activation"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    // use 32bit index to speed up computation
    bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
    bool is_gpu_place =
        context.GetPlace().GetType() == phi::AllocationType::GPU;
    if (use_32bit_index && is_gpu_place) {
      functor(*place, To32BitIndex(x), To32BitIndex(out));
    } else {
      functor(*place, x, out);
    }
  }
};

template <typename DeviceContext, typename Functor>
class ActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor *X, *Out, *dOut;
    phi::DenseTensor* dX = nullptr;
    X = Out = dOut = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(
        context, &X, &Out, &dOut, &dX);
    dX->mutable_data<T>(context.GetPlace());
    auto dout = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "ActivationGrad"));
    auto out = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "ActivationGrad"));
    auto dx = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Input", "X@GRAD", "ActivationGrad"));
    auto x = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "ActivationGrad"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    // use 32bit index to speed up computation
    bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
    bool is_gpu_place =
        context.GetPlace().GetType() == phi::AllocationType::GPU;
    if (use_32bit_index && is_gpu_place) {
      functor(*place,
              To32BitIndex(x),
              To32BitIndex(out),
              To32BitIndex(dout),
              To32BitIndex(dx));
    } else {
      functor(*place, x, out, dout, dx);
    }
  }
};

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char*, float*>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

#define USE_PHI_FUNCTOR(name)                         \
  template <typename T>                               \
  using name##Functor = phi::funcs::name##Functor<T>; \
  template <typename T>                               \
  using name##GradFunctor = phi::funcs::name##GradFunctor<T>;

#define USE_PHI_DOUBLE_GRAD_FUNCTOR(name) \
  template <typename T>                   \
  using name##GradGradFunctor = phi::funcs::name##GradGradFunctor<T>;

#define USE_PHI_TRIPLE_GRAD_FUNCTOR(name) \
  template <typename T>                   \
  using name##TripleGradFunctor = phi::funcs::name##TripleGradFunctor<T>;

template <typename T>
using BReluFunctor = phi::funcs::HardTanhFunctor<T>;
template <typename T>
using BReluGradFunctor = phi::funcs::HardTanhGradFunctor<T>;

USE_PHI_FUNCTOR(Tanh)
USE_PHI_FUNCTOR(Relu6)
USE_PHI_FUNCTOR(HardShrink)
USE_PHI_FUNCTOR(ELU)
USE_PHI_FUNCTOR(Sigmoid)
USE_PHI_FUNCTOR(HardSigmoid)
USE_PHI_FUNCTOR(Swish)
USE_PHI_FUNCTOR(HardSwish)
USE_PHI_FUNCTOR(Pow)
USE_PHI_FUNCTOR(Mish)

template <typename T>
using ELUGradNegativeAlphaFunctor = phi::funcs::ELUGradNegativeAlphaFunctor<T>;

template <typename T>
using RoundFunctor = phi::funcs::RoundFunctor<T>;

template <typename T>
using FloorFunctor = phi::funcs::FloorFunctor<T>;

template <typename T>
using CeilFunctor = phi::funcs::CeilFunctor<T>;

template <typename T>
using ZeroGradFunctor = phi::funcs::ZeroGradFunctor<T>;

template <typename T>
using ELUGradNegativeAlphaFunctor = phi::funcs::ELUGradNegativeAlphaFunctor<T>;

// relu(x) = max(x, 0)

template <typename T>
using ReluCPUFunctor = phi::funcs::ReluCPUFunctor<T>;
template <typename T>
using ReluGradFunctor = phi::funcs::ReluGradFunctor<T>;

template <typename T>
using ReluGradGradFunctor = phi::funcs::ReluGradGradFunctor<T>;

template <typename T>
using ReluCUDAFunctor = phi::funcs::ReluCUDAFunctor<T>;

template <typename T>
struct SoftReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto tmp = static_cast<T>(threshold);
    auto temp = x.cwiseMax(-tmp).cwiseMin(tmp);
    out.device(d) = (static_cast<T>(1) + temp.exp()).log();
  }
};

template <typename T>
struct SoftReluGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    auto tmp = static_cast<T>(threshold);
    auto temp = ((out > -tmp) * (out < tmp)).template cast<T>();
    dx.device(d) = dout * (static_cast<T>(1) - (-out).exp()) * temp;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct AbsGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const phi::DenseTensor* X,
                  const phi::DenseTensor* Out,
                  const phi::DenseTensor* ddX,
                  phi::DenseTensor* ddOut,
                  phi::DenseTensor* dOut,
                  phi::DenseTensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "AbsGradGrad"));
    auto x = phi::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "AbsGradGrad"));
    if (ddOut) {
      auto ddout = phi::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "AbsGradGrad"));
      ddout.device(*d) = ddx * x.sign();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// TODO(dengkaipeng): double gradient calculation for Square/Sqrt need
// DOut(dy) as input(not output), tensor extraction is different from
// others. Implement extraction kernel separately here.
inline void ExtractDoubleGradTensorWithInputDOut(
    const framework::ExecutionContext& ctx,
    const phi::DenseTensor** X,
    const phi::DenseTensor** ddX,
    phi::DenseTensor** dX,
    const phi::DenseTensor** dOut,
    phi::DenseTensor** ddOut) {
  // extract ddX(output), ddOut(input)
  auto ddx_var = ctx.InputVar("DDX");
  auto ddo_var = ctx.OutputVar("DDOut");
  PADDLE_ENFORCE_NOT_NULL(
      ddx_var,
      common::errors::NotFound(
          "Cannot get input Variable Out, variable name = %s",
          ctx.InputName("DDX")));
  *ddX = ctx.Input<phi::DenseTensor>("DDX");
  if (ddo_var) {
    *ddOut = ctx.Output<phi::DenseTensor>("DDOut");
  }
  PADDLE_ENFORCE_NOT_NULL(
      ddX,
      common::errors::NotFound(
          "Cannot get the tensor from the Variable DDX, variable name = %s",
          ctx.OutputName("DDX")));

  // extract x(input), dx(output)
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NOT_NULL(
      x_var,
      common::errors::NotFound(
          "Cannot get input Variable Out, variable name = %s",
          ctx.InputName("X")));
  auto dx_var = ctx.OutputVar("DX");
  *X = ctx.Input<phi::DenseTensor>("X");
  if (dx_var) {
    *dX = ctx.Output<phi::DenseTensor>("DX");
  }

  // extract dOut(input)
  auto dout_var = ctx.InputVar("DOut");
  if (dout_var) {
    *dOut = ctx.Input<phi::DenseTensor>("DOut");
  }
}

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_ACTIVATION_OP(__macro) \
  __macro(soft_relu, SoftRelu, SoftReluFunctor, SoftReluGradFunctor);
