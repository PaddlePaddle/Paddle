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
#include <string>
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {

enum ArgMinMaxV2Type { kArgMinV2, kArgMaxV2 };

template <typename DeviceContext, typename T, typename Tout, int64_t Rank,
          ArgMinMaxV2Type argMinMaxValue>
struct ArgMinMaxV2Functor {};

#define DECLARE_ARG_MIN_MAX_FUNCTOR(eigen_op_type, enum_argminmax_value)      \
  template <typename DeviceContext, typename T, typename Tout, int64_t Rank>  \
  struct ArgMinMaxV2Functor<DeviceContext, T, Tout, Rank,                     \
                            enum_argminmax_value> {                           \
    void operator()(const DeviceContext& ctx, const framework::LoDTensor& in, \
                    framework::LoDTensor* out, framework::DDim x_dims,        \
                    int64_t axis, bool keepdims) {                            \
      auto in_eigen = framework::EigenTensor<T, Rank>::From(in, x_dims);      \
      if (keepdims) {                                                         \
        auto out_eigen = framework::EigenTensor<Tout, Rank>::From(*out);      \
        out_eigen.device(*(ctx.eigen_device())) =                             \
            in_eigen.eigen_op_type(axis).template cast<Tout>();               \
      } else {                                                                \
        auto out_eigen = framework::EigenTensor<Tout, Rank - 1>::From(*out);  \
        out_eigen.device(*(ctx.eigen_device())) =                             \
            in_eigen.eigen_op_type(axis).template cast<Tout>();               \
      }                                                                       \
    }                                                                         \
  }

template <typename DeviceContext, typename T,
          ArgMinMaxV2Type EnumArgMinMaxValue>
struct VisitDataArgMinMaxV2Functor {
  const framework::ExecutionContext& ctx;

  explicit VisitDataArgMinMaxV2Functor(const framework::ExecutionContext& ctx)
      : ctx(ctx) {}
  template <typename Tout>
  void apply() const {
    const auto& x = *(ctx.Input<framework::LoDTensor>("X"));
    auto& out = *(ctx.Output<framework::LoDTensor>("Out"));
    out.template mutable_data<Tout>(ctx.GetPlace());
    auto axis = ctx.Attr<int64_t>("axis");
    const auto& keepdims = ctx.Attr<bool>("keepdims");
    const bool& flatten = ctx.Attr<bool>("flatten");

    // if flatten, will construct the new dims for the cacluate
    framework::DDim x_dims;
    if (flatten) {
      x_dims = framework::make_ddim({x.numel()});
      // if flatten, the axis just as 0
      axis = 0;
    } else {
      x_dims = x.dims();
      if (axis < 0) axis += x_dims.size();
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

#define CALL_ARG_MINMAX_FUNCTOR(rank)                                  \
  ArgMinMaxV2Functor<DeviceContext, T, Tout, rank, EnumArgMinMaxValue> \
      functor##rank;                                                   \
  functor##rank(dev_ctx, x, &out, x_dims, axis, keepdims)

    switch (x_dims.size()) {
      case 1:
        CALL_ARG_MINMAX_FUNCTOR(1);
        break;
      case 2:
        CALL_ARG_MINMAX_FUNCTOR(2);
        break;
      case 3:
        CALL_ARG_MINMAX_FUNCTOR(3);
        break;
      case 4:
        CALL_ARG_MINMAX_FUNCTOR(4);
        break;
      case 5:
        CALL_ARG_MINMAX_FUNCTOR(5);
        break;
      case 6:
        CALL_ARG_MINMAX_FUNCTOR(6);
        break;
      default:
        PADDLE_THROW(
            "%s operator doesn't supports tensors whose ranks are greater "
            "than 6.",
            (EnumArgMinMaxValue == kArgMinV2 ? "argmin" : "argmax"));
        break;
#undef CALL_ARG_MINMAX_FUNCTOR
    }
  }
};

template <typename DeviceContext, typename T,
          ArgMinMaxV2Type EnumArgMinMaxValue>
class ArgMinMaxV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dtype = ctx.Attr<int>("dtype");
    if (dtype < 0) {
      framework::VisitDataType(
          static_cast<framework::proto::VarType::Type>(
              framework::proto::VarType::INT64),
          VisitDataArgMinMaxV2Functor<DeviceContext, T, EnumArgMinMaxValue>(
              ctx));
      return;
    }
    framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(dtype),
        VisitDataArgMinMaxV2Functor<DeviceContext, T, EnumArgMinMaxValue>(ctx));
  }
};

class ArgMinMaxV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "arg_min_max");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "arg_min_max");
    const auto& x_dims = ctx->GetInputDim("X");
    int64_t axis = ctx->Attrs().Get<int64_t>("axis");
    bool keepdims = ctx->Attrs().Get<bool>("keepdims");
    const bool& flatten = ctx->Attrs().Get<bool>("flatten");

    PADDLE_ENFORCE_GE(axis, -x_dims.size(),
                      platform::errors::InvalidArgument(
                          "'axis'(%d) must be greater than or equal to"
                          " -Rank(X)(%d).",
                          axis, -x_dims.size()));
    PADDLE_ENFORCE_LT(
        axis, x_dims.size(),
        platform::errors::InvalidArgument(
            "'axis'(%d) must be less than Rank(X)(%d).", axis, x_dims.size()));

    std::vector<int64_t> vec;
    if (flatten) {
      // if is flatten, will return the only on element
      if (keepdims) {
        vec.emplace_back(static_cast<int64_t>(1));
      }
    } else {
      auto x_rank = x_dims.size();
      if (axis < 0) axis += x_rank;
      for (int64_t i = 0; i < axis; i++) vec.emplace_back(x_dims[i]);
      if (keepdims) {
        vec.emplace_back(static_cast<int64_t>(1));
      }
      for (int64_t i = axis + 1; i < x_rank; i++) vec.emplace_back(x_dims[i]);
    }
    ctx->SetOutputDim("Out", framework::make_ddim(vec));
  }
};

// define the basic op maker for the argmin, argmax
class BaseArgMinMaxOpV2Maker : public framework::OpProtoAndCheckerMaker {
 protected:
  virtual const char* OpName() const = 0;
  virtual const char* Name() const = 0;

 public:
  void Make() override {
    AddInput("X", "Input tensor.");
    AddOutput("Out", "Output tensor.");
    AddAttr<int64_t>("axis", "The axis in which to compute the arg indices.");
    AddAttr<bool>("keepdims", "Keep the dim that to reduce.").SetDefault(false);
    AddAttr<int>("dtype", "Keep the dim that to reduce.").SetDefault(-1);
    AddAttr<bool>("flatten",
                  "Flatten the input value, and search the min or max indices")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
      %s Operator.

      Computes the indices of the %s elements of the input tensor's element
      along the provided axis.
)DOC",
                               OpName(), Name()));
  }
};

// define the ArgMinOpV2Maker
class ArgMinOpV2Maker : public BaseArgMinMaxOpV2Maker {
 protected:
  const char* OpName() const override { return "ArgMinV2"; }
  const char* Name() const override { return "min_v2"; }
};

// define the ArgMaxOpV2Maker
class ArgMaxOpV2Maker : public BaseArgMinMaxOpV2Maker {
 protected:
  const char* OpName() const override { return "ArgMax"; }
  const char* Name() const override { return "max_v2"; }
};

// declare the argmin, argmax functor
DECLARE_ARG_MIN_MAX_FUNCTOR(argmin, ArgMinMaxV2Type::kArgMinV2);
DECLARE_ARG_MIN_MAX_FUNCTOR(argmax, ArgMinMaxV2Type::kArgMaxV2);

// define the ArgMin kernel
template <typename DeviceContext, typename T>
using ArgMinV2Kernel =
    ArgMinMaxV2Kernel<DeviceContext, T, ArgMinMaxV2Type::kArgMinV2>;

// define the ArgMax kernel
template <typename DeviceContext, typename T>
using ArgMaxV2Kernel =
    ArgMinMaxV2Kernel<DeviceContext, T, ArgMinMaxV2Type::kArgMaxV2>;

}  // namespace operators
}  // namespace paddle
