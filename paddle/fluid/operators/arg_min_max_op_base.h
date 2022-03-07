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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {

class ArgMinMaxOp : public framework::OperatorWithKernel {
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
            "'axis'(%d) must be less than Rank(X)(%d) of Input(X).", axis,
            x_dims.size()));

    const int& dtype = ctx->Attrs().Get<int>("dtype");
    PADDLE_ENFORCE_EQ(
        (dtype < 0 || dtype == 2 || dtype == 3), true,
        platform::errors::InvalidArgument(
            "The attribute of dtype in argmin/argmax must be [%s] or [%s], but "
            "received [%s]",
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64),
            paddle::framework::DataTypeToString(
                static_cast<framework::proto::VarType::Type>(dtype))));

    auto x_rank = x_dims.size();
    if (axis < 0) axis += x_rank;
    if (ctx->IsRuntime()) {
      if (dtype == framework::proto::VarType::INT32) {
        int64_t all_element_num = 0;
        if (flatten) {
          all_element_num = phi::product(x_dims);

        } else {
          all_element_num = x_dims[axis];
        }
        PADDLE_ENFORCE_LE(
            all_element_num, INT_MAX,
            platform::errors::InvalidArgument(
                "The element num of the argmin/argmax input at axis is "
                "%d, is larger than int32 maximum value:%d, you must "
                "set the dtype of argmin/argmax to 'int64'.",
                all_element_num, INT_MAX));
      }
    }
    std::vector<int64_t> vec;
    if (flatten) {
      vec.emplace_back(static_cast<int64_t>(1));
    } else {
      for (int64_t i = 0; i < axis; i++) vec.emplace_back(x_dims[i]);
      if (keepdims) {
        vec.emplace_back(static_cast<int64_t>(1));
      }
      for (int64_t i = axis + 1; i < x_rank; i++) vec.emplace_back(x_dims[i]);
    }
    ctx->SetOutputDim("Out", phi::make_ddim(vec));
  }
};

class BaseArgMinMaxOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  virtual const char* OpName() const = 0;
  virtual const char* Name() const = 0;

 public:
  void Make() override {
    AddInput("X", "Input tensor.");
    AddOutput("Out", "Output tensor.");
    AddAttr<int64_t>("axis", "The axis in which to compute the arg indics.");
    AddAttr<bool>("keepdims", "Keep the dim that to reduce.").SetDefault(false);
    AddAttr<bool>("flatten",
                  "Flatten the input value, and search the min or max indices")
        .SetDefault(false);
    AddAttr<int>("dtype",
                 "(int, 3), the dtype of indices, the indices dtype must be "
                 "int32, int64."
                 "default dtype is int64, and proto value is 3.")
        .SetDefault(3);
    AddComment(string::Sprintf(R"DOC(
      %s Operator.

      Computes the indices of the %s elements of the input tensor's element
      along the provided axis.
)DOC",
                               OpName(), Name()));
  }
};

class ArgMinOpMaker : public BaseArgMinMaxOpMaker {
 protected:
  const char* OpName() const override { return "ArgMin"; }
  const char* Name() const override { return "min"; }
};

class ArgMaxOpMaker : public BaseArgMinMaxOpMaker {
 protected:
  const char* OpName() const override { return "ArgMax"; }
  const char* Name() const override { return "max"; }
};
}  // namespace operators
}  // namespace paddle
