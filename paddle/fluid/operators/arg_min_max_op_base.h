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
