// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle::operators {

class LogspaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class LogspaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Start",
             "Exponent of first entry in the sequence. It is a tensor of "
             "shape [1], should be of type int32, int64, float32 or float64.");
    AddInput("Stop",
             "Exponent of last entry in the sequence. It is a tensor of "
             "shape [1], should be of type int32, int64, float32 or float64.");
    AddInput("Num",
             "Number of entry in the sequence. It is a tensor of shape [1], "
             "should be of type int32.");
    AddInput("Base",
             "Base of the logarithm function. It is a tensor of shape [1], "
             "should be of type int32, int64, float32 or float64.");
    AddAttr<int>("dtype", "The output data type.");
    AddOutput("Out", "A sequence of numbers.");
    AddComment(R"DOC(
        Return fixed number of logarithmically-evenly spaced values within a given
        interval. First entry is exponential of Start with base Base, and last
        entry is exponential of Stop with base Base. In the case when Num is 1,
        only exponential of Start with base Base is returned. If dtype is int32
        or int64, the decimal part of values will be truncated.
        Like logspace function of numpy.
    )DOC");
  }
};
}  // namespace paddle::operators

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(logspace,
                            LogspaceInferShapeFunctor,
                            PD_INFER_META(phi::LogspaceInferMeta));
REGISTER_OPERATOR(
    logspace,
    ops::LogspaceOp,
    ops::LogspaceOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    LogspaceInferShapeFunctor);
