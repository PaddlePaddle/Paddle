/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
namespace paddle {
namespace operators {
class UnaryOp : public framework::OperatorWithKernel {
 public:
  UnaryOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto *in = ctx.Input<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");
    out->Resize(in->dims());
  }
};

class UnaryOpInformation {
 public:
  virtual ~UnaryOpInformation() {}
  virtual std::string Name() const = 0;
  virtual std::string Comment() const = 0;
  virtual void AddAttrs(framework::OpProtoAndCheckerMaker *maker) const {};
};

template <typename OpInformation>
class UnaryOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  UnaryOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    OpInformation info;
    AddInput("X",
             string::Sprintf(
                 "The input tensor of %s operator, which num of elements is N",
                 info.Name()));
    AddOutput(
        "Out",
        string::Sprintf(
            "The output tensor of %s operator, which num of elements is N",
            info.Name()));

    AddComment(info.Comment());
    info.AddAttrs(this);
  }
};

}  // namespace operators
}  // namespace paddle
