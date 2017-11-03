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
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class LoDTensorToArrayOp : public framework::OperatorBase {
 public:
  LoDTensorToArrayOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {}
};

class LoDTensorToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoDTensorToArrayOpProtoMaker(framework::OpProto *proto,
                               framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("RankTable", "");
    AddOutput("Out", "");
    AddComment("");
  }
};

class LoDTensorToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {}
};

class LoDTensorToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    for (auto &out_var : op_desc.Output("Out")) {
      block->Var(out_var)->SetType(framework::VarDesc::LOD_TENSOR_ARRAY);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_tensor_to_array, ops::LoDTensorToArrayOp,
                  ops::LoDTensorToArrayOpProtoMaker,
                  ops::LoDTensorToArrayInferShape,
                  ops::LoDTensorToArrayInferVarType);
