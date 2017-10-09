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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

#include <stdio.h>
#include <string.h>
#include <fstream>

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::LoDTensor;

class SaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"),
                   "Input(X) of SaveOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("absolutePath"),
                   "Input(absolutePath) of SaveOp should not be null.");
  }
};

class SaveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(tensor), the tensor count can be 1~INT_MAX, tensors names which "
             "values will be saved.")
        .AsDuplicable()
        .NotInGradient();
    AddAttr<std::string>("absolutePath", "the absolutePath for save model.");
    AddComment(R"DOC(
Save the input tensors to a binary file based on input tensor names and absolute path.

All the inputs can carry the LoD (Level of Details) information,
or not.
)DOC");
  }
};

template <typename T>
class SaveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<LoDTensor>("X");
    std::string absolutePath = ctx.template Attr<std::string>("absolutePath");

    std::ofstream fout(absolutePath, std::fstream::app);
    PADDLE_ENFORCE(!fout.is_open(), "open file for model failed.");
    for (size_t i = 0; i < ins.size(); ++i) {
      std::string bytes = ins[i]->SerializeToString();
      fout << bytes << '\n';
    }

    fout.close();
  }
};

class RestoreOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutputs("Out"),
                   "Output(X) of RestoreOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("absolutePath"),
                   "Input(absolutePath) of Restore Op should not be null.");
  }
};

class RestoreOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RestoreOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out",
              "(tensor), the tensor count can be 1~INT_MAX, tensors which "
              "values will be restores.")
        .AsDuplicable()
        .NotInGradient();
    AddAttr<std::string>("absolutePath", "the absolutePath for model file.");
    AddComment(R"DOC(
Restore the tensors from model file based on absolute path.

All the tensors outputs may carry the LoD (Level of Details) information,
or not.
)DOC");
  }
};

template <typename Place, typename T>
class RestoreKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto outs = ctx.MultiOutput<LoDTensor>("Out");
    std::string absolutePath = ctx.template Attr<std::string>("absolutePath");

    std::ifstream fin(absolutePath);
    PADDLE_ENFORCE(!fin.is_open(), "open model file failed.");

    std::string line;
    int i = 0;
    while (std::getline(fin, line)) {
      outs[i++]->DeserializeFromString(line, ctx.GetPlace());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(save, ops::SaveOp, ops::SaveOpMaker);
REGISTER_OP_CPU_KERNEL(save, ops::SaveKernel<float>);

REGISTER_OP_WITHOUT_GRADIENT(restore, ops::RestoreOp, ops::RestoreOpMaker);
REGISTER_OP_CPU_KERNEL(restore,
                       ops::RestoreKernel<paddle::platform::CPUPlace, float>);
