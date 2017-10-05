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

namespace paddle {
namespace operators {

using framework::Tensor;

class SaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"),
                   "Input(X) of SaveOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("absolute_path"),
                   "Input(absolute_path) of SaveOp should not be null.");
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
    AddAttr<std::string>("absolute_path", "the absolute_path for save model.");
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
  void Compute(const framework::ExecutionContext& context) const override {
    auto ins = context.MultiInput<Tensor>("X");
    std::string absolute_path = ctx.Attr<std::string>("absolute_path");

    FILE* fp;
    fp = fopen(absolute_path.c_str())
        PADDLE_ENFORCE(fp != nullptr, "open file for model failed.");
    int N = ins.size();
    for (int i = 0; i < N; i++) {
      // at present, we only support tensor serialization instead of variable

      std::string bytes = ins[i].SerializeToString();
      size_t count =
          fwrite(bytes.c_str(), sizeof(char), sizeof(char) * bytes.size(), fp);
      PADDLE_ENFORCE(count == bytes.size(), "write to model file failed.");
    }
    fclose(fp);
  }
};

class RestoreOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"),
                   "Input(X) of RestoreOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("absolute_path"),
                   "Input(absolute_path) of Restore Op should not be null.");
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
    AddAttr<std::string>("absolute_path", "the absolute_path for model file.");
    AddComment(R"DOC(
Restore the tensors from model file based on absolute path.

All the tensors outputs may carry the LoD (Level of Details) information,
or not.
)DOC");
  }
};

template <typename T>
class RestoreKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto outs = context.MultiOutput<std::string>("Out");
    std::string absolute_path = ctx.Attr<std::string>("absolute_path");

    FILE* fp;
    fp = fopen(absolute_path.c_str())
        PADDLE_ENFORCE(fp != nullptr, "open model file failed.");
    for (int i = 0; i < N; i++) {
      // at present, we only support tensor serialization instead of variable

      std::string bytes = ins[i].DeserializeFromString();
      size_t count =
          fwrite(bytes.c_str(), sizeof(char), sizeof(char) * bytes.size(), fp);
      PADDLE_ENFORCE(count == bytes.size(), "write to model file failed.");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(save, ops::SaveOp, ops::SaveOpMaker);
REGISTER_OP_CPU_KERNEL(save,
                       ops::SaveKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_WITHOUT_GRADIENT(save, ops::SaveOp, ops::SaveOpMaker);
REGISTER_OP_CPU_KERNEL(save,
                       ops::RestoreKernel<paddle::platform::CPUPlace, float>);
