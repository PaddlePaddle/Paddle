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

#include <array>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
class Variable;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

// const char kCond[] = "Cond";
std::array<const char, 5> kCond = {"Cond"};
std::array<const char, 5> kData = {"Data"};
std::array<const char, 10> kSummarize = {"summarize"};

namespace paddle {
namespace operators {

class AssertOp : public framework::OperatorBase {
 public:
  AssertOp(const std::string &type,
           const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {
    const framework::Variable *cond_var_ptr =
        scope.FindVar(Input(kCond.data()));
    PADDLE_ENFORCE_NOT_NULL(
        cond_var_ptr,
        common::errors::NotFound("Input(Condition) of AssertOp is not found."));
    const phi::DenseTensor &cond = cond_var_ptr->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        cond.numel(),
        1,
        common::errors::InvalidArgument(
            "The numel of Input(Condition) of AssertOp must be 1. But now "
            "the Condition's shape is %s.",
            cond.dims().to_str()));

    bool cond_data = GetCondData(cond);
    if (cond_data) {
      return;
    }

    funcs::TensorFormatter formatter;
    formatter.SetSummarize(Attr<int64_t>(kSummarize.data()));

    const std::vector<std::string> &x_names = Inputs(kData.data());
    for (const std::string &name : x_names) {
      const framework::Variable *x_var_ptr = scope.FindVar(name);
      const phi::DenseTensor &x_tensor = x_var_ptr->Get<phi::DenseTensor>();
      formatter.Print(x_tensor, name);
    }

    PADDLE_THROW(common::errors::InvalidArgument(
        "The condition variable '%s' of AssertOp must be "
        "true, but received false",
        Input(kCond.data())));
  }
};

class AssertOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        kCond.data(),
        "The boolean scalar condition tensor which is asserted to be true.");
    AddInput(kData.data(),
             "The tensors to print when the assert condition is not true.")
        .AsDuplicable();
    AddAttr<int64_t>(
        kSummarize.data(),
        "The number of entries of each tensor to print when the "
        "assert condition is not true. -1 means print all entries. If "
        "the number of entries of a tensor is less then "
        "summarize_num, this OP will print all entries of the tensor.")
        .SetDefault(-1);
    AddComment(
        R"DOC(Assert the input Condition Tensor is true and print Tensors if the Condition Tensor is false.)DOC");
  }
};

class AssertOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(
        context->HasInputs(kCond.data()), "Input", "Condition", "AssertOp");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    assert,
    ops::AssertOp,
    ops::AssertOpProtoMaker,
    ops::AssertOpInferShape,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
