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

#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {
using framework::Tensor;

void LodTensorArray2LodTensorVector(const framework::Scope &scope,
                                    std::string base_name,
                                    const std::string lod_tensor_array_name,
                                    std::vector<std::string> *res_names) {
  auto &inx =
      scope.FindVar(lod_tensor_array_name)->Get<framework::LoDTensorArray>();
  for (size_t i = 0; i < inx.size(); i++) {
    std::string var_name = base_name + std::to_string(i);
    framework::Variable *g_feed_value =
        const_cast<framework::Scope &>(scope).Var(var_name);
    auto &feed_input =
        *(g_feed_value->GetMutable<paddle::framework::LoDTensor>());
    feed_input.ShareDataWith(inx[i]);
    res_names->push_back(var_name);
  }
}

void LodTensorVectorResizefromLodTensorArray(
    const framework::Scope &scope, std::string base_name,
    const std::string lod_tensor_array_name,
    std::vector<std::string> *res_names) {
  auto &inx =
      scope.FindVar(lod_tensor_array_name)->Get<framework::LoDTensorArray>();
  for (size_t i = 0; i < inx.size(); i++) {
    std::string var_name = base_name + std::to_string(i);
    framework::Variable *g_feed_value =
        const_cast<framework::Scope &>(scope).Var(var_name);
    auto &feed_input =
        *(g_feed_value->GetMutable<paddle::framework::LoDTensor>());
    auto dims = inx[i].dims();
    feed_input.Resize(dims);
    res_names->push_back(var_name);
  }
}

void LodTensorArrayResizefromLodTensorArray(
    const framework::Scope &scope,
    const std::string input_lod_tensor_array_name,
    const std::string output_lod_tensor_array_name) {
  auto &inx = scope.FindVar(input_lod_tensor_array_name)
                  ->Get<framework::LoDTensorArray>();
  auto &grad_inx = *scope.FindVar(output_lod_tensor_array_name)
                        ->GetMutable<framework::LoDTensorArray>();

  for (size_t i = 0; i < inx.size(); i++) {
    std::string var_name = output_lod_tensor_array_name + std::to_string(i);
    framework::Variable *g_feed_value =
        const_cast<framework::Scope &>(scope).Var(var_name);
    auto &feed_input =
        *(g_feed_value->GetMutable<paddle::framework::LoDTensor>());
    auto dims = inx[i].dims();
    feed_input.Resize(dims);
    grad_inx.push_back(feed_input);
  }
}

class LoDTensorArrayConcatOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto axis = Attr<int>("axis");

    framework::AttributeMap attrs;
    attrs["axis"] = axis;

    // auto* ins = ctx.Input<framework::LoDTensorArray>("X");
    auto &inx = scope.FindVar(Input("X"))->Get<framework::LoDTensorArray>();
    auto &out =
        *scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();

    std::string base_name = Inputs("X")[0];
    std::vector<std::string> names;
    names.clear();

    auto out_dims = inx[0].dims();
    out_dims[axis] = inx.size();
    out.Resize(out_dims);

    LodTensorArray2LodTensorVector(scope, base_name, Input("X"), &names);
    // Invoke Reshape Op
    auto concat_op = framework::OpRegistry::CreateOp(
        "concat", {{"X", names}}, {{"Out", {Output("Out")}}}, attrs);

    concat_op->Run(scope, place);
  }
};

class LoDTensorArrayConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input LoDTensorArray of concat operator.");
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated.")
        .SetDefault(0);
    AddComment(R"DOC(
Concat Operator.

Concatenate the input LoDTensorArray along dimension axis.
Examples:
  Input = {[1,2], [3,4], [5,6]}
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]

)DOC");
  }
};

class LoDTensorArrayConcatOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

class LoDTensorArrayConcatGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {}
};

class LoDTensorArrayConcatGradInferVarType
    : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    for (auto &out_var : op_desc.Output(framework::GradVarName("X"))) {
      block->Var(out_var)->SetType(framework::proto::VarType::LOD_TENSOR_ARRAY);
    }
  }
};

class LoDTensorArrayConcatGradOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto axis = Attr<int>("axis");

    framework::AttributeMap attrs;
    attrs["axis"] = axis;

    std::string base_name = Inputs("X")[0];
    std::vector<std::string> names;
    names.clear();

    LodTensorArray2LodTensorVector(scope, base_name, Input("X"), &names);

    // grad
    auto dx_name = Output(framework::GradVarName("X"));
    auto dout_name = Input(framework::GradVarName("Out"));

    std::vector<std::string> grad_names;
    grad_names.clear();

    LodTensorVectorResizefromLodTensorArray(scope, "grad_name", Input("X"),
                                            &grad_names);

    auto concat_grad_op = framework::OpRegistry::CreateOp(
        "concat_grad", {{"X", names}, {"Out@GRAD", {dout_name}}},
        {{"X@GRAD", grad_names}}, attrs);

    concat_grad_op->Run(scope, place);

    LodTensorArrayResizefromLodTensorArray(scope, Input("X"), dx_name);
    auto &grad_inx =
        *scope.FindVar(dx_name)->GetMutable<framework::LoDTensorArray>();

    for (size_t i = 0; i < grad_names.size(); i++) {
      std::string var_name = grad_names[i];
      auto &feed_input = scope.FindVar(var_name)->Get<framework::LoDTensor>();
      grad_inx[i].ShareDataWith(feed_input);
    }
  }
};

}  // namespace operators
}  // namespace paddle
USE_OP(concat);
// USE_OP(concat_grad);

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_tensor_array_concat, ops::LoDTensorArrayConcatOp,
                  ops::LoDTensorArrayConcatOpMaker,
                  ops::LoDTensorArrayConcatOpInferShape,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(lod_tensor_array_concat_grad, ops::LoDTensorArrayConcatGradOp,
                  ops::LoDTensorArrayConcatGradInferShape,
                  ops::LoDTensorArrayConcatGradInferVarType);
