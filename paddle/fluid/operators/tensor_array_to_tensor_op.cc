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

#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {
using framework::Tensor;

void LodTensorArray2LodTensorVector(const framework::Scope &scope,
                                    const std::string &base_name,
                                    const std::string &lod_tensor_array_name,
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

void LodTensorVectorResizeFromLodTensorArray(
    const framework::Scope &scope, const std::string &base_name,
    const std::string &lod_tensor_array_name,
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

void LodTensorArrayCreateFromLodTensorArray(
    const framework::Scope &scope,
    const std::string &input_lod_tensor_array_name,
    const std::string &output_lod_tensor_array_name) {
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
    grad_inx.push_back(feed_input);
  }
}

class LoDTensorArray2TensorOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto axis = Attr<int>("axis");

    framework::AttributeMap attrs;
    attrs["axis"] = axis;

    auto &inx = scope.FindVar(Input("X"))->Get<framework::LoDTensorArray>();
    auto &out =
        *scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();
    auto &out_inx =
        *scope.FindVar(Output("OutIndex"))->GetMutable<framework::LoDTensor>();

    const size_t n = inx.size();
    PADDLE_ENFORCE_GT(n, 0, platform::errors::InvalidArgument(
                                "Input tensorarray size should > 0,"
                                "but the received is %d",
                                n));

    std::string base_name = Inputs("X")[0];
    std::vector<std::string> names;

    // get the input tensorarray items' dim in out_inx
    auto out_inx_dim = out_inx.dims();
    out_inx_dim[0] = inx.size();
    out_inx.Resize(out_inx_dim);
    int *tmp_index_data = out_inx.mutable_data<int>(platform::CPUPlace());

    auto out_dims = inx[0].dims();
    size_t out_dim_sum = 0;
    for (size_t index = 0; index < inx.size(); index++) {
      auto inx_dims = inx[index].dims();
      out_dim_sum += inx_dims[axis];
      tmp_index_data[index] = inx_dims[axis];
    }

    // get input array items' dims
    out_dims[axis] = out_dim_sum;
    out.Resize(out_dims);

    LodTensorArray2LodTensorVector(scope, base_name, Input("X"), &names);

    auto use_stack = Attr<bool>("use_stack");

    // Invoke concat Op or stack Op
    auto op =
        use_stack
            ? framework::OpRegistry::CreateOp("stack", {{"X", names}},
                                              {{"Y", {Output("Out")}}}, attrs)
            : framework::OpRegistry::CreateOp(
                  "concat", {{"X", names}}, {{"Out", {Output("Out")}}}, attrs);

    op->Run(scope, place);
  }
};

class LoDTensorArray2TensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input LoDTensorArray of tensor_array_to_tensor operator.");
    AddOutput("Out", "Output tensor of tensor_array_to_tensor operator.");
    AddOutput("OutIndex",
              "Output input LoDTensorArray items' dims of "
              "tensor_array_to_tensor operator.");
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated.")
        .SetDefault(0);
    AddAttr<bool>("use_stack",
                  "Act as concat_op or stack_op. For stack mode, all tensors "
                  "in the tensor array must have the same shape.")
        .SetDefault(false);
    AddComment(R"DOC(
tensor_array_to_tensor Operator.

If use concat mode, concatenate all tensors in the input LoDTensorArray along
axis into the output Tensor.

Examples:
  Input = {[1,2], [3,4], [5,6]}
  axis = 0
  Output = [1,2,3,4,5,6]
  OutputIndex = [2,2,2]

If use stack mode, stack all tensors in the input LoDTensorArray along axis into
the output Tensor.

Examples:
  Input = {[1,2], [3,4], [5,6]}
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]
  OutputIndex = [2,2,2]

)DOC");
  }
};

class LoDTensorArray2TensorOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    // in runtime, shape is determined by RunImpl
    if (ctx->IsRuntime()) return;
    auto dims = ctx->GetInputDim("X");
    // if the shape is empty
    if (dims == phi::make_ddim({0UL})) return;
    // otherwise, suppose the shape of array is the shape of tensor in the
    // array, which is consistent with what tensor_array_read_write dose
    auto axis = ctx->Attrs().Get<int>("axis");
    auto use_stack = ctx->Attrs().Get<bool>("use_stack");
    if (use_stack) {
      auto dim_vec = phi::vectorize<int>(dims);
      // use -1 for the stack dim size
      dim_vec.insert(dim_vec.begin() + axis, -1);
      dims = phi::make_ddim(dim_vec);
    } else {
      // use -1 for the concat dim size
      dims[axis] = -1;
    }
    ctx->SetOutputDim("Out", dims);
  }
};

class LoDTensorArray2TensorGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class LoDTensorArray2TensorGradInferVarType
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType(framework::GradVarName("X"),
                       framework::proto::VarType::LOD_TENSOR_ARRAY,
                       framework::ALL_ELEMENTS);
  }
};

class LoDTensorArray2TensorGradOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto axis = Attr<int>("axis");
    framework::AttributeMap attrs;
    attrs["axis"] = axis;

    auto &inx = scope.FindVar(Input("X"))->Get<framework::LoDTensorArray>();
    const size_t n = inx.size();
    PADDLE_ENFORCE_GT(n, 0, platform::errors::InvalidArgument(
                                "Input tensorarray size should > 0, "
                                "but the received is: %d. ",
                                n));

    std::string base_name = Inputs("X")[0];
    std::vector<std::string> names;

    LodTensorArray2LodTensorVector(scope, base_name, Input("X"), &names);

    // grad
    auto dx_name = Output(framework::GradVarName("X"));
    auto dout_name = Input(framework::GradVarName("Out"));

    std::vector<std::string> grad_names;
    // NOTE(Aurelius84): Generating grad base name by Input("X") instead of
    // fixed string to avoid incorrectly sharing same var's allocation in
    // multi-thread that will cause wrong calculation result.
    std::string grad_base_name = base_name + "_temp_grad_";

    LodTensorVectorResizeFromLodTensorArray(scope, grad_base_name, Input("X"),
                                            &grad_names);

    auto use_stack = Attr<bool>("use_stack");

    auto grad_op = use_stack ? framework::OpRegistry::CreateOp(
                                   "stack_grad", {{"Y@GRAD", {dout_name}}},
                                   {{"X@GRAD", grad_names}}, attrs)
                             : framework::OpRegistry::CreateOp(
                                   "concat_grad",
                                   {{"X", names}, {"Out@GRAD", {dout_name}}},
                                   {{"X@GRAD", grad_names}}, attrs);

    grad_op->Run(scope, place);

    LodTensorArrayCreateFromLodTensorArray(scope, Input("X"), dx_name);
    auto &grad_inx =
        *scope.FindVar(dx_name)->GetMutable<framework::LoDTensorArray>();

    for (size_t i = 0; i < grad_names.size(); i++) {
      std::string var_name = grad_names[i];
      auto &feed_input = scope.FindVar(var_name)->Get<framework::LoDTensor>();
      grad_inx[i].ShareDataWith(feed_input);
    }
  }
};

template <typename T>
class TensorArrayToTensorGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tensor_array_to_tensor_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle
USE_OP_ITSELF(concat);

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    tensor_array_to_tensor, ops::LoDTensorArray2TensorOp,
    ops::LoDTensorArray2TensorOpMaker, ops::LoDTensorArray2TensorOpInferShape,
    ops::TensorArrayToTensorGradOpMaker<paddle::framework::OpDesc>,
    ops::TensorArrayToTensorGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(tensor_array_to_tensor_grad, ops::LoDTensorArray2TensorGradOp,
                  ops::LoDTensorArray2TensorGradInferShape,
                  ops::LoDTensorArray2TensorGradInferVarType);
