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
#include "paddle/fluid/operators/array_operator.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

class WriteToArrayOp : public ArrayOp {
 public:
  WriteToArrayOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *x = scope.FindVar(Input("X"));
    if (x == nullptr) return;
    auto &x_tensor = x->Get<framework::LoDTensor>();
    size_t offset = GetOffset(scope, place);
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensorArray>();
    if (offset >= out->size()) {
      VLOG(10) << "Resize " << Output("Out") << " from " << out->size()
               << " to " << offset + 1;
      out->resize(offset + 1);
    }
    auto *out_tensor = &out->at(offset);
    out_tensor->set_lod(x_tensor.lod());
    if (x_tensor.memory_size() > 0) {
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);

      TensorCopy(x_tensor, place, dev_ctx, out_tensor);
    } else {
      VLOG(10) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                  "nothing has been written to output array["
               << offset << "].";
    }
  }
};

class WriteToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) the tensor will be written to tensor array");
    AddInput(
        "I",
        "(Tensor) the subscript index in tensor array. The number of element "
        "should be 1");
    AddOutput("Out", "(TensorArray) the tensor array will be written");
    AddComment(R"DOC(
WriteToArray Operator.

This operator writes a LoDTensor to a LoDTensor array.

Assume $T$ is LoDTensor, $i$ is the subscript of the array, and $A$ is the array. The
equation is

$$A[i] = T$$

)DOC");
  }
};

class WriteToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("I"), "Must set the subscript index");
    if (context->IsRuntime()) {
      PADDLE_ENFORCE_EQ(framework::product(context->GetInputDim("I")), 1,
                        "The number of element of subscript index must be 1");
    }
    if (!context->HasInput("X")) {
      return;
    }

    PADDLE_ENFORCE(context->HasOutput("Out"), NotHasOutError());
    context->SetOutputDim("Out", context->GetInputDim("X"));

    // When compile time, we need to:
    // - for ReadFromArray, share tensor_array X's lod_level to Out
    // - for WriteToArray, share X's lod_level to tensor_array Out
    // When runtime, we need to:
    // - for ReadFromArray, share X[I]'s lod to Out
    // - for WriteToArray, share X's lod to Out[I]
    // but we cannot get I's value here, so leave this work to detail
    // kernel implementation.
    if (!context->IsRuntime()) {
      context->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  virtual const char *NotHasXError() const { return "Must set the lod tensor"; }

  virtual const char *NotHasOutError() const {
    return "Must set the lod tensor array";
  }
};

class WriteToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = ctx->Input("X")[0];
    auto out_name = ctx->Output("Out")[0];
    VLOG(10) << "Set Variable " << out_name << " as LOD_TENSOR_ARRAY";
    ctx->SetType(out_name, framework::proto::VarType::LOD_TENSOR_ARRAY);
    if (ctx->HasVar(x_name)) {
      ctx->SetDataType(out_name, ctx->GetDataType(x_name));
    }
  }
};

class ReadFromArrayOp : public ArrayOp {
 public:
  ReadFromArrayOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *x = scope.FindVar(Input("X"));
    PADDLE_ENFORCE(x != nullptr, "X must be set");
    auto &x_array = x->Get<framework::LoDTensorArray>();
    auto *out = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE(out != nullptr, "Out must be set");
    size_t offset = GetOffset(scope, place);
    if (offset < x_array.size()) {
      auto *out_tensor = out->GetMutable<framework::LoDTensor>();
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);
      framework::TensorCopy(x_array[offset], place, dev_ctx, out_tensor);
      out_tensor->set_lod(x_array[offset].lod());
    } else {
      VLOG(10) << "offset " << offset << " >= " << x_array.size();
      // set grad of the writed tensor to 0 when used as write_to_array_grad
      auto *fw_var = scope.FindVar(Input("X_W"));
      if (fw_var == nullptr) return;
      auto &fw_var_tensor = fw_var->Get<framework::LoDTensor>();

      framework::AttributeMap attrs;
      attrs["dtype"] = fw_var_tensor.type();
      attrs["shape"] = framework::vectorize<int>(fw_var_tensor.dims());
      attrs["value"] = 0.0f;

      auto zero_op = framework::OpRegistry::CreateOp(
          "fill_constant", {}, {{"Out", {Output("Out")}}}, attrs);
      zero_op->Run(scope, place);
      auto *out_tensor = out->GetMutable<framework::LoDTensor>();
      out_tensor->set_lod(fw_var_tensor.lod());
    }
  }
};

class ReadFromArrayProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(TensorArray) the array will be read from.");
    AddInput("I",
             "(Tensor) the subscript index in tensor array. The number of "
             "element should be 1");
    AddInput("X_W",
             "(Tensor) the writed tensor when used as the grad op of "
             "write_to_array. We use this to fill zero gradient.")
        .AsDispensable();
    AddOutput("Out", "(LoDTensor) the tensor will be read from.");
    AddComment(R"DOC(
ReadFromArray Operator.

Read a LoDTensor from a LoDTensor Array.

Assume $T$ is LoDTensor, $i$ is the subscript of the array, and $A$ is the array. The
equation is

$$T = A[i]$$

)DOC");
  }
};

class ReadFromArrayInferShape : public WriteToArrayInferShape {
 protected:
  const char *NotHasXError() const override {
    return "The input array X must be set";
  }
  const char *NotHasOutError() const override {
    return "The output tensor out must be set";
  }
};

template <typename T>
class WriteToArrayGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("read_from_array");
    grad_op->SetInput("I", this->Input("I"));
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetInput("X_W", this->Input("X"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

template <typename T>
class ReadFromArrayGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("write_to_array");
    grad_op->SetInput("I", this->Input("I"));
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(write_to_array, ops::WriteToArrayOp,
                  ops::WriteToArrayInferShape, ops::WriteToArrayOpProtoMaker,
                  ops::WriteToArrayGradMaker<paddle::framework::OpDesc>,
                  ops::WriteToArrayGradMaker<paddle::imperative::OpBase>,
                  ops::WriteToArrayInferVarType);
REGISTER_OPERATOR(read_from_array, ops::ReadFromArrayOp,
                  ops::ReadFromArrayInferShape, ops::ReadFromArrayProtoMaker,
                  ops::ReadFromArrayGradMaker<paddle::framework::OpDesc>,
                  ops::ReadFromArrayGradMaker<paddle::imperative::OpBase>);
