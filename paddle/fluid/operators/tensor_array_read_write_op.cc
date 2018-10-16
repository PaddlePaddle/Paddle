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
    PADDLE_ENFORCE_EQ(framework::product(context->GetInputDim("I")), 1,
                      "The number of element of subscript index must be 1");
    if (!context->HasInput("X")) {
      return;
    }
    PADDLE_ENFORCE(context->HasOutput("Out"), NotHasOutError());
    context->SetOutputDim("Out", context->GetInputDim("X"));
  }

 protected:
  virtual const char *NotHasXError() const { return "Must set the lod tensor"; }

  virtual const char *NotHasOutError() const {
    return "Must set the lod tensor array";
  }
};

class WriteToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto x_name = op_desc.Input("X")[0];
    auto out_name = op_desc.Output("Out")[0];
    VLOG(10) << "Set Variable " << out_name << " as LOD_TENSOR_ARRAY";
    auto &out = block->FindRecursiveOrCreateVar(out_name);
    out.SetType(framework::proto::VarType::LOD_TENSOR_ARRAY);
    auto *x = block->FindVarRecursive(x_name);
    if (x != nullptr) {
      out.SetDataType(x->GetDataType());
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

class WriteToArrayGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("read_from_array");
    grad_op->SetInput("I", Input("I"));
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class ReadFromArrayGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("write_to_array");
    grad_op->SetInput("I", Input("I"));
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(write_to_array, ops::WriteToArrayOp,
                  ops::WriteToArrayInferShape, ops::WriteToArrayOpProtoMaker,
                  ops::WriteToArrayGradMaker, ops::WriteToArrayInferVarType);
REGISTER_OPERATOR(read_from_array, ops::ReadFromArrayOp,
                  ops::ReadFromArrayInferShape, ops::ReadFromArrayProtoMaker,
                  ops::ReadFromArrayGradMaker);
