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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

class SplitSelectedVarOp : public framework::OperatorBase {
 public:
  SplitSelectedVarOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    auto &mask = scope.FindVar(Input("Mask"))->Get<framework::LoDTensor>();
    size_t output_branch = static_cast<size_t>(GetOutputBranch(mask, dev_ctx));

    const std::vector<std::string> &out_names = Outputs("Out");
    PADDLE_ENFORCE_LT(output_branch, out_names.size(),
                      "Selected branch number is greater than actual branch "
                      "num in MergeSelectedVarOp");

    const framework::Variable *x = scope.FindVar(Input("X"));
    framework::Variable *selected_out = scope.FindVar(out_names[output_branch]);

    if (x->IsType<framework::LoDTensor>()) {
      auto &to_copy = x->Get<framework::LoDTensor>();
      auto *out = selected_out->GetMutable<framework::LoDTensor>();
      framework::TensorCopy(to_copy, to_copy.place(), dev_ctx, out);
      out->set_lod(to_copy.lod());
    } else if (x->IsType<framework::SelectedRows>()) {
      auto &to_copy = x->Get<framework::SelectedRows>();
      auto *out = selected_out->GetMutable<framework::SelectedRows>();
      out->set_rows(to_copy.rows());
      out->set_height(to_copy.height());
      framework::TensorCopy(to_copy.value(), to_copy.place(), dev_ctx,
                            out->mutable_value());
    } else if (x->IsType<framework::LoDTensorArray>()) {
      auto &to_copy = x->Get<framework::LoDTensorArray>();
      auto *out = selected_out->GetMutable<framework::LoDTensorArray>();
      out->resize(to_copy.size());
      for (size_t i = 0; i < to_copy.size(); ++i) {
        framework::LoDTensor &out_tensor = (*out)[i];
        framework::TensorCopy(to_copy[i], to_copy[i].place(), dev_ctx,
                              &out_tensor);
        out_tensor.set_lod(to_copy[i].lod());
      }
    } else {
      PADDLE_THROW("Type %s is not supported in MergeSelectedVarOp",
                   framework::ToTypeName(x->Type()));
    }
  }

  // Returns which branch to output
  int GetOutputBranch(const framework::LoDTensor &mask,
                      const platform::DeviceContext &dev_ctx) const {
    PADDLE_ENFORCE_EQ(mask.numel(), 1,
                      "Mask in SplitSelectedVarOp must have numel 1.");
    if (platform::is_cpu_place(mask.place())) {
      return mask.data<int>()[0];
    }
    // when platform::is_gpu_place(mask.place()) is ture
    std::unique_ptr<framework::LoDTensor> cpu_mask{new framework::LoDTensor()};
#ifdef PADDLE_WITH_CUDA
    framework::TensorCopy(mask, platform::CPUPlace(), dev_ctx, cpu_mask.get());
#else
    PADDLE_THROW(
        "This version of PaddlePaddle doen NOT support GPU but got GPU tensor "
        "Mask in SplitSelectedVarOp. Please compile WITH_GPU option");
#endif
    return cpu_mask->data<int>()[0];
  }
};

class SplitSelectedVarOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input LoDTensor or LoDTensorArray or SelectedRows.");
    AddInput("Mask", "Tensor with numel 1 specifying which branch to output");
    AddOutput("Out",
              "The output can contains multiple variables. The output of "
              "selected branch will be same as input. We do nothing for "
              "variables in other branch")
        .AsDuplicable();
    // TODO(huihuangzheng): decide whether to add support for lod level
    // Because this op is blocking whole control flow. I am implementing MVP
    // (minimal viable product) here.
    AddComment(R"DOC(
Split input variable into one output branch. The mask is an integer tensor to
specify which output branch should copy the input. 
)DOC");
  }
};

class SplitSelectedVarInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "SplitSelectedVarOp must has input X.");
    PADDLE_ENFORCE(context->HasInput("Mask"),
                   "SplitSelectedVarOp must has input Mask.");
    PADDLE_ENFORCE(context->HasOutputs("Out"),
                   "SplitSelectedVarOp must has output Out.");
  }
};

template <typename T>
class SplitSelectedVarGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("merge_selected_var");
    grad_op->SetInput("Mask", this->Input("Mask"));
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(split_selected_var, ops::SplitSelectedVarOp,
                  ops::SplitSelectedVarOpProtoMaker,
                  ops::SplitSelectedVarInferShape,
                  ops::SplitSelectedVarGradMaker<paddle::framework::OpDesc>,
                  ops::SplitSelectedVarGradMaker<paddle::imperative::OpBase>);
