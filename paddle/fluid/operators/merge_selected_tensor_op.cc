/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace operators {

class MergeSelectedTensorOp : public framework::OperatorBase {
 public:
  MergeSelectedTensorOp(const std::string &type,
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

    const std::vector<std::string> &x_names = Inputs("X");
    PADDLE_ENFORCE_LT(output_branch, x_names.size(),
                      "Selected branch number is greater than actual branch "
                      "num in MergeSelectedTensorOp");

    const framework::Variable *selected_x =
        scope.FindVar(x_names[output_branch]);

    if (selected_x->IsType<framework::LoDTensor>()) {
      auto &to_copy = selected_x->Get<framework::LoDTensor>();
      auto *out =
          scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();
      framework::TensorCopy(to_copy, to_copy.place(), dev_ctx, out);
      out->set_lod(to_copy.lod());
    } else if (selected_x->IsType<framework::SelectedRows>()) {
      auto &to_copy = selected_x->Get<framework::SelectedRows>();
      auto *out =
          scope.FindVar(Output("Out"))->GetMutable<framework::SelectedRows>();
      out->set_rows(to_copy.rows());
      out->set_height(to_copy.height());
      framework::TensorCopy(to_copy.value(), to_copy.place(), dev_ctx,
                            out->mutable_value());
    } else if (selected_x->IsType<framework::LoDTensorArray>()) {
      auto &to_copy = selected_x->Get<framework::LoDTensorArray>();
      auto *out =
          scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensorArray>();
      out->resize(to_copy.size());
      for (size_t i = 0; i < to_copy.size(); ++i) {
        framework::LoDTensor &out_tensor = (*out)[i];
        framework::TensorCopy(to_copy[i], to_copy[i].place(), dev_ctx,
                              &out_tensor);
        out_tensor.set_lod(to_copy[i].lod());
      }
    } else {
      PADDLE_THROW("Type %s of %s is not supported in MergeSelectedTensorOp",
                   framework::ToTypeName(selected_x->Type()),
                   x_names[output_branch]);
    }
  }

  // Returns which branch to output in MergeSelectedTensorOp
  int GetOutputBranch(const framework::LoDTensor &mask,
                      const platform::DeviceContext &dev_ctx) const {
    PADDLE_ENFORCE_EQ(mask.numel(), 1,
                      "Mask in MergeSelectedTensorOp must have numel 1.");
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
        "Mask in MergeSelectedTensorOp. Please compile WITH_GPU option");
#endif
    return cpu_mask->data<int>()[0];
  }
};

class MergeSelectedTensorOpProtoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input LoDTensors or LoDTensorArray or SelectedRows. All "
             "input must have same var_type")
        .AsDuplicable();
    AddInput("Mask",
             "A integer tensor with numel 1 specifying which input to output");
    AddOutput("Out", "The merged output. The type of output must be same as X");
    // TODO(huihuangzheng): decide whether to add support for lod level
    // Because this op is blocking whole control flow. I am implementing MVP
    // (minimal value product) here.
    AddComment(R"DOC(
Merge branches of LoDTensor into a single Output with a mask interger
specifying the output branch
)DOC");
  }
};

class MergeSelectedTensorInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "MergeSelectedTensorOp must has input X.");
    PADDLE_ENFORCE(context->HasInput("Mask"),
                   "MergeLoDTensorOp must has input Mask.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "MergeLoDTensorOp must has output Out.");
  }
};

template <typename T>
class MergeSelectedTensorGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("split_selected_tensor");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetInput("Mask", this->Input("Mask"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    merge_selected_tensor, ops::MergeSelectedTensorOp,
    ops::MergeSelectedTensorOpProtoMaker, ops::MergeSelectedTensorInferShape,
    ops::MergeSelectedTensorGradMaker<paddle::framework::OpDesc>,
    ops::MergeSelectedTensorGradMaker<paddle::imperative::OpBase>);
