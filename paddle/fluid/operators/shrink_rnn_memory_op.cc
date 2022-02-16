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
#include "paddle/pten/kernels/funcs/math_function.h"

#include "paddle/pten/core/lod_utils.h"

namespace paddle {
namespace framework {
class OpDesc;
class Scope;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class ShrinkRNNMemoryOp : public ArrayOp {
 public:
  ShrinkRNNMemoryOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *x_var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(x_var,
                            platform::errors::NotFound(
                                "Input(X) of ShrinkRNNMemoryOp is not found."));
    auto &x_tensor = x_var->Get<framework::LoDTensor>();
    size_t offset = this->GetOffset(scope, place);
    auto *rank_table_var = scope.FindVar(Input("RankTable"));
    PADDLE_ENFORCE_NOT_NULL(
        rank_table_var,
        platform::errors::NotFound(
            "Input(RankTable) of ShrinkRNNMemoryOp is not found."));
    auto &rank_table = rank_table_var->Get<framework::LoDRankTable>();

    auto &rank_items = rank_table.items();
    int dst_num_rows =
        std::lower_bound(rank_items.begin(), rank_items.end(), offset,
                         [](const framework::LoDRankTable::TableItem &a,
                            size_t b) { return a.length > b; }) -
        rank_items.begin();

    auto *out_var = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "Output(Out) of ShrinkRNNMemoryOp is not found."));
    auto &out_tensor = *out_var->GetMutable<framework::LoDTensor>();

    size_t height = dst_num_rows;

    // do shrink for the top level LoD
    if (x_tensor.lod().size() > 0 &&
        x_tensor.lod()[0].size() > static_cast<size_t>(dst_num_rows)) {
      auto lod_offset = framework::GetSubLoDAndAbsoluteOffset(x_tensor.lod(), 0,
                                                              dst_num_rows, 0);
      height = lod_offset.second.second;
      auto out_lod = out_tensor.mutable_lod();
      pten::AppendLoD(out_lod, lod_offset.first);
    }

    if (dst_num_rows != 0) {
      out_tensor.mutable_data(place, x_tensor.dtype());
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      framework::TensorCopy(x_tensor.Slice(0, height), place, *dev_ctx,
                            &out_tensor);
    }
  }
};

class ShrinkRNNMemoryOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The RNN step memory to be shrank.");
    AddInput("RankTable", "(LoDRankTable) The lod_rank_table of dynamic RNN.");
    AddInput("I",
             "(LoDTensor) The step index. The RNN step memory 'X' will be "
             "shrank to match the size of the input of the index'th step.");
    AddOutput("Out", "(LoDTensor) The shrank RNN step memory.");
    AddComment(R"DOC(
This operator is used to shrink output batch of memory defined in dynamic RNN.

Dynamic RNN is able to handle variable-length sequences, in which, sequences in
a mini-batch are sorted by their lengths first. After that, the longest sequence
becomes the first one in the sorted batch, followed by the second longest, the
third longest, and so on. Dynamic RNN then slices a batch input timestep by
timestep from the sorted input. Once any sequence in the input batch reaches its
end, memory defined in dynamicRNN has to shrink its outputs to adapt to the input
batch size for the next time step.
)DOC");
  }
};

class ShrinkRNNMemoryInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "ShrinkRNNMemory");
    OP_INOUT_CHECK(context->HasInput("I"), "Input", "I", "ShrinkRNNMemory");
    OP_INOUT_CHECK(context->HasInput("RankTable"), "Input", "RankTable",
                   "ShrinkRNNMemory");
    context->SetOutputDim("Out", context->GetInputDim("X"));
    // For runtime, output's lod is computed according to input's lod, but
    // remove the finished sequence. It is set in detail kernel implementation.
    if (!context->IsRuntime()) {
      context->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class ShrinkRNNMemoryGradOp : public ArrayOp {
 public:
  ShrinkRNNMemoryGradOp(const std::string &type,
                        const framework::VariableNameMap &inputs,
                        const framework::VariableNameMap &outputs,
                        const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *dout_var = scope.FindVar(Input(framework::GradVarName("Out")));
    auto *dx_var = scope.FindVar(Output(framework::GradVarName("X")));
    PADDLE_ENFORCE_NOT_NULL(
        dx_var, platform::errors::NotFound(
                    "Input(X@GRAD) of ShrinkRNNMemoryGradOp is not found."));
    auto *x_var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(
        x_var, platform::errors::NotFound(
                   "Input(x) of ShrinkRNNMemoryGradOp is not found."));
    auto &x_tensor = x_var->Get<framework::LoDTensor>();
    auto &dx_tensor = *dx_var->GetMutable<framework::LoDTensor>();
    dx_tensor.Resize(x_tensor.dims());
    dx_tensor.mutable_data(x_tensor.place(), x_tensor.dtype());

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    if (dout_var == nullptr) {  // dx_tensor fill zero
      pten::funcs::set_constant(dev_ctx, &dx_tensor, 0.0f);
    } else {
      auto &dout_tensor = dout_var->Get<framework::LoDTensor>();
      auto height = dout_tensor.dims()[0];
      auto slice = dx_tensor.Slice(0, static_cast<int>(height));
      framework::TensorCopy(dout_tensor, dout_tensor.place(), dev_ctx, &slice);
      if (dx_tensor.dims()[0] > height) {
        auto rest_tensor = dx_tensor.Slice(
            static_cast<int>(height), static_cast<int>(dx_tensor.dims()[0]));
        pten::funcs::set_constant(dev_ctx, &rest_tensor, 0.0f);
      }
    }
    dx_tensor.set_lod(x_tensor.lod());
  }
};

class ShrinkRNNMemoryGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "ShrinkRNNMemoryGrad");
    OP_INOUT_CHECK(context->HasOutput(framework::GradVarName("X")), "Output",
                   "X", "ShrinkRNNMemoryGrad");

    context->ShareDim("X", /*->*/ framework::GradVarName("X"));
    context->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }
};

template <typename T>
class ShrinkRNNGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("shrink_rnn_memory_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(shrink_rnn_memory, ops::ShrinkRNNMemoryOp,
                  ops::ShrinkRNNMemoryInferShape,
                  ops::ShrinkRNNMemoryOpProtoMaker,
                  ops::ShrinkRNNGradOpMaker<paddle::framework::OpDesc>,
                  ops::ShrinkRNNGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(shrink_rnn_memory_grad, ops::ShrinkRNNMemoryGradOp,
                  ops::ShrinkRNNMemoryGradInferShape);
