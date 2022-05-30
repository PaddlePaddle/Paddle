//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/pull_sparse_op.h"
#include <string>

namespace paddle {
namespace operators {

class PullSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("Ids").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Input(Ids) of PullSparseOp can not be null"));
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Output(Out) of PullSparseOp can not be null"));

    auto hidden_size =
        static_cast<uint32_t>(ctx->Attrs().Get<int>("EmbeddingDim"));
    auto all_ids_dim = ctx->GetInputsDim("Ids");
    const size_t n_ids = all_ids_dim.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(n_ids);
    for (size_t i = 0; i < n_ids; ++i) {
      const auto ids_dims = all_ids_dim[i];
      int ids_rank = ids_dims.size();
      PADDLE_ENFORCE_EQ(ids_dims[ids_rank - 1], 1,
                        platform::errors::InvalidArgument(
                            "Shape error in %lu id, the last dimension of "
                            " the 'Ids' tensor must be 1.",
                            i));
      auto out_dim = phi::vectorize(phi::slice_ddim(ids_dims, 0, ids_rank - 1));
      out_dim.push_back(hidden_size);
      outs_dims[i] = phi::make_ddim(out_dim);
    }
    ctx->SetOutputsDim("Out", outs_dims);
    for (size_t i = 0; i < n_ids; ++i) {
      ctx->ShareLoD("Ids", "Out", i, i);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class PullSparseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "Input tensors with type int64 contains "
             "the ids to be looked up in PSLib. "
             "The last dimension size must be 1.")
        .AsDuplicable();
    AddInput("W", "The lookup table tensors.").AsDuplicable();
    AddOutput("Out", "The lookup results tensors.").AsDuplicable();
    AddAttr<int>("EmbeddingDim", "(int, the embedding hidden size")
        .SetDefault(11);
    AddAttr<int>("TableId", "(int, the table id of this embedding")
        .SetDefault(0);
    AddAttr<std::string>("AccessorClass", "(string, the class name of accessor")
        .SetDefault("");
    AddAttr<std::string>("CtrLabelName", "(string, ctr label name")
        .SetDefault("");
    AddAttr<int>("PaddingId", "(int, the padding id of this embedding")
        .SetDefault(0);
    AddAttr<bool>("ScaleSparseGrad",
                  "(bool, whether scale sparse gradient with batch size")
        .SetDefault(true);
    AddAttr<std::vector<std::string>>("InputNames", "(vector, slot names")
        .SetDefault(std::vector<std::string>());
    AddAttr<bool>("is_distributed", "(bool, it must be true").SetDefault(true);
    AddComment(R"DOC(
Pull Sparse Operator.

This operator is used to perform lookups on the PSLib
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

template <typename T>
class PushSparseOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("push_sparse");
    retv->SetInput("Ids", this->Input("Ids"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("W", this->Input("W"));
    retv->SetOutput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
  }
};

class PushSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(pull_sparse, ops::PullSparseOp, ops::PullSparseOpMaker,
                  ops::PushSparseOpMaker<paddle::framework::OpDesc>,
                  ops::PushSparseOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(push_sparse, ops::PushSparseOp);
REGISTER_OP_CPU_KERNEL(pull_sparse, ops::PullSparseCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(push_sparse, ops::PushSparseCPUKernel<float>)
