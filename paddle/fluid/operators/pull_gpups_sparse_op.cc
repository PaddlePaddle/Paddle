//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/pull_gpups_sparse_op.h"

namespace paddle {
namespace operators {

class PullGpuPSSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(
        ctx->Inputs("Ids").size(), 1UL,
        platform::errors::InvalidArgument(
            "Inputs(Ids) of PullGpuPSSparseOp should not be empty."));
    PADDLE_ENFORCE_GE(
        ctx->Outputs("Out").size(), 1UL,
        platform::errors::InvalidArgument(
            "Outputs(Out) of PullGpuPSSparseOp should not be empty."));
    auto embedding_size_vec = ctx->Attrs().Get<std::vector<int>>("size");
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("Ids").size(), embedding_size_vec.size(),
        platform::errors::InvalidArgument("The ids size: %lu must be equal to "
                                          "the length of embedding size: %lu.",
                                          ctx->Inputs("Ids").size(),
                                          embedding_size_vec.size()));
    auto all_ids_dim = ctx->GetInputsDim("Ids");
    const size_t n_ids = all_ids_dim.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(n_ids);
    for (size_t i = 0; i < n_ids; ++i) {
      int embedding_size = embedding_size_vec[i];
      const auto ids_dims = all_ids_dim[i];
      int ids_rank = ids_dims.size();
      PADDLE_ENFORCE_EQ(ids_dims[ids_rank - 1], 1,
                        platform::errors::InvalidArgument(
                            "Shape error in %lu id, the last dimension of the "
                            "'Ids' tensor must be 1.",
                            i));
      auto out_dim = framework::vectorize(
          framework::slice_ddim(ids_dims, 0, ids_rank - 1));
      out_dim.push_back(embedding_size);
      outs_dims[i] = framework::make_ddim(out_dim);
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

class PullGpuPSSparseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.")
        .AsDispensable();
    AddInput("Ids",
             "Input tensors with type int32 or int64 "
             "contains the ids to be looked up in GpuPS. "
             "The last dimension size must be 1.")
        .AsDuplicable();
    AddOutput("Out", "The lookup results tensors.").AsDuplicable();
    AddAttr<std::vector<int>>(
        "size", "(vector<int>, the embedding size of corresponding slot")
        .SetDefault(std::vector<int>());
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<bool>("is_distributed",
                  "(boolean, default false) distributed lookup table.")
        .SetDefault(false);
    AddComment(R"DOC(
Pull GpuPS Sparse Operator.

This operator is used to perform lookups on the GpuPS,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

template <typename T>
class PushGpuPSSparseOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("push_gpups_sparse");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
  }
};

class PushGpuPSSparseOp : public framework::OperatorWithKernel {
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
REGISTER_OPERATOR(pull_gpups_sparse, ops::PullGpuPSSparseOp,
                  ops::PullGpuPSSparseOpMaker,
                  ops::PushGpuPSSparseOpMaker<paddle::framework::OpDesc>,
                  ops::PushGpuPSSparseOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(push_gpups_sparse, ops::PushGpuPSSparseOp);
REGISTER_OP_CPU_KERNEL(pull_gpups_sparse, ops::PullGpuPSSparseCPUKernel<float>,
                       ops::PullGpuPSSparseCPUKernel<double>)
REGISTER_OP_CPU_KERNEL(push_gpups_sparse, ops::PushGpuPSSparseCPUKernel<float>,
                       ops::PushGpuPSSparseCPUKernel<double>)