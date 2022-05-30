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

#include "paddle/fluid/operators/lookup_table_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/platform/bfloat16.h"

namespace paddle {
namespace operators {

class LookupTableOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "LookupTable");
    OP_INOUT_CHECK(ctx->HasInput("Ids"), "Input", "Ids", "LookupTable");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "LookupTable");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    int ids_rank = ids_dims.size();
    VLOG(5) << "ids rank is " << ids_rank << std::endl;
    PADDLE_ENFORCE_EQ(
        table_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ShapeError: The dimensions of the 'lookup table' must be 2. "
            "But received lookup table's dimensions = %d, "
            "lookup table's shape = [%s].",
            table_dims.size(), table_dims));
    PADDLE_ENFORCE_EQ(
        ids_dims[ids_rank - 1], 1,
        platform::errors::InvalidArgument(
            "ShapeError: The last dimensions of the 'Ids' tensor must be 1. "
            "But received Ids's last dimensions = %d, Ids's shape = [%s].",
            ids_dims[ids_rank - 1], ids_dims));

    auto output_dims =
        phi::vectorize(phi::slice_ddim(ids_dims, 0, ids_rank - 1));
    output_dims.push_back(table_dims[1]);
    ctx->SetOutputDim("Out", phi::make_ddim(output_dims));

    if (ctx->GetOutputsVarType("Out")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("Ids", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "W");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int64 "
             "contains the ids to be looked up in W. "
             "The last dimension size must be 1.");
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<bool>("is_distributed",
                  "(boolean, default false) distributed lookup table.")
        .SetDefault(false);
    AddAttr<int64_t>("padding_idx",
                     "(int64, default -1) "
                     "If the value is -1, it makes no effect to lookup. "
                     "Otherwise the given value indicates padding the output "
                     "with zeros whenever lookup encounters it in Ids.")
        .SetDefault(kNoPadding);

    // for parameter training config
    AddAttr<bool>("remote_prefetch",
                  "pull sparse params from parameters, this can only be used "
                  "in distributed training")
        .SetDefault(false);

    AddAttr<std::string>("entry_config",
                         "embedding sparse feature entry config, "
                         " probability entry / counting "
                         " this can only be used in distributed training"
                         "entry")
        .SetDefault("");

    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);

    AddAttr<std::string>("entry",
                         "(std::string, default "
                         ") for entry attribute.")
        .SetDefault("none");

    AddAttr<std::string>("table_class",
                         "(std::string, default "
                         ") for table_class.")
        .SetDefault("none");

    AddAttr<std::vector<std::string>>(
        "table_names",
        "(string vector, the split table names that will be fetched from "
        "parameter server)"
        "in the order of input variables for mapping")
        .SetDefault({});
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<bool>("grad_inplace",
                  "(boolean, default false) "
                  "If the grad op reuse the input's variable.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>(
        "epmap",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));
    AddComment(R"DOC(
Lookup Table Operator.

This operator is used to perform lookups on the parameter W,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(LookupTableGradOpNoBufferVarsInferer, "W");

template <typename T>
class LookupTableGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("lookup_table_grad");

    op->SetInput("W", this->Input("W"));
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));

    op->SetAttrMap(this->Attrs());
  }
};

class LookupTableOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto out_var_name = framework::GradVarName("W");
    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = BOOST_GET(bool, attr);
    if (is_sparse) {
      VLOG(3) << "lookup_table_grad op " << framework::GradVarName("W")
              << " is set to SelectedRows";
      ctx->SetOutputType(out_var_name,
                         framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "lookup_table_grad op " << framework::GradVarName("W")
              << " is set to LoDTensor";
      ctx->SetOutputType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetOutputDataType(out_var_name, ctx->GetInputDataType("W"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lookup_table, ops::LookupTableOp, ops::LookupTableOpMaker,
                  ops::LookupTableGradOpMaker<paddle::framework::OpDesc>,
                  ops::LookupTableGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(lookup_table_grad, ops::LookupTableOpGrad,
                  ops::LookupTableGradOpNoBufferVarsInferer,
                  ops::LookupTableOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(lookup_table, ops::LookupTableKernel<float>,
                       ops::LookupTableKernel<double>,
                       ops::LookupTableKernel<int8_t>,
                       ops::LookupTableKernel<int16_t>,
                       ops::LookupTableKernel<paddle::platform::bfloat16>);
REGISTER_OP_CPU_KERNEL(lookup_table_grad, ops::LookupTableGradKernel<float>,
                       ops::LookupTableGradKernel<double>,
                       ops::LookupTableGradKernel<paddle::platform::bfloat16>);

/* ==========================  register checkpoint ===========================*/

REGISTER_OP_VERSION(lookup_table)
    .AddCheckpoint(
        R"ROC(
      Upgrade lookup_table add 1 attribute [entry_config].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "entry_config",
            "(std::string) embedding sparse feature entry config.", ""));

/* ========================================================================== */
