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

#include "paddle/fluid/operators/lookup_table_v2_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class LookupTableV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      "Input(W) of LookupTableV2Op should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Ids"), true,
                      "Input(Ids) of LookupTableV2Op should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of LookupTableV2Op should not be null.");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    int ids_rank = ids_dims.size();
    VLOG(5) << "ids rank is " << ids_rank << std::endl;
    PADDLE_ENFORCE_EQ(table_dims.size(), 2);

    auto output_dims = framework::vectorize(ids_dims);
    output_dims.push_back(table_dims[1]);
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));

    if (ctx->GetOutputsVarType("Out")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("Ids", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int32 or int64 "
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

    // for parameter prefetch
    AddAttr<bool>("remote_prefetch", "").SetDefault(false);
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));
    AddAttr<std::vector<std::string>>(
        "epmap",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(
        "table_names",
        "(string vector, the splited table names that will be fetched from "
        "parameter server)"
        "in the order of input variables for mapping")
        .SetDefault({});

    AddComment(R"DOC(
Lookup Table V2 Operator.

This operator is used to perform lookups on the parameter W,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(LookupTableV2GradOpNoBuffer, "W");

class LookupTableV2GradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());

    op->SetType("lookup_table_v2_grad");

    op->SetInput("W", Input("W"));
    op->SetInput("Ids", Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("W"), InputGrad("W"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

class LookupTableV2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("Out")));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableV2OpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto out_var_name = ctx->Output(framework::GradVarName("W")).front();
    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "lookup_table_v2_grad op " << framework::GradVarName("W")
              << " is set to SelectedRows";
      ctx->SetType(out_var_name, framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "lookup_table_v2_grad op " << framework::GradVarName("W")
              << " is set to LoDTensor";
      ctx->SetType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetDataType(out_var_name, ctx->GetDataType(ctx->Input("W")[0]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lookup_table_v2, ops::LookupTableV2Op,
                  ops::LookupTableV2OpMaker, ops::LookupTableV2GradOpDescMaker);

REGISTER_OPERATOR(lookup_table_v2_grad, ops::LookupTableV2OpGrad,
                  ops::LookupTableV2GradOpNoBuffer,
                  ops::LookupTableV2OpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(lookup_table_v2, ops::LookupTableV2Kernel<float>,
                       ops::LookupTableV2Kernel<double>);
REGISTER_OP_CPU_KERNEL(lookup_table_v2_grad,
                       ops::LookupTableV2GradKernel<float>,
                       ops::LookupTableV2GradKernel<double>);
