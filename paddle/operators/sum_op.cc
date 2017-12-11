/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/sum_op.h"
#include <vector>
#include "paddle/framework/var_type_inference.h"
#include "paddle/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"), "Inputs(X) should not be null");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SumOp should not be null.");
    if (ctx->IsRuntime() &&
        ctx->GetOutputsVarType("Out")[0] ==
            framework::VarDesc::LOD_TENSOR_ARRAY) {
      return;  // skip runtime infershape when is tensor array;
    }

    auto x_dims = ctx->GetInputsDim("X");
    size_t N = x_dims.size();
    PADDLE_ENFORCE_GT(N, 1, "Input tensors count should > 1.");

    framework::DDim in_dim({0});
    for (auto& x_dim : x_dims) {
      if (framework::product(x_dim) == 0) {
        continue;
      }
      if (framework::product(in_dim) == 0) {
        in_dim = x_dim;
      } else {
        PADDLE_ENFORCE_EQ(in_dim, x_dim, "Input tensors must have same shape");
      }
    }
    ctx->SetOutputDim("Out", in_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto x_vars = ctx.MultiInputVar("X");
    if (x_vars[0]->IsType<framework::LoDTensor>()) {
      int dtype = -1;
      for (auto& x_var : x_vars) {
        auto& lod_tensor = x_var->Get<framework::LoDTensor>();
        if (lod_tensor.numel() == 0) {
          continue;
        }
        if (dtype == -1) {
          dtype = framework::ToDataType(lod_tensor.type());
        } else {
          PADDLE_ENFORCE_EQ(dtype, framework::ToDataType(lod_tensor.type()));
        }
      }
      PADDLE_ENFORCE_NE(dtype, -1,
                        "Sum operator should have at least one tensor");

      return framework::OpKernelType(static_cast<framework::DataType>(dtype),
                                     ctx.device_context());
    } else if (x_vars[0]->IsType<framework::SelectedRows>()) {
      return framework::OpKernelType(
          framework::ToDataType(
              x_vars[0]->Get<framework::SelectedRows>().value().type()),
          ctx.device_context());
    } else if (x_vars[0]->IsType<framework::LoDTensorArray>()) {
      for (auto& x_var : x_vars) {
        auto& array = x_var->Get<framework::LoDTensorArray>();
        for (auto& each : array) {
          if (each.numel() != 0) {
            return framework::OpKernelType(framework::ToDataType(each.type()),
                                           ctx.device_context());
          }
        }
      }
      PADDLE_THROW("Cannot find the input data type by all input data");
    }
    PADDLE_THROW("Unexpected branch. Input type is %s",
                 x_vars[0]->Type().name());
  }
};

class SumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SumOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(vector<Tensor>) The input tensors of sum operator.")
        .AsDuplicable();
    AddOutput("Out", "(Tensor) The output tensor of sum operator.");
    AddComment(R"DOC(
Sum operator.

This operators sums the input tensors. All the inputs can carry the 
LoD (Level of Details) information. However, the output only shares 
the LoD information with the first input.
)DOC");
  }
};

class SumOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {
    auto& inputs = op_desc.Input("X");
    auto var_type = framework::VarDesc::SELECTED_ROWS;

    for (auto& name : op_desc.Input("X")) {
      VLOG(10) << name << " "
               << block->FindRecursiveOrCreateVar(name)->GetType();
    }

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [block](const std::string& name) {
          return block->FindRecursiveOrCreateVar(name)->GetType() ==
                 framework::VarDesc::LOD_TENSOR;
        });

    auto is_tensor_array = [block](const std::string& name) {
      return detail::Ref(block->FindRecursiveOrCreateVar(name)).GetType() ==
             framework::VarDesc::LOD_TENSOR_ARRAY;
    };

    bool any_input_is_tensor_array =
        std::any_of(inputs.begin(), inputs.end(), is_tensor_array);
    bool all_inputs_are_tensor_array =
        std::all_of(inputs.begin(), inputs.end(), is_tensor_array);

    if (any_input_is_tensor_array) {
      if (!all_inputs_are_tensor_array) {
        std::ostringstream os;
        for (auto& each : inputs) {
          os << "    " << each << " type is "
             << detail::Ref(block->FindRecursiveOrCreateVar(each)).GetType()
             << "\n";
        }
        PADDLE_ENFORCE(all_inputs_are_tensor_array,
                       "Not all inputs are tensor array:\n%s", os.str());
      }
      var_type = framework::VarDesc::LOD_TENSOR_ARRAY;
    } else if (any_input_is_lod_tensor) {
      var_type = framework::VarDesc::LOD_TENSOR;
    }

    auto out_var_name = op_desc.Output("Out").front();
    auto& out_var = detail::Ref(block->FindRecursiveOrCreateVar(out_var_name));
    out_var.SetType(var_type);
    auto& in_var = detail::Ref(block->FindVarRecursive(inputs.front()));
    out_var.SetDataType(in_var.GetDataType());
  }
};

class SumGradMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDescBind>> operator()()
      const override {
    auto x_grads = InputGrad("X");
    std::vector<std::unique_ptr<framework::OpDescBind>> grad_ops;
    grad_ops.reserve(x_grads.size());
    auto og = OutputGrad("Out");
    std::transform(x_grads.begin(), x_grads.end(), std::back_inserter(grad_ops),
                   [&og](const std::string& x_grad) {
                     auto* grad_op = new framework::OpDescBind();
                     grad_op->SetType("scale");
                     grad_op->SetInput("X", og);
                     grad_op->SetOutput("Out", {x_grad});
                     grad_op->SetAttr("scale", 1.0f);
                     return std::unique_ptr<framework::OpDescBind>(grad_op);
                   });
    return grad_ops;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sum, ops::SumOp, ops::SumOpMaker, ops::SumGradMaker,
                  ops::SumOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(sum, ops::SumKernel<paddle::platform::CPUPlace, float>,
                       ops::SumKernel<paddle::platform::CPUPlace, double>,
                       ops::SumKernel<paddle::platform::CPUPlace, int>,
                       ops::SumKernel<paddle::platform::CPUPlace, int64_t>);
