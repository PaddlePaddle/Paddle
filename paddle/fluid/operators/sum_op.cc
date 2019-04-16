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

#include "paddle/fluid/operators/sum_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

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
            framework::proto::VarType::LOD_TENSOR_ARRAY) {
      return;  // skip runtime infershape when is tensor array;
    }

    auto x_var_types = ctx->GetInputsVarType("X");
    auto x_dims = ctx->GetInputsDim("X");

    size_t N = x_dims.size();
    PADDLE_ENFORCE_GT(N, 0, "Input tensors count should > 0.");
    if (N == 1) {
      VLOG(3) << "Warning: sum have only one input, may waste memory";
    }

    framework::DDim in_dim({0});
    for (size_t i = 0; i < x_dims.size(); ++i) {
      auto& x_dim = x_dims[i];
      // x_dim.size() == 1 means the real dim of selected rows is [0]
      if (x_var_types[i] == framework::proto::VarType::SELECTED_ROWS &&
          x_dim.size() == 1) {
        continue;
      }
      if (framework::product(x_dim) == 0) {
        continue;
      }
      if (framework::product(in_dim) == 0) {
        in_dim = x_dim;
      } else {
        if (ctx->IsRuntime()) {
          PADDLE_ENFORCE_EQ(in_dim, x_dim,
                            "Input tensors must have same shape");
        } else {
          PADDLE_ENFORCE_EQ(in_dim.size(), x_dim.size(),
                            "Input tensors must have same shape size");
          // if in_dim or x_dim has -1, not check equal
          for (int i = 0; i < x_dim.size(); ++i) {
            if (x_dim[i] == -1 || in_dim[i] == -1) {
              continue;
            }
            PADDLE_ENFORCE_EQ(in_dim[i], x_dim[i],
                              "Input tensors must have same shape if not -1");
          }
        }
      }
    }
    ctx->SetOutputDim("Out", in_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto x_vars = ctx.MultiInputVar("X");
    auto x_vars_name = ctx.Inputs("X");

    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout{framework::DataLayout::kAnyLayout};

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    if (x_vars[0]->IsType<framework::LoDTensor>()) {
      int dtype = -1;
      for (size_t idx = 0; idx < x_vars.size(); ++idx) {
        PADDLE_ENFORCE(x_vars[idx] != nullptr,
                       "Input var[%s] should not be nullptr", x_vars_name[idx]);
        auto tensor =
            framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_vars[idx]);
        if (tensor->numel() == 0) {
          continue;
        }
        if (dtype == -1) {
          dtype = tensor->type();
        } else {
          PADDLE_ENFORCE_EQ(dtype, tensor->type());
        }
      }
      PADDLE_ENFORCE_NE(dtype, -1,
                        "Sum operator should have at least one tensor");

      return framework::OpKernelType(
          static_cast<framework::proto::VarType::Type>(dtype), ctx.GetPlace(),
          layout, library);
    } else if (x_vars[0]->IsType<framework::SelectedRows>()) {
      for (auto& var : x_vars) {
        auto& value = var->Get<framework::SelectedRows>().value();
        if (value.IsInitialized()) {
          return framework::OpKernelType(value.type(), ctx.device_context(),
                                         layout, library);
        }
      }
      // if input sparse vars are not initialized, use an default kernel type.
      return framework::OpKernelType(framework::proto::VarType::FP32,
                                     ctx.device_context(), layout, library);
    } else if (x_vars[0]->IsType<framework::LoDTensorArray>()) {
      for (auto& x_var : x_vars) {
        auto& array = x_var->Get<framework::LoDTensorArray>();
        for (auto& each : array) {
          if (each.numel() != 0) {
            return framework::OpKernelType(each.type(), ctx.device_context(),
                                           layout, library);
          }
        }
      }
      PADDLE_THROW("Cannot find the input data type by all input data");
    }
    PADDLE_THROW("Unexpected branch. Input type is %s",
                 framework::ToTypeName(x_vars[0]->Type()));
  }
};

class SumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(vector<Tensor>) The input tensors of sum operator.")
        .AsDuplicable();
    AddOutput("Out", "(Tensor) The output tensor of sum operator.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
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
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto& inputs = ctx->Input("X");
    auto var_type = framework::proto::VarType::SELECTED_ROWS;
    for (auto& name : ctx->Input("X")) {
      VLOG(10) << name << " " << ctx->GetType(name);
    }

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [ctx](const std::string& name) {
          return ctx->GetType(name) == framework::proto::VarType::LOD_TENSOR;
        });

    auto is_tensor_array = [ctx](const std::string& name) {
      return ctx->GetType(name) == framework::proto::VarType::LOD_TENSOR_ARRAY;
    };

    bool any_input_is_tensor_array =
        std::any_of(inputs.begin(), inputs.end(), is_tensor_array);
    bool all_inputs_are_tensor_array =
        std::all_of(inputs.begin(), inputs.end(), is_tensor_array);

    if (any_input_is_tensor_array) {
      if (!all_inputs_are_tensor_array) {
        std::ostringstream os;
        for (auto& each : inputs) {
          os << "    " << each << " type is " << ctx->GetType(each) << "\n";
        }
        PADDLE_ENFORCE(all_inputs_are_tensor_array,
                       "Not all inputs are tensor array:\n%s", os.str());
      }
      var_type = framework::proto::VarType::LOD_TENSOR_ARRAY;
    } else if (any_input_is_lod_tensor) {
      var_type = framework::proto::VarType::LOD_TENSOR;
    }

    auto out_var_name = ctx->Output("Out").front();
    ctx->SetType(out_var_name, var_type);
    ctx->SetDataType(out_var_name, ctx->GetDataType(inputs.front()));
  }
};

class SumGradMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    auto x_grads = InputGrad("X", false);
    std::vector<std::unique_ptr<framework::OpDesc>> grad_ops;
    grad_ops.reserve(x_grads.size());
    auto og = OutputGrad("Out");
    std::transform(x_grads.begin(), x_grads.end(), std::back_inserter(grad_ops),
                   [&og](const std::string& x_grad) {
                     auto* grad_op = new framework::OpDesc();
                     grad_op->SetType("scale");
                     grad_op->SetInput("X", og);
                     grad_op->SetOutput("Out", {x_grad});
                     grad_op->SetAttr("scale", 1.0f);
                     return std::unique_ptr<framework::OpDesc>(grad_op);
                   });
    return grad_ops;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sum, ops::SumOp, ops::SumOpMaker, ops::SumGradMaker,
                  ops::SumOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(
    sum, ops::SumKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SumKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SumKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SumKernel<paddle::platform::CPUDeviceContext, int64_t>);
