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
#include <unordered_map>
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
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"), true,
                      "Inputs(X) should not be null");

    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of SumOp should not be null.");
    if (ctx->IsRuntime() &&
        ctx->GetOutputsVarType("Out")[0] ==
            framework::proto::VarType::LOD_TENSOR_ARRAY) {
      return;  // skip runtime infershape when is tensor array;
    }

    auto x_var_types = ctx->GetInputsVarType("X");
    auto x_dims = ctx->GetInputsDim("X");

    auto N = x_dims.size();
    PADDLE_ENFORCE_GT(
        N, 0,
        "ShapeError: The input tensor X's dimensions of SumOp "
        "should be larger than 0. But received X's dimensions %d, "
        "X's shape = [%s].",
        N, &x_dims);
    if (N == 1) {
      VLOG(3) << "Warning: SumOp have only one input, may waste memory";
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
          PADDLE_ENFORCE_EQ(
              in_dim, x_dim,
              "ShapeError: The input tensor X of SumOp must have same shape."
              "But received X[0]'s shape = [%s], X[%d]'s shape = [%s].",
              in_dim, i, x_dim);
        } else {
          PADDLE_ENFORCE_EQ(
              in_dim.size(), x_dim.size(),
              "ShapeError: The input tensor X of SumOp must have same "
              "dimensions. But received X[0]'s dimensions = %d, X[0]'s shape = "
              "[%s], X[%d]'s dimensions = %d, X[%d]'s shape = [%s].",
              in_dim.size(), in_dim, i, x_dim.size(), i, x_dim);
          // if in_dim or x_dim has -1, not check equal
          for (int j = 0; j < x_dim.size(); ++j) {
            if (x_dim[j] == -1 || in_dim[j] == -1) {
              continue;
            }
            PADDLE_ENFORCE_EQ(
                in_dim[j], x_dim[j],
                "ShapeError: The input tensor X of SumOp must have same shape "
                "if not -1."
                "But received X[0]'s shape = [%s], X[%d]'s shape = [%s].",
                in_dim, i, x_dim);
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
    auto x_vars_name = ctx.InputNames("X");

    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout{framework::DataLayout::kAnyLayout};

    if (x_vars[0]->IsType<framework::LoDTensor>()) {
      int dtype = -1;
      for (size_t idx = 0; idx < x_vars.size(); ++idx) {
        PADDLE_ENFORCE_NOT_NULL(x_vars[idx],
                                "Input var[%s] should not be nullptr",
                                x_vars_name[idx]);
        auto tensor =
            framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_vars[idx]);
        if (tensor->numel() <= 0 || (!tensor->IsInitialized())) {
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

#ifdef PADDLE_WITH_MKLDNN
      if (library == framework::LibraryType::kPlain &&
          platform::CanMKLDNNBeUsed(ctx) &&
          static_cast<framework::proto::VarType::Type>(dtype) ==
              framework::proto::VarType::FP32 &&
          ctx.OutputVar("Out")->IsType<framework::LoDTensor>()) {
        if (std::all_of(x_vars.begin(), x_vars.end(),
                        [](const framework::Variable* v) {
                          return v->IsType<framework::LoDTensor>();
                        })) {
          return framework::OpKernelType(
              framework::proto::VarType::FP32, ctx.GetPlace(),
              framework::DataLayout::kMKLDNN, framework::LibraryType::kMKLDNN);
        }
      }
#endif

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
          if (each.numel() != 0 && each.IsInitialized()) {
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
    AddInput("X",
             "A Varaible list. The shape and data type of the list elements"
             "should be consistent. Variable can be multi-dimensional Tensor"
             "or LoDTensor, and data types can be: float32, float64, int32, "
             "int64.")
        .AsDuplicable();
    AddOutput("Out",
              "the sum of input :code:`x`. its shape and data types are "
              "consistent with :code:`x`.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(This OP is used to sum one or more Tensor or LoDTensor
                    of the input. If the input is LoDTensor, the output only
                    shares LoD information with the first input.)DOC");
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
        PADDLE_ENFORCE_EQ(all_inputs_are_tensor_array, true,
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

class SumGradDescMaker : public framework::GradOpDescMakerBase {
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

class SumGradOpBaseMaker : public imperative::GradOpBaseMakerBase {
 public:
  using imperative::GradOpBaseMakerBase::GradOpBaseMakerBase;

  std::vector<std::unique_ptr<imperative::OpBase>> operator()() const override {
    auto x_grads = InputGrad("X", false);
    std::vector<std::unique_ptr<imperative::OpBase>> grad_ops;
    grad_ops.reserve(x_grads.size());
    auto og = OutputGrad("Out");
    std::transform(x_grads.begin(), x_grads.end(), std::back_inserter(grad_ops),
                   [&og](const std::shared_ptr<imperative::VarBase>& x_grad) {
                     auto* grad_op = new imperative::OpBase();
                     grad_op->SetType("scale");
                     grad_op->SetInput("X", og);
                     grad_op->SetOutput("Out", {x_grad});
                     grad_op->SetAttr("scale", 1.0f);
                     return std::unique_ptr<imperative::OpBase>(grad_op);
                   });

    return grad_ops;
  }
};

DECLARE_INPLACE_OP_INFERER(SumInplace, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sum, ops::SumOp, ops::SumOpMaker, ops::SumGradDescMaker,
                  ops::SumGradOpBaseMaker, ops::SumOpVarTypeInference,
                  ops::SumInplace);

REGISTER_OP_CPU_KERNEL(
    sum, ops::SumKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SumKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SumKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SumKernel<paddle::platform::CPUDeviceContext, int64_t>);
