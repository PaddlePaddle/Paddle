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

#include "paddle/fluid/operators/optimizers/sgd_op.h"

#include <string>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Param");

#ifdef PADDLE_WITH_MKLDNN
    using dnnl::memory;
    if (this->CanMKLDNNBeUsed(ctx, data_type)) {
      const auto *param_var = ctx.InputVar("Param");
      const auto *grad_var = ctx.InputVar("Grad");

      // supported cases
      bool dense_param_sparse_grad =
          param_var->IsType<framework::LoDTensor>() &&
          grad_var->IsType<phi::SelectedRows>();
      bool dense_param_and_grad = param_var->IsType<framework::LoDTensor>() &&
                                  grad_var->IsType<framework::LoDTensor>();

      if (dense_param_sparse_grad || dense_param_and_grad)
        return framework::OpKernelType(data_type,
                                       ctx.GetPlace(),
                                       framework::DataLayout::kMKLDNN,
                                       framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(data_type, ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (var_name == "LearningRate") {
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()),
          tensor.place(),
          tensor.layout());
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class SGDOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto in_var_type = ctx->GetInputType("Param");
    PADDLE_ENFORCE_EQ(in_var_type == framework::proto::VarType::SELECTED_ROWS ||
                          in_var_type == framework::proto::VarType::LOD_TENSOR,
                      true,
                      platform::errors::InvalidArgument(
                          "The input Var's type should be LoDtensor or "
                          "SelectedRows, but the received type is %s",
                          in_var_type));

    ctx->SetOutputType("ParamOut", in_var_type, framework::ALL_ELEMENTS);
  }
};

class SGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor or SelectedRows) Input parameter");
    AddInput("LearningRate", "(Tensor) Learning rate of SGD");
    AddInput("Grad", "(Tensor or SelectedRows) Input gradient");
    AddInput("MasterParam", "FP32 master weight for AMP.").AsDispensable();
    AddOutput("ParamOut",
              "(Tensor or SelectedRows, same with Param) "
              "Output parameter, should share the same memory with Param");
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable();

    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);

    AddComment(R"DOC(

SGD operator

This operator implements one step of the stochastic gradient descent algorithm.

$$param\_out = param - learning\_rate * grad$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(sgd,
                            SGDInferShapeFunctor,
                            PD_INFER_META(phi::SgdInferMeta));
REGISTER_OPERATOR(
    sgd,
    ops::SGDOp,
    ops::SGDOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::SGDOpInferVarType,
    SGDInferShapeFunctor);
