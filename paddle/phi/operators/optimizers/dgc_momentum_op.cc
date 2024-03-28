// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class DGCMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (var_name == "current_step" || var_name == "nranks") {
      VLOG(10) << "var_name:" << var_name << " need not to transform";
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }

    return framework::OperatorWithKernel::GetKernelTypeForVar(
        var_name, tensor, expected_kernel_type);
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class DGCMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input learning rate");
    AddInput("MasterParam", "FP32 master weight for AMP.").AsDispensable();
    AddInput("current_step", "(Tensor) Current step.");
    AddInput("nranks", "(Tensor) The number of trainers.");

    AddOutput("ParamOut",
              "(phi::DenseTensor) This output is updated parameter. "
              "It shared memory with Input(Param).");
    AddOutput("VelocityOut",
              "(phi::DenseTensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).");
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable();
    AddOutput("Grad_out", "(Tensor) Output grad gradient");

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<bool>("use_nesterov",
                  "(bool, default false) "
                  "Use Nesterov Momentum")
        .SetDefault(false);
    AddAttr<std::string>("regularization_method",
                         "(string) regularization_method, right now only "
                         "support l2decay or none")
        .SetDefault("");
    AddAttr<float>("regularization_coeff", "(float) regularization_coeff")
        .SetDefault(0.0f);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    AddAttr<float>(
        "rescale_grad",
        "(float, default 1.0) Multiply the gradient with `rescale_grad`"
        "before updating. Often choose to be `1.0/batch_size`.")
        .SetDefault(1.0f);
    AddAttr<float>("rampup_begin_step",
                   "(float, -1.0)"
                   "The period when begin DGC.")
        .SetDefault(-1.0);

    AddComment(R"DOC(
DGC Momentum Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(dgc_momentum,
                            DGCMomentumInferShapeFunctor,
                            PD_INFER_META(phi::DGCMomentumInferMeta));

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dgc_momentum,
                             ops::DGCMomentumOp,
                             ops::DGCMomentumOpMaker,
                             DGCMomentumInferShapeFunctor);
