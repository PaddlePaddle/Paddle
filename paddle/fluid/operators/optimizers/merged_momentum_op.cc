// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class MergedMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto param_dtype =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(param_dtype, ctx.GetPlace());
  }
};

class MergedMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter that has to be updated")
        .AsDuplicable();
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter")
        .AsDuplicable();
    AddInput("Velocity",
             "(Tensor, default Tensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated")
        .AsDuplicable();
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "Input learning rate")
        .AsDuplicable();
    AddInput("MasterParam", "FP32 master weight for AMP.")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("ParamOut",
              "(Tensor) This output is updated parameter. "
              "It shared memory with Input(Param).")
        .AsDuplicable();
    AddOutput("VelocityOut",
              "(Tensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).")
        .AsDuplicable();
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable()
        .AsDuplicable();
    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<bool>("use_nesterov",
                  "(bool, default false) "
                  "Use Nesterov Momentum or not.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>(
        "regularization_method",
        "(string) regularization_method, right now only "
        "support l2decay or none")
        .SetDefault({});
    AddAttr<std::vector<float>>("regularization_coeff",
                                "(float) regularization_coeff")
        .SetDefault({});
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    AddAttr<float>(
        "rescale_grad",
        "(float, default 1.0) Multiply the gradient with `rescale_grad`"
        "before updating. Often choose to be `1.0/batch_size`.")
        .SetDefault(1.0f);
    AddComment(R"DOC(Merged Momentum Optimizer.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(merged_momentum,
                            MergedMomentumInferShapeFunctor,
                            PD_INFER_META(phi::MergedMomentumInferMeta));

REGISTER_OP_WITHOUT_GRADIENT(merged_momentum,
                             ops::MergedMomentumOp,
                             ops::MergedMomentumOpMaker,
                             MergedMomentumInferShapeFunctor);
