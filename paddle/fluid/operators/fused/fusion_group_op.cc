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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class FusionGroupOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, phi::GPUPlace(0));
  };
};

class FusionGroupOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Inputs",
             "(std::vector<phi::DenseTensor>) The inputs of fusion_group op.")
        .AsDuplicable();
    AddOutput("Outs",
              "(std::vector<phi::DenseTensor>) The outputs of fusion_group op.")
        .AsDuplicable();
    AddAttr<std::vector<int>>("outs_dtype",
                              "The data type of Outputs in fusion_group op.")
        .SetDefault({});
    AddAttr<std::vector<int>>("inputs_dtype",
                              "The data type of Inputs in fusion_group op.")
        .SetDefault({});
    AddAttr<int>("type", "Fusion type.").SetDefault(0);
    AddAttr<std::string>("func_name", "Name of the generated functions.")
        .SetDefault("");
    AddComment(R"DOC(
fusion_group Operator.

It is used to execute a generated CUDA kernel which fuse the computation of
multiple operators into one. It supports several types:
0, fused computation of elementwise operations in which all the dims of inputs
    and outputs should be exactly the same.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(fusion_group,
                            FusionGroupInferShapeFunctor,
                            PD_INFER_META(phi::FusionGroupInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_group,
                  ops::FusionGroupOp,
                  ops::FusionGroupOpMaker,
                  FusionGroupInferShapeFunctor);
