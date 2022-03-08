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
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class GatherTreeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Ids"),
        ctx.device_context());
  }
};

class GatherTreeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "The Tensor with shape [length, batch_size, beam_size] containing "
             "the selected ids of all time steps.");
    AddInput("Parents",
             "The Tensor has the same shape as Ids and contains the parents "
             "corresponding to selected ids when searching among beams.");
    AddOutput(
        "Out",
        "A Tensor with shape [length, batch_size, beam_size] containing the "
        "full sequences. The sequences is collected by backtracing from the "
        "last time step of Ids.");
    AddComment(R"DOC(
GatherTree Operator.

Backtrace from the last time step and generate the full sequences by collecting beam search
selected ids.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(gather_tree, GatherTreeInferShapeFunctor,
                            PD_INFER_META(phi::GatherTreeMeta));

REGISTER_OPERATOR(gather_tree, ops::GatherTreeOp, ops::GatherTreeOpMaker,
                  GatherTreeInferShapeFunctor);
