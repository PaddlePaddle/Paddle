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

#include <memory>

#include "paddle/fluid/operators/batch_norm_op.h"

namespace paddle {
namespace operators {

class SyncBatchNormGradOpDescMaker : public BatchNormGradMaker {
 public:
  using BatchNormGradMaker::BatchNormGradMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("sync_batch_norm_grad");

    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

    op->SetInput("Scale", Input("Scale"));
    op->SetInput("SavedMean", Output("SavedMean"));
    op->SetInput("SavedVariance", Output("SavedVariance"));

    // sync_batch_norm doesn't support use_global_stats True, no
    // op->SetInput("Mean", Output("MeanOut"));
    // op->SetInput("Variance", Output("VarianceOut"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sync_batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormOpInferVarType,
                  ops::SyncBatchNormGradOpDescMaker);
REGISTER_OPERATOR(sync_batch_norm_grad, ops::BatchNormGradOp);
