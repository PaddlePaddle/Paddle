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

#include "paddle/fluid/operators/batch_norm_op.h"

namespace paddle {
namespace operators {

class SyncBatchNormInplaceInToOut : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc &op_desc, bool use_cuda) const override {
    return {{"Mean", "MeanOut"}, {"Variance", "VarianceOut"}, {"X", "Y"}};
  }
};

class SyncBatchNormGradInplaceInToOut : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc &op_desc, bool use_cuda) const override {
    return {
        {framework::GradVarName("Y"), framework::GradVarName("X")},
        {"SavedMean", framework::GradVarName("Scale")},
        {"SavedVariance", framework::GradVarName("Bias")},
    };
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sync_batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormOpInferVarType, ops::BatchNormGradMaker,
                  ops::SyncBatchNormInplaceInToOut);
REGISTER_OPERATOR(sync_batch_norm_grad, ops::BatchNormGradOp,
                  ops::SyncBatchNormGradInplaceInToOut);
