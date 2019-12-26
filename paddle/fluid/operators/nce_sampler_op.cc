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

#include "paddle/fluid/operators/nce_sampler_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

class NCESamplerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    auto num_neg_samples = ctx->Attrs().Get<int>("num_neg_samples");
    auto sample_batch_size = ctx->Attrs().Get<int>("sample_batch_size");

    if (ctx->HasOutput("Out")) {
      std::vector<int64_t> sample_out_dims;
      sample_out_dims.push_back(sample_batch_size);
      sample_out_dims.push_back(num_neg_samples);
      ctx->SetOutputDim("Out", framework::make_ddim(sample_out_dims));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   platform::CPUPlace());
  }
};

class NCESamplerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput(
        "CustomDistProbs",
        "(Tensor) It is used in 'CostumDist' sampler. "
        "It is a tensor with shape [num_total_classes]."
        "The i-th element is the probsbility of the i-th class being sampled.")
        .AsDispensable();
    AddInput(
        "CustomDistAlias",
        "(Tensor) It is used in 'CostumDist' sampler. "
        "It is a tensor with shape [num_total_classes]."
        "The i-th element is the probsbility of the i-th class being sampled.")
        .AsDispensable();
    AddInput(
        "CustomDistAliasProbs",
        "(Tensor) It is used in 'CostumDist' sampler. "
        "It is a tensor with shape [num_total_classes]."
        "The i-th element is the probsbility of the i-th class being sampled.")
        .AsDispensable();
    AddOutput("Out",
              "An intermediate tensor of shape[batch_size, num_neg_samples + "
              "num_pos_samples]."
              "This tensor is output of forward kernel and used in backward "
              "kernel to compute grads.")
        .AsDispensable();
    AddOutput("CustomDistProbsInit", "CustomDistProbsInit").AsDispensable();
    AddOutput("CustomDistAliasInit", "CustomDistAliasInit").AsDispensable();
    AddOutput("CustomDistAliasProbsInit", "CustomDistAliasProbsInit")
        .AsDispensable();
    AddAttr<std::string>("filename",
                         "filepath used to save sample count. i-th line is the "
                         "count of sample[i]")
        .SetDefault("");
    AddAttr<bool>(
        "init_flag",
        "If it is true, this op will initialize Probs, Alias and AliasProbS.")
        .SetDefault(false);
    AddAttr<int>("sample_batch_size",
                 "(int) the shape of output of this op is [sample_batch_size, "
                 "num_neg_samples]")
        .SetDefault(1);
    AddAttr<int>("seed",
                 "(int) The seed used in sampler. If it is 0, "
                 "the sampler will generate a seed randomly.")
        .SetDefault(0);
    AddAttr<int>("num_total_classes",
                 "Total number of classes in all samples.");
    AddAttr<int>("num_neg_samples",
                 "The number of negative classes needed to sample. The default "
                 "value is 10.")
        .SetDefault(10);
    AddAttr<float>("factor",
                   "(float) factor to increase the probability of element with "
                   "low frequency to be sampled.")
        .SetDefault(0.75);
    AddComment(R"DOC("Negative Sampler")DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    nce_sampler, ops::NCESamplerOp, ops::NCESamplerOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    nce_sampler, ops::NCESamplerKernel<paddle::platform::CPUPlace, float>,
    ops::NCESamplerKernel<paddle::platform::CPUPlace, double>);
