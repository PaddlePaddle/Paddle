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

#include "paddle/fluid/operators/nce_op.h"

#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class NCEOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"));
    PADDLE_ENFORCE(ctx->HasInput("Label"));
    PADDLE_ENFORCE(ctx->HasInput("Weight"));
    PADDLE_ENFORCE(ctx->HasOutput("Cost"));
    PADDLE_ENFORCE(ctx->HasOutput("SampleLogits"));
    PADDLE_ENFORCE(ctx->HasOutput("SampleLabels"));

    auto x_dims = ctx->GetInputDim("Input");
    auto label_dims = ctx->GetInputDim("Label");
    if (ctx->IsRuntime() || (x_dims[0] > 0 && label_dims[0] > 0)) {
      PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0]);
    }
    int num_true_classes = label_dims.size() == 2 ? label_dims[1] : 1;
    if (ctx->HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Weight")[0],
                        ctx->GetInputDim("Bias")[0]);
    }
    auto num_neg_samples = ctx->Attrs().Get<int>("num_neg_samples");
    auto num_total_classes = ctx->Attrs().Get<int>("num_total_classes");
    std::vector<int> custom_neg_classes =
        ctx->Attrs().Get<std::vector<int>>("custom_neg_classes");
    PADDLE_ENFORCE_EQ(num_total_classes, ctx->GetInputDim("Weight")[0]);
    if (custom_neg_classes.size() > 0) {
      PADDLE_ENFORCE_EQ(custom_neg_classes.size(),
                        static_cast<size_t>(num_neg_samples));
    }
    // set dims of output(Out)
    std::vector<int64_t> out_dims;
    out_dims.push_back(x_dims[0]);
    out_dims.push_back(1);
    ctx->SetOutputDim("Cost", framework::make_ddim(out_dims));

    // set dims of output(SampleOut)
    std::vector<int64_t> sample_out_dims;
    sample_out_dims.push_back(x_dims[0]);
    sample_out_dims.push_back(
        (num_true_classes == -1) ? -1 : (num_neg_samples + num_true_classes));
    ctx->SetOutputDim("SampleLogits", framework::make_ddim(sample_out_dims));
    ctx->SetOutputDim("SampleLabels", framework::make_ddim(sample_out_dims));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   platform::CPUPlace());
  }
};

class NCEOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) A tensor of shape [batch_size, dim].");
    AddInput(
        "Label",
        "(Tensor) A tensor of shape [batch_size, num_true_class]. "
        "'num_true_class' is the number of target classes in each sample."
        "The number of target classes per sample should be same. "
        "If you have a variable number of target classes, "
        "you can pad them out to a constant number by either repeating them"
        " or by padding with an otherwise unused class.)");
    AddInput("Weight",
             "(Tensor) A tensor of shape [num_class, dim]. 'num_class' is the "
             "total number of class.");
    AddInput(
        "Bias",
        "(Tensor) A tensor of shape [num_class, 1]. 'num_class' is the total "
        "number of class. It is a dispensable input.")
        .AsDispensable();
    AddInput("SampleWeight",
             "(Tensor) A tensor of shape [batch_size, 1] storing a weight for "
             "each sample. And it is a dispensable input. The default value of "
             "sample is 1.")
        .AsDispensable();

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

    AddOutput("Cost",
              "(Tensor) A tensor of shape [batch_size, 1]. Cost of samples.");
    AddOutput("SampleLogits",
              "An intermediate tensor of shape[batch_size, num_neg_samples + "
              "num_pos_samples]."
              "This tensor is output of forward kernel and used in backward "
              "kernel to compute grads."
              "Given X is  the dot product of input tensor and sampled labels' "
              "weights."
              "Then 'SampleLogits' is sigmoid(X).")
        .AsIntermediate();
    AddOutput("SampleLabels",
              "An intermediate tensor of shape[batch_size, num_neg_samples + "
              "num_pos_samples]."
              "This tensor is output of forward kernel and used in backward "
              "kernel to compute grads."
              "")
        .AsIntermediate();

    AddAttr<int>("num_total_classes",
                 "Total number of classes in all samples.");
    AddAttr<int>("num_neg_samples",
                 "The number of negative classes. The default value is 10.")
        .SetDefault(10);
    AddAttr<int>("sampler",
                 "(int) Which sampler to be used to sample negative class."
                 "0: Uniform; 1: LogUniform; 2: CostumDist.")
        .SetDefault(0);
    AddAttr<int>("seed",
                 "(int) The seed used in sampler. If it is 0, "
                 "the sampler will generate a seed randomly.")
        .SetDefault(0);
    AddAttr<bool>("is_sparse", "(boolean, default false) Sparse update.")
        .SetDefault(false);

    // for parameter prefetch
    AddAttr<bool>("remote_prefetch", "").SetDefault(false);
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));
    AddAttr<std::vector<std::string>>(
        "epmap",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(
        "table_names",
        "(string vector, the splited table names that will be fetched from "
        "parameter server)"
        "in the order of input variables for mapping")
        .SetDefault({});

    AddAttr<std::vector<int>>("custom_neg_classes",
                              "This attribute only be used in unitest. Classes "
                              "in this list wiil be used as negative classes "
                              "for every samples. Under normal conditions, "
                              "user should avoid setting this attribute.")
        .SetDefault({});
    AddComment(R"DOC(
Compute and return the noise-contrastive estimation training loss. See
`Noise-contrastive estimation: A new estimation principle for unnormalized
statistical models
 <http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_.
By default this operator uses a uniform distribution for sampling.
)DOC");
  }
};

class NCEOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"));
    PADDLE_ENFORCE(ctx->HasInput("Weight"));
    PADDLE_ENFORCE(ctx->HasInput("Cost"));
    PADDLE_ENFORCE(ctx->HasInput("SampleLogits"));
    PADDLE_ENFORCE(ctx->HasInput("SampleLabels"));
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Cost")),
                   "The input(Out@GRAD) should not be null.");

    auto x_dims = ctx->GetInputDim("Input");
    auto x_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }

    auto w_dims = ctx->GetInputDim("Weight");
    auto w_grad_name = framework::GradVarName("Weight");
    if (ctx->HasOutput(w_grad_name)) {
      ctx->SetOutputDim(w_grad_name, w_dims);
    }

    auto bias_grad_name = framework::GradVarName("Bias");
    if (ctx->HasOutput(bias_grad_name)) {
      auto bias_dims = ctx->GetInputDim("Bias");
      ctx->SetOutputDim(bias_grad_name, bias_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   platform::CPUPlace());
  }
};

class NCEOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto weight_grad = ctx->Output(framework::GradVarName("Weight")).front();

    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "nce_op_grad op " << weight_grad << " and "
              << " is set to SelectedRows";
      ctx->SetType(weight_grad, framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "nce_op_grad op " << weight_grad << " and "
              << " is set to LoDTensor";
      ctx->SetType(weight_grad, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetDataType(weight_grad, ctx->GetDataType(ctx->Input("Input")[0]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(nce, ops::NCEOp,
                  paddle::framework::DefaultGradOpDescMaker<true>,
                  ops::NCEOpMaker);
REGISTER_OPERATOR(nce_grad, ops::NCEOpGrad, ops::NCEOpGradVarTypeInference);
REGISTER_OP_CPU_KERNEL(nce, ops::NCEKernel<paddle::platform::CPUPlace, float>,
                       ops::NCEKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(nce_grad,
                       ops::NCEGradKernel<paddle::platform::CPUPlace, float>,
                       ops::NCEGradKernel<paddle::platform::CPUPlace, double>);
