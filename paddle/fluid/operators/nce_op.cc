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

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class NCEOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "nce");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "nce");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "nce");

    OP_INOUT_CHECK(ctx->HasOutput("Cost"), "Output", "Cost", "nce");
    bool is_test = ctx->Attrs().Get<bool>("is_test");
    if (!is_test) {
      OP_INOUT_CHECK(ctx->HasOutput("SampleLogits"), "Output", "SampleLogits",
                     "nce");
      OP_INOUT_CHECK(ctx->HasOutput("SampleLabels"), "Output", "SampleLabels",
                     "nce");
    }

    auto x_dims = ctx->GetInputDim("Input");
    auto label_dims = ctx->GetInputDim("Label");
    if (ctx->IsRuntime() || (x_dims[0] > 0 && label_dims[0] > 0)) {
      PADDLE_ENFORCE_EQ(
          x_dims[0], label_dims[0],
          platform::errors::InvalidArgument(
              "The first dimension of Input(Input) and Input(Label) should be "
              "equal in runtime. But received: Input(Input)'s shape = [%s] "
              "with 1st dim =  %d, Input(Label)'s shape = [%s] with 1st dim = "
              "%d.",
              x_dims, x_dims[0], label_dims, label_dims[0]));
    }
    int num_true_classes = label_dims.size() == 2 ? label_dims[1] : 1;
    if (ctx->HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Weight")[0], ctx->GetInputDim("Bias")[0],
          platform::errors::InvalidArgument(
              "The first dimension of Input(Weight) and Input(Bias) "
              "should be equal. But received: Input(Weight)'s shape = [%s] "
              "with 1st dim = %d, and Input(Bias)'s shape = [%s] with 1st dim "
              "= %d.",
              ctx->GetInputDim("Weight"), ctx->GetInputDim("Weight")[0],
              ctx->GetInputDim("Bias"), ctx->GetInputDim("Bias")[0]));
    }
    auto num_neg_samples = ctx->Attrs().Get<int>("num_neg_samples");
    auto num_total_classes = ctx->Attrs().Get<int>("num_total_classes");
    std::vector<int> custom_neg_classes =
        ctx->Attrs().Get<std::vector<int>>("custom_neg_classes");
    PADDLE_ENFORCE_EQ(
        num_total_classes, ctx->GetInputDim("Weight")[0],
        platform::errors::InvalidArgument(
            "The number of total classes should be equal to the first "
            "dimension of Input(Weight). But received: Attr(num_total_classes) "
            "= %d, Input(Weight)'s shape = [%s] with 1st dim = %d.",
            num_total_classes, ctx->GetInputDim("Weight"),
            ctx->GetInputDim("Weight")[0]));
    if (custom_neg_classes.size() > 0) {
      PADDLE_ENFORCE_EQ(
          custom_neg_classes.size(), static_cast<size_t>(num_neg_samples),
          platform::errors::InvalidArgument(
              "The size of Attr(custom_neg_classes) should be equal "
              "to the number of negative samples. But received: "
              "custom_neg_classes.size() = %d, num_neg_samples = %d.",
              custom_neg_classes.size(), num_neg_samples));
    }
    // set dims of output(Out)
    std::vector<int64_t> out_dims;
    out_dims.push_back(x_dims[0]);
    out_dims.push_back(1);
    ctx->SetOutputDim("Cost", phi::make_ddim(out_dims));

    if (!is_test) {
      // set dims of output(SampleOut)
      std::vector<int64_t> sample_out_dims;
      sample_out_dims.push_back(x_dims[0]);
      sample_out_dims.push_back(
          (num_true_classes == -1) ? -1 : (num_neg_samples + num_true_classes));
      ctx->SetOutputDim("SampleLogits", phi::make_ddim(sample_out_dims));
      ctx->SetOutputDim("SampleLabels", phi::make_ddim(sample_out_dims));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
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
        "The i-th element is the probability of the i-th class being sampled.")
        .AsDispensable();
    AddInput(
        "CustomDistAlias",
        "(Tensor) It is used in 'CostumDist' sampler. "
        "It is a tensor with shape [num_total_classes]."
        "The i-th element is the probability of the i-th class being sampled.")
        .AsDispensable();
    AddInput(
        "CustomDistAliasProbs",
        "(Tensor) It is used in 'CostumDist' sampler. "
        "It is a tensor with shape [num_total_classes]."
        "The i-th element is the probability of the i-th class being sampled.")
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
        .AsIntermediate()
        .AsExtra();
    AddOutput("SampleLabels",
              "An intermediate tensor of shape[batch_size, num_neg_samples + "
              "num_pos_samples]."
              "This tensor is output of forward kernel and used in backward "
              "kernel to compute grads."
              "")
        .AsIntermediate()
        .AsExtra();

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
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.")
        .SetDefault(0)
        .AsExtra();
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}))
        .AsExtra();
    AddAttr<std::vector<std::string>>(
        "epmap",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<std::string>>(
        "table_names",
        "(string vector, the split table names that will be fetched from "
        "parameter server)"
        "in the order of input variables for mapping")
        .SetDefault({})
        .AsExtra();

    AddAttr<std::vector<int>>("custom_neg_classes",
                              "This attribute only be used in unitest. Classes "
                              "in this list wiil be used as negative classes "
                              "for every samples. Under normal conditions, "
                              "user should avoid setting this attribute.")
        .SetDefault({})
        .AsExtra();
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference "
                  "only, false for training.")
        .SetDefault(false);
    AddComment(R"DOC(
Compute and return the noise-contrastive estimation training loss. See
`Noise-contrastive estimation: A new estimation principle for unnormalized
statistical models
 <http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_.
By default this operator uses a uniform distribution for sampling.
)DOC");
  }
};

template <typename T>
class NCEGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput("SampleLogits", this->Output("SampleLogits"));
    op->SetInput("SampleLabels", this->Output("SampleLabels"));
    op->SetInput("SampleWeight", this->Input("SampleWeight"));
    op->SetInput("CustomDistProbs", this->Input("CustomDistProbs"));
    op->SetInput("CustomDistAlias", this->Input("CustomDistAlias"));
    op->SetInput("CustomDistAliasProbs", this->Input("CustomDistAliasProbs"));
    op->SetInput(framework::GradVarName("Cost"), this->OutputGrad("Cost"));
    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetOutput(framework::GradVarName("Weight"), this->InputGrad("Weight"));
    op->SetAttrMap(this->Attrs());
  }
};

class NCEOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "nce_grad");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "nce_grad");
    OP_INOUT_CHECK(ctx->HasInput("SampleLogits"), "Input", "SampleLogits",
                   "nce_grad");
    OP_INOUT_CHECK(ctx->HasInput("SampleLabels"), "Input", "SampleLabels",
                   "nce_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Cost")), "Input",
                   framework::GradVarName("Cost"), "nce_grad");

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
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        platform::CPUPlace());
  }
};

class NCEOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto weight_grad = framework::GradVarName("Weight");

    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = BOOST_GET(bool, attr);
    if (is_sparse) {
      VLOG(3) << "nce_op_grad op " << weight_grad << " and "
              << " is set to SelectedRows";
      ctx->SetOutputType(weight_grad, framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "nce_op_grad op " << weight_grad << " and "
              << " is set to LoDTensor";
      ctx->SetOutputType(weight_grad, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetOutputDataType(weight_grad, ctx->GetInputDataType("Input"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(NCEGradOpNoNeedBufferVarInferer, "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(nce, ops::NCEOp, ops::NCEOpMaker,
                  ops::NCEGradOpMaker<paddle::framework::OpDesc>,
                  ops::NCEGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(nce_grad, ops::NCEOpGrad, ops::NCEOpGradVarTypeInference,
                  ops::NCEGradOpNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(nce, ops::NCEKernel<paddle::platform::CPUPlace, float>,
                       ops::NCEKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(nce_grad,
                       ops::NCEGradKernel<paddle::platform::CPUPlace, float>,
                       ops::NCEGradKernel<paddle::platform::CPUPlace, double>);
