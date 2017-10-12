/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/linear_chain_crf_op.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;
using framework::LoD;

class LinearChainCrfOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LinearChainCrfOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Emission",
        "(LoDTensor, default: LoDTensor<float>). "
        "The unscaled emission weight matrix for the linear chain CRF. "
        "This input is a LoDTensor with shape [N x D] where N is the total "
        "element number of all input squences in a mini-batch, "
        "and D is the total tag number.");
    AddInput(
        "Transition",
        "(Tensor, default: Tensor<float>). A Tensor with shape [(D + 2) x D]. "
        "The learnable parameter for linear_chain_crf operator. "
        "See more details in the operator's comments.");
    AddInput(
        "Label",
        "(LoDTensor, default: LoDTensor<int>). The ground truth which is a 2-D "
        "LoDTensor with shape [N x 1], where N is the total element number in "
        "a mini-batch.");
    AddOutput(
        "Alpha",
        "Tensor, default: Tensor<float>. The forward vectors for the entire "
        "batch. A two dimensional tensor with shape [N x D], "
        "denoted as \f$\alpha\f$. \f$\alpha$\f is a memo table used to "
        "calculate the normalization factor in CRF. \f$\alpha[k, v]$\f stores "
        "the unnormalized probabilites of all possible unfinished sequences of "
        "tags that end at position \f$k$\f with tag \f$v$\f. For each \f$k$\f, "
        "\f$\alpha[k, v]$\f is a vector of length \f$D$\f with a component for "
        "each tag value \f$v$\f. This vector is called a forward vecotr and "
        "will also be used in backward computations.")
        .AsIntermediate();
    AddOutput(
        "LogLikelihood",
        "(Tensor, default: Tensor<float>). The logarithm of the conditional "
        "likelihood of each training sample in a mini-batch. This is a 2-D "
        "tensor with shape [S x 1], where S is the sequence number in a "
        "mini-batch. "
        "Note: S is equal to the sequence number in a mini-batch. The output "
        "is no longer a LoDTensor.");
    AddComment(R"DOC(
Conditional Random Field defines an undirected probabilistic graph with nodes
denoting random variables and edges denoting dependencies between these
variables. CRF learns the conditional probability \f$P(Y|X)\f$, where
\f$X = (x_1, x_2, ... , x_n)\f$ are structured inputs and
\f$Y = (y_1, y_2, ... , y_n)\f$ are labels for the inputs.

Linear chain CRF is a special case of CRF that is useful for sequence labeling
task. Sequence labeling tasks do not assume a lot of conditional
independences among inputs. They only concern about the input and the output
being linear sequences. Thus, the graph model of CRF is a simple chain or
a line, which results in a linear chain CRF.

This operator implements the Forward-Backward algorithm for linear chain CRF.
Please see http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

Equation:

- Denote Input(Emission) to this operator as \f$x\f$ here.
- The first D values of Input(Transition) to this operator are for starting
weights, denoted as \f$a\f$ here.
- The next D values of Input(Transition) of this operator are for ending
weights, denoted as \f$b\f$ here.
- The remaning values of Input(Transition) are for transition weights,
denoted as \f$w\f$ here.
- Denote Input(Label) as \f$s\f$ here.

The probability of a sequence \f$s\f$ of length \f$L\f$ is defined as:
\f$P(s) = (1/Z) exp(a_{s_1} + b_{s_L}
                 + \sum_{l=1}^L x_{s_l}
                 + \sum_{l=2}^L w_{s_{l-1},s_l})\f$
where \f$Z\f$ is a normalization value so that the sum of \f$P(s)\f$ over
all possible sequences is \f$1\f$, and \f$x\f$ is the emission feature weight
to the linear chain CRF.

Finaly, the linear chain CRF operator outputs the logarithm of the conditional
likelihood of each training sample in a mini-batch.

NOTE:
1. The feature function for a CRF is made up of the emission features and the
transition features. The emission feature weights are NOT computed in
this operator. They MUST be computed first before this operator is called.

2. Because this operator performs globally normaliztion over all possible
sequences internally, it expects UNSCALED emission feature weights.
Please do not call this op with the emission feature being output of any
nonlinear activation.

3. The 2nd dimension of Input(Emission) MUST be equal to the tag number.

)DOC");
  }
};

class LinearChainCrfOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Emission"),
                   "Input(Emission) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Transition"),
                   "Input(Transition) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Alpha"),
                   "Output(Alpha) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("LogLikelihood"),
                   "Output(LogLikelihood) should be not null.");

    auto emission_dims = ctx->GetInputDim("Emission");
    auto transition_dims = ctx->GetInputDim("Transition");
    auto label_dims = ctx->GetInputDim("Label");

    PADDLE_ENFORCE_EQ(emission_dims.size(), 2UL,
                      "The Input(Emission) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(transition_dims.size(), 2UL,
                      "The Input(Transition) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        transition_dims[0] - 2, transition_dims[1],
        "An invalid dimension for the Input(Transition), which should "
        "be a 2-D tensor with shape [D + 2 x D].");
    PADDLE_ENFORCE_EQ(
        emission_dims[1], transition_dims[1],
        "The 2nd dimension of the Input(Emission) and the Input(Transition) "
        "should be equal to the tag number.");
    PADDLE_ENFORCE(label_dims.size() == 2UL && label_dims[1] == 1UL,
                   "The Input(Label) should be a 2-D tensor with the 2nd "
                   "dimensions fixed to 1.");
    PADDLE_ENFORCE_EQ(
        emission_dims[0], label_dims[0],
        "The height of Input(Emission) and the height of Input(Label) "
        "should be the same.");

    ctx->SetOutputDim("Alpha", emission_dims);

    // (TODO caoying) This is tricky. The 1st dimension of Output(LogLikelihood)
    // is the sequence number in a mini-batch. The dimension set here should be
    // resized to its correct size in the function Compute.
    ctx->SetOutputDim("LogLikelihood", {emission_dims[0], 1});
  }

  // Explicitly set that the data type of output of the linear_chain_crf
  // operator is determined by its input "Emission".
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("Emission")->type());
  }
};

template <typename T>
class LinearChainCrfOpKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");

    auto* emission_weights = ctx.Input<LoDTensor>("Emission");
    auto* transition_weights = ctx.Input<Tensor>("Transition");
    auto* label = ctx.Input<LoDTensor>("Label");

    auto in_lod = emission_weights->lod();
    // TODO(caoying) The checks related to LoD information should be
    // moved into InferShape once after the InferShape is refactored.
    PADDLE_ENFORCE_EQ(emission_weights->NumLevels(), 1UL,
                      "The Input(Emission) should be a sequence.");
    PADDLE_ENFORCE_EQ(label->NumLevels(), 1UL,
                      "The Input(Label) should be a sequence.");
    const size_t level = 0;

    auto emission_dims = emission_weights->dims();
    const size_t seq_num = in_lod[level].size() - 1;

    // TODO(caoying) These local variables seems to be created and destroied
    // every time this function is called. Will this bring additional overhead?
    Tensor emission_exps;
    Tensor emission_row_max;
    Tensor transition_exps;
    emission_exps.mutable_data<T>(emission_dims, platform::CPUPlace());
    emission_row_max.mutable_data<T>(
        framework::make_ddim({emission_dims[0], 1}), platform::CPUPlace());
    transition_exps.mutable_data<T>(transition_weights->dims(),
                                    platform::CPUPlace());

    auto* alpha = ctx.Output<Tensor>("Alpha");
    alpha->mutable_data<T>(ctx.GetPlace());
    auto* ll = ctx.Output<Tensor>("LogLikelihood");
    // resize the output tensor to the correct dimension.
    ll->Resize({static_cast<int>(seq_num), 1});
    T* log_likelihood = ll->mutable_data<T>(ctx.GetPlace());

    for (size_t i = 0; i < seq_num; ++i) {
      int start_pos = static_cast<int>(in_lod[level][i]);
      int end_pos = static_cast<int>(in_lod[level][i + 1]);

      const Tensor one_seq = emission_weights->Slice<T>(start_pos, end_pos);
      Tensor one_seq_row_max = emission_row_max.Slice<T>(start_pos, end_pos);
      Tensor one_seq_exps = emission_exps.Slice<T>(start_pos, end_pos);
      const Tensor one_seq_label = label->Slice<T>(start_pos, end_pos);
      Tensor one_seq_alpha = alpha->Slice<T>(start_pos, end_pos);

      log_likelihood[i] = ForwardOneSequence(
          ctx.device_context(), one_seq, one_seq_row_max, one_seq_exps,
          (*transition_weights), transition_exps, one_seq_label, one_seq_alpha);
    }
  }

 protected:
  T ForwardOneSequence(const platform::DeviceContext& ctx,
                       const Tensor& emission, Tensor& emission_row_max,
                       Tensor& emission_exps, const Tensor& trans_weights,
                       Tensor& trans_weight_exps, const Tensor& label,
                       Tensor& alpha) const {
    // (TODO caoying) Evaluate and optimize this.
    // The Eigen compution kernel will be invoked for multiple times.
    // Some computations regardless of sequence inforamtion could be performed
    // only one time for the entire batch. This potentially could be optimized.

    auto x_dims = emission.dims();
    const size_t seq_length = x_dims[0];
    const size_t tag_num = x_dims[1];

    T* alpha_value = alpha.data<T>();

    auto x = EigenMatrix<T>::From(emission);
    auto x_row_max = EigenMatrix<T>::From(emission_row_max);
    const int class_dim = 1;
    x_row_max.device(*ctx.GetEigenDevice<platform::CPUPlace>()) =
        x.maximum(Eigen::DSizes<int, 1>(class_dim))
            .reshape(Eigen::DSizes<int, 2>(int(seq_length), 1));

    auto x_exps = EigenMatrix<T>::From(emission_exps);
    x_exps.device(*ctx.GetEigenDevice<platform::CPUPlace>()) =
        (x - x_row_max.broadcast(Eigen::DSizes<int, 2>(1, tag_num))).exp();

    auto w = EigenMatrix<T>::From(trans_weights);
    auto w_exps = EigenMatrix<T>::From(trans_weight_exps);
    w_exps.device(*ctx.GetEigenDevice<platform::CPUPlace>()) = w.exp();
    // The 1st row of w are transition weights for start mask.
    const size_t start_ridx = 0;
    // The 2nd row of w are transition weights for end mask.
    const size_t end_ridx = 1;
    // Transition weights among other tags begins from the 3rd row of w.
    const size_t state_base_ridx = 2;

    for (size_t i = 0; i < tag_num; ++i) {
      alpha_value[i] = w_exps(start_ridx, i) * x_exps(0, i);
    }
    T ll = -x_row_max(0, 1) - std::log(NormalizeL1(alpha_value, tag_num));

    for (size_t k = 1; k < seq_length; ++k) {
      for (size_t i = 0; i < tag_num; ++i) {
        T sum = 0.;
        for (size_t j = 0; j < tag_num; ++j) {
          sum += alpha_value[(k - 1) * tag_num + j] *
                 w_exps(j + state_base_ridx, i);
        }
        alpha_value[k * tag_num + i] = x_exps(k, i) * sum;
      }
      ll -= x_row_max(k, 1) +
            std::log(NormalizeL1(alpha_value + k * tag_num, tag_num));
    }
    T sum = 0.;
    for (size_t i = 0; i < tag_num; ++i) {
      sum += alpha_value[(seq_length - 1) * tag_num + i] * w_exps(end_ridx, i);
    }
    ll -= std::log(sum);

    const int* lbl = label.data<int>();
    PADDLE_ENFORCE_LT(
        *std::max_element(lbl, lbl + seq_length), tag_num,
        "An invalid tag label that execesses the largest tag number.");

    // Calculate the nominator part, which depends on the label sequence.
    ll += w(start_ridx, lbl[0]) + x(start_ridx, lbl[0]) +
          w(end_ridx, lbl[seq_length - 1]);
    for (size_t k = 1; k < seq_length; ++k)
      ll += x(k, lbl[k]) + w(lbl[k - 1], lbl[k]);
    return -ll;
  }

 private:
  T NormalizeL1(T* x, size_t len) const {
    T sum = 0.;
    for (size_t i = 0; i < len; ++i) sum += x[i];
    // (This comment is from the old LinearChainCRFLayer.)
    // Right now, we just bet that sum won't be zero. If this really happens, we
    // will figure out what should be done then.
    PADDLE_ENFORCE(sum,
                   "The unnormalized probabilites of all possible unfinished "
                   "sequences must be greater than 0.");
    for (size_t i = 0; i < len; ++i) x[i] /= sum;
    return sum;
  }
};

class LinearChainCrfGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
};

template <typename T>
class LinearChainCrfGradOpKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(linear_chain_crf, ops::LinearChainCrfOp, ops::LinearChainCrfOpMaker,
            linear_chain_crf_grad, ops::LinearChainCrfGradOp);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf,
    ops::LinearChainCrfOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf_grad,
    ops::LinearChainCrfGradOpKernel<paddle::platform::CPUPlace, float>);
