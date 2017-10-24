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

namespace {
template <typename T>
T NormalizeL1(T* x, size_t len) {
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
}  // namespace

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
    AddOutput("EmissionExps",
              "The exponentials of Input(Emission). This is an intermediate "
              "computational result in forward computation, and will be reused "
              "in backward computation.")
        .AsIntermediate();
    AddOutput("TransitionExps",
              "The exponentials of Input(Transition). This is an intermediate "
              "computational result in forward computation, and will be reused "
              "in backward computation.")
        .AsIntermediate();
    AddOutput(
        "LogLikelihood",
        "(Tensor, default: Tensor<float>). The logarithm of the "
        "conditional "
        "likelihood of each training sample in a mini-batch. This is a 2-D "
        "tensor with shape [S x 1], where S is the sequence number in a "
        "mini-batch. "
        "Note: S is equal to the sequence number in a mini-batch. The "
        "output "
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

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Emission"),
                   "Input(Emission) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Transition"),
                   "Input(Transition) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Alpha"),
                   "Output(Alpha) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("EmissionExps"),
                   "Output(EmissionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("TransitionExps"),
                   "Output(TransitionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("LogLikelihood"),
                   "Output(LogLikelihood) should be not null.");

    auto emission_dims = ctx->GetInputDim("Emission");
    PADDLE_ENFORCE_EQ(emission_dims.size(), 2UL,
                      "The Input(Emission) should be a 2-D tensor.");
    PADDLE_ENFORCE(emission_dims[0], "An empty mini-batch is not allowed.");

    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(transition_dims.size(), 2UL,
                      "The Input(Transition) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        transition_dims[0] - 2, transition_dims[1],
        "An invalid dimension for the Input(Transition), which should "
        "be a 2-D tensor with shape [(D + 2) x D].");
    PADDLE_ENFORCE_EQ(
        emission_dims[1], transition_dims[1],
        "The 2nd dimension of the Input(Emission) and the Input(Transition) "
        "should be equal to the tag number.");

    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE(label_dims.size() == 2UL && label_dims[1] == 1UL,
                   "The Input(Label) should be a 2-D tensor with the 2nd "
                   "dimensions fixed to 1.");
    PADDLE_ENFORCE_EQ(
        emission_dims[0], label_dims[0],
        "The height of Input(Emission) and the height of Input(Label) "
        "should be the same.");

    ctx->SetOutputDim("Alpha", emission_dims);
    ctx->SetOutputDim("EmissionExps", emission_dims);
    ctx->SetOutputDim("TransitionExps", transition_dims);
    // (TODO caoying) This is tricky. The 1st dimension of Output(LogLikelihood)
    // is the sequence number in a mini-batch. The dimension set here should be
    // resized to its correct size in the function Compute.
    ctx->SetOutputDim("LogLikelihood", {emission_dims[0], 1});

    ctx->ShareLoD("Emission", /*->*/ "EmissionExps");
  }

 protected:
  // Explicitly set that the data type of output of the linear_chain_crf
  // operator is determined by its input "Emission".
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<LoDTensor>("Emission")->type());
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
    auto* emission_exps = ctx.Output<LoDTensor>("EmissionExps");
    emission_exps->mutable_data<T>(platform::CPUPlace());
    auto* transition_exps = ctx.Output<Tensor>("TransitionExps");
    transition_exps->mutable_data<T>(platform::CPUPlace());
    auto* label = ctx.Input<LoDTensor>("Label");

    auto in_lod = emission_weights->lod();
    PADDLE_ENFORCE(in_lod.size(), "Input(Emission) is not a sequence.");

    // TODO(caoying) The checks related to LoD information should be
    // moved into InferShape once after the InferShape is refactored.
    PADDLE_ENFORCE_EQ(emission_weights->NumLevels(), 1UL,
                      "The Input(Emission) should be a sequence.");
    PADDLE_ENFORCE_EQ(label->NumLevels(), 1UL,
                      "The Input(Label) should be a sequence.");
    const size_t level = 0;

    auto emission_dims = emission_weights->dims();
    const size_t batch_size = emission_dims[0];
    const size_t tag_num = emission_dims[1];
    const size_t seq_num = in_lod[level].size() - 1;

    Tensor emission_row_max;
    emission_row_max.mutable_data<T>(
        framework::make_ddim({static_cast<int>(batch_size), 1}),
        platform::CPUPlace());

    auto place = ctx.GetEigenDevice<platform::CPUPlace>();
    auto x = EigenMatrix<T>::From(*emission_weights);
    auto x_row_max = EigenMatrix<T>::From(emission_row_max);
    x_row_max.device(place) =
        x.maximum(Eigen::DSizes<int, 1>(1))
            .reshape(Eigen::DSizes<int, 2>(int(batch_size), 1));

    auto x_exps = EigenMatrix<T>::From(*emission_exps);
    x_exps.device(place) =
        (x - x_row_max.broadcast(Eigen::DSizes<int, 2>(1, tag_num))).exp();

    auto w = EigenMatrix<T>::From(*transition_weights);
    auto w_exps = EigenMatrix<T>::From(*transition_exps);
    w_exps.device(place) = w.exp();

    auto* alpha = ctx.Output<LoDTensor>("Alpha");
    alpha->mutable_data<T>(ctx.GetPlace());
    auto* ll = ctx.Output<LoDTensor>("LogLikelihood");
    // resize the output tensor to the correct dimension.
    ll->Resize({static_cast<int>(seq_num), 1});
    T* log_likelihood = ll->mutable_data<T>(ctx.GetPlace());
    for (size_t i = 0; i < seq_num; ++i) {
      int start_pos = static_cast<int>(in_lod[level][i]);
      int end_pos = static_cast<int>(in_lod[level][i + 1]);
      if (end_pos == start_pos) {
        // If an empty input sequence is given, pad 0 for its cost.
        log_likelihood[i] = static_cast<T>(0.);
        continue;
      }

      const Tensor one_seq = emission_weights->Slice(start_pos, end_pos);
      Tensor one_seq_row_max = emission_row_max.Slice(start_pos, end_pos);
      Tensor one_seq_exps = emission_exps->Slice(start_pos, end_pos);
      const Tensor one_seq_label = label->Slice(start_pos, end_pos);
      Tensor one_seq_alpha = alpha->Slice(start_pos, end_pos);

      log_likelihood[i] = ForwardOneSequence(
          &one_seq, &one_seq_row_max, &one_seq_exps, transition_weights,
          transition_exps, &one_seq_label, &one_seq_alpha);
    }
  }

 protected:
  T ForwardOneSequence(const Tensor* emission, const Tensor* emission_row_max,
                       const Tensor* emission_exps, const Tensor* trans_weights,
                       const Tensor* trans_weight_exps, const Tensor* label,
                       Tensor* alpha) const {
    const T* x = emission->data<T>();
    const T* x_row_max = emission_row_max->data<T>();
    const T* x_exps = emission_exps->data<T>();
    const T* w = trans_weights->data<T>();
    const T* w_exps = trans_weight_exps->data<T>();
    T* alpha_value = alpha->data<T>();

    auto x_dims = emission->dims();
    const size_t seq_length = x_dims[0];
    const size_t tag_num = x_dims[1];
    // The 1st row of w are transition weights for start mask.
    // The 2nd row of w are transition weights for end mask.
    // Transition weights among other tags begins from the 3rd row of w.
    const size_t state_trans_base_idx = 2;

    for (size_t i = 0; i < tag_num; ++i) {
      alpha_value[i] = w_exps[i] * x_exps[i];
    }
    T ll = -x_row_max[0] - std::log(NormalizeL1<T>(alpha_value, tag_num));

    for (size_t k = 1; k < seq_length; ++k) {
      for (size_t i = 0; i < tag_num; ++i) {
        T sum = static_cast<T>(0.);
        for (size_t j = 0; j < tag_num; ++j) {
          sum += alpha_value[(k - 1) * tag_num + j] *
                 w_exps[(j + state_trans_base_idx) * tag_num + i];
        }
        alpha_value[k * tag_num + i] = x_exps[k * tag_num + i] * sum;
      }
      ll -= x_row_max[k] +
            std::log(NormalizeL1<T>(alpha_value + k * tag_num, tag_num));
    }
    T sum = 0.;
    for (size_t i = 0; i < tag_num; ++i) {
      sum += alpha_value[(seq_length - 1) * tag_num + i] * w_exps[tag_num + i];
    }
    ll -= std::log(sum);

    const int* lbl = label->data<int>();
    PADDLE_ENFORCE_LT(
        *std::max_element(lbl, lbl + seq_length), tag_num,
        "An invalid tag label that execesses the largest tag number.");

    // Calculate the nominator part, which depends on the label sequence.
    ll += w[lbl[0]] /*start transition*/ + x[lbl[0]] +
          w[tag_num + lbl[seq_length - 1]] /*end transition*/;
    for (size_t k = 1; k < seq_length; ++k) {
      ll += x[k * tag_num + lbl[k]] +
            w[(lbl[k - 1] + state_trans_base_idx) * tag_num + lbl[k]];
    }
    return -ll;
  }
};

class LinearChainCrfGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("EmissionExps"),
                   "Input(EmissionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("TransitionExps"),
                   "Input(TransitionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("LogLikelihood")),
                   "Input(LogLikelihood@GRAD) shoudl be not null.");

    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Emission")),
                   "Output(Emission@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Transition")),
                   "Output(Transition@GRAD) should be not null.");

    auto emission_exps_dims = ctx->GetInputDim("EmissionExps");
    PADDLE_ENFORCE_EQ(emission_exps_dims.size(), 2UL,
                      "The Input(EmissionExps) should be a 2-D tensor.");
    PADDLE_ENFORCE(emission_exps_dims[0],
                   "An empty mini-batch is not allowed.");

    auto transition_exps_dims =
        ctx->GetInputDim(framework::GradVarName("TransitionExps"));
    PADDLE_ENFORCE_EQ(transition_exps_dims.size(), 2UL,
                      "The Input(TransitionExps) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        transition_exps_dims[0] - 2, transition_exps_dims[1],
        "An invalid dimension for the Input(TransitionExps), which should "
        "be a 2-D tensor with shape [(D + 2) x D].");
    PADDLE_ENFORCE_EQ(
        emission_exps_dims[1], transition_exps_dims[1],
        "The 2nd dimension of the Input(EmissionExps) and the "
        "Input(TransitionExps) should be equal to the tag number.");

    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE(label_dims.size() == 2UL && label_dims[1] == 1UL,
                   "The Input(Label) should be a 2-D tensor with the 2nd "
                   "dimensions fixed to 1.");
    PADDLE_ENFORCE_EQ(
        emission_exps_dims[0], label_dims[0],
        "The height of Input(EmissionExps) and the height of Input(Label) "
        "should be the same.");

    ctx->SetOutputDim(framework::GradVarName("Emission"), emission_exps_dims);
    ctx->SetOutputDim(framework::GradVarName("Transition"),
                      transition_exps_dims);
  }

 protected:
  // Explicitly set that the data type of output of the linear_chain_crf_grad
  // operator is determined by its input "EmissionExps".
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<LoDTensor>("EmissionExps")->type());
  }
};

template <typename T>
class LinearChainCrfGradOpKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
    auto* label = ctx.Input<LoDTensor>("Label");
    auto* emission_exps = ctx.Input<LoDTensor>("EmissionExps");
    auto* transition_exps = ctx.Input<Tensor>("TransitionExps");
    auto* alpha = ctx.Input<LoDTensor>("Alpha");
    const T* ll_grad =
        ctx.Input<Tensor>(framework::GradVarName("LogLikelihood"))->data<T>();

    auto* emission_grad =
        ctx.Output<Tensor>(framework::GradVarName("Emission"));
    emission_grad->mutable_data<T>(platform::CPUPlace());

    auto* trans_grad = ctx.Output<Tensor>(framework::GradVarName("Transition"));
    if (trans_grad) trans_grad->mutable_data<T>(platform::CPUPlace());

    auto emission_dims = emission_exps->dims();

    // Beta is the memo table used in dynamic programming to calculate the
    // backwark vectors. For a backward vector i (the i-th row of beta), it
    // captures the unnormalized probabilities of partial sequences starting at
    // position i.
    Tensor beta;
    beta.mutable_data<T>(emission_dims, platform::CPUPlace());

    const size_t level = 0;  // currently, only support sequence.
    auto lod = label->lod();
    PADDLE_ENFORCE(lod.size(), "Input(Label) is not a sequence.");

    for (size_t i = 0; i < lod[level].size() - 1; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);
      if (end_pos == start_pos) continue;

      const Tensor one_seq_emission_exps =
          emission_exps->Slice(start_pos, end_pos);
      const Tensor one_seq_label = label->Slice(start_pos, end_pos);
      const Tensor one_seq_alpha = alpha->Slice(start_pos, end_pos);
      Tensor one_seq_beta = beta.Slice(start_pos, end_pos);
      Tensor one_seq_emission_grad = emission_grad->Slice(start_pos, end_pos);

      BackwardOneSequence(ctx.device_context(), ll_grad[i],
                          &one_seq_emission_exps, transition_exps,
                          &one_seq_alpha, &one_seq_label, &one_seq_beta,
                          trans_grad, &one_seq_emission_grad);
    }
  }

 protected:
  void BackwardOneSequence(const platform::DeviceContext& ctx, const T ll_grad,
                           const Tensor* emission_exps,
                           const Tensor* transition_exps, const Tensor* alpha,
                           const Tensor* label, Tensor* beta,
                           Tensor* transition_grad,
                           Tensor* emission_grad) const {
    const T* w_exps = transition_exps->data<T>();
    const T* x_exps = emission_exps->data<T>();
    const int* label_value = label->data<int>();
    T* beta_value = beta->data<T>();

    auto x_dims = emission_exps->dims();
    const size_t seq_length = x_dims[0];
    const size_t tag_num = x_dims[1];
    const size_t state_trans_base_idx = 2;

    // Calculate the backwark vectors beta.
    // First, calculate the initialition state.
    for (int i = 0; i < tag_num; ++i) {
      beta_value[(seq_length - 1) * tag_num + i] = w_exps[tag_num + i];
    }
    NormalizeL1<T>(beta_value + (seq_length - 1) * tag_num, tag_num);

    for (int k = seq_length - 2; k >= 0; --k) {
      for (int i = 0; i < tag_num; ++i) {
        T sum = static_cast<T>(0.);
        for (int j = 0; j < tag_num; ++j) {
          sum += w_exps[(i + state_trans_base_idx) * tag_num + j] *
                 x_exps[(k + 1) * tag_num + j] *
                 beta_value[(k + 1) * tag_num + j];
        }
        beta_value[k * tag_num + i] = sum;
      }
      NormalizeL1<T>(beta_value + k * tag_num, tag_num);
    }

    auto alpha_mat = EigenMatrix<T>::From(*alpha);
    auto beta_mat = EigenMatrix<T>::From(*beta);
    auto x_grad_mat = EigenMatrix<T>::From(*emission_grad);
    auto* place = ctx.GetEigenDevice<platform::CPUPlace>();
    x_grad_mat.device(*place) = alpha_mat * beta_mat;
    x_grad_mat /= x_grad_mat.sum(Eigen::DSizes<int, 1>(1))
                      .reshape(Eigen::DSizes<int, 2>(seq_length, 1))
                      .broadcast(Eigen::DSizes<int, 2>(1, tag_num));

    for (int k = 0; k < seq_length; ++k) {
      x_grad_mat(k, label_value[k]) -= static_cast<T>(1);
    }

    if (transition_grad) {
      T* trans_grad = transition_grad->data<T>();
      for (size_t k = 0; k < tag_num; ++k) {
        trans_grad[k] += x_grad_mat(/*from start state*/ 0, k);
        trans_grad[tag_num + k] +=
            x_grad_mat(/*to end state*/ seq_length - 1, k);
      }

      auto x_exps_mat = EigenMatrix<T>::From(*emission_exps);
      beta_mat = beta_mat * x_exps_mat;
      beta_mat /= beta_mat.sum(Eigen::DSizes<int, 1>(1))
                      .reshape(Eigen::DSizes<int, 2>(seq_length, 1))
                      .broadcast(Eigen::DSizes<int, 2>(1, tag_num));

      for (int k = 1; k < seq_length; ++k) {
        T sum = static_cast<T>(0.);
        for (int i = 0; i < tag_num; ++i) {
          for (int j = 0; j < tag_num; ++j) {
            sum += w_exps[(i + state_trans_base_idx) * tag_num + j] *
                   alpha_mat(k - 1, i) * beta_mat(k, j);
          }
        }
        sum = static_cast<T>(1.) / sum;
        for (int i = 0; i < tag_num; ++i) {
          for (int j = 0; j < tag_num; ++j) {
            trans_grad[(i + state_trans_base_idx) * tag_num + j] +=
                sum * w_exps[(i + state_trans_base_idx) * tag_num + j] *
                alpha_mat(k - 1, i) * beta_mat(k, j);
          }
        }
        trans_grad[label_value[k - 1] * tag_num + label_value[k]] -=
            static_cast<T>(1.);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(linear_chain_crf, ops::LinearChainCrfOp, ops::LinearChainCrfOpMaker,
            linear_chain_crf_grad, ops::LinearChainCrfGradOp);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf,
    ops::LinearChainCrfOpKernel<paddle::platform::CPUPlace, float>,
    ops::LinearChainCrfOpKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf_grad,
    ops::LinearChainCrfGradOpKernel<paddle::platform::CPUPlace, float>,
    ops::LinearChainCrfGradOpKernel<paddle::platform::CPUPlace, double>);
