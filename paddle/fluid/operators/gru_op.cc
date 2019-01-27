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

#include "paddle/fluid/operators/gru_op.h"
#include <string>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/gru_cpu_kernel.h"
#include "paddle/fluid/operators/math/detail/gru_kernel.h"

DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace operators {

using framework::Tensor;

class GRUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(%s) of GRUOp should not be null.", "Input");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(%s) of GRUOp should not be null.", "Weight");
    PADDLE_ENFORCE(ctx->HasOutput("BatchGate"),
                   "Output(%s) of GRUOp should not be null.", "BatchGate");
    PADDLE_ENFORCE(ctx->HasOutput("BatchResetHiddenPrev"),
                   "Output(%s) of GRUOp should not be null.",
                   "BatchResetHiddenPrev");
    PADDLE_ENFORCE(ctx->HasOutput("BatchHidden"),
                   "Output(%s) of GRUOp should not be null.", "BatchHidden");
    PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                   "Output(%s) of GRUOp should not be null.", "Hidden");
    auto input_dims = ctx->GetInputDim("Input");
    auto weight_dims = ctx->GetInputDim("Weight");
    int input_size = input_dims[1];
    int frame_size = weight_dims[0];
    PADDLE_ENFORCE_EQ(input_size, frame_size * 3,
                      "The input_size must be 3 times of frame_size in GRUOp.");
    PADDLE_ENFORCE_EQ(
        weight_dims[1], frame_size * 3,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    if (ctx->HasInput("H0")) {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(h0_dims[1], frame_size,
                        "The width of H0 must be equal to frame_size.");
    }
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(bias_height, 1,
                        "The shape of Bias must be [1, frame_size * 3].");
      PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                        "The shape of Bias must be [1, frame_size * 3].");
    }
    ctx->SetOutputDim("BatchGate", input_dims);
    ctx->SetOutputDim("BatchResetHiddenPrev", {input_dims[0], frame_size});
    ctx->SetOutputDim("BatchHidden", {input_dims[0], frame_size});
    ctx->SetOutputDim("Hidden", {input_dims[0], frame_size});
    ctx->ShareLoD("Input", "Hidden");
  }
};

class GRUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(LoDTensor) The first input is a LodTensor, which supports "
             "variable-time length input sequence. The underlying tensor in "
             "this LoDTenosr is a matrix with shape (T X 3D), where, T is the "
             "total time steps in this mini-batch, D is the hidden size.");
    AddInput("H0",
             "(Tensor, optional) The initial hidden state is an optional "
             "input. This is a tensor with shape (N x D), where N is the "
             "batch size, D is the hidden size.")
        .AsDispensable();
    AddInput(
        "Weight",
        "(Tensor) The learnable hidden-hidden weight matrix with shape "
        "(D x 3D), where D is the hidden size. The elements continuous in "
        "memory can be divided into two parts. The first part are weights of "
        "the update gate and reset gate with shape (D x 2D), and the second "
        "part are weights of output candidate with shape (D x D).");
    AddInput("Bias",
             "(Tensor, optional) Bias vector with shape (1 x 3D) concating "
             "bias of the update gate, reset gate and output candidate.")
        .AsDispensable();
    AddOutput("BatchGate",
              "(LoDTensor) To compute with batches, sequence data will be "
              "reorganized into several successive batches each containing "
              "data from the same time step. The LoDTensor BatchGate contains "
              "the update gate, reset gate and output candidate values "
              "organized in batches. The LoD size is 2. The first LoD contains "
              "the batch offsets and the second LoD contains the indexes in "
              "the raw sequence data.")
        .AsIntermediate();
    AddOutput(
        "BatchResetHiddenPrev",
        "(LoDTensor) The reseted hidden state LoDTensor organized in batches. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.")
        .AsIntermediate();
    AddOutput(
        "BatchHidden",
        "(LoDTensor) The hidden state LoDTensor organized in batches.  "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.")
        .AsIntermediate();
    AddOutput(
        "Hidden",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddAttr<std::string>("activation",
                         "(string, default tanh) "
                         "The activation type used for output candidate {h}_t.")
        .SetDefault("tanh");
    AddAttr<std::string>(
        "gate_activation",
        "(string, default sigmoid) "
        "The activation type used in update gate and reset gate.")
        .SetDefault("sigmoid");
    AddAttr<bool>("is_reverse",
                  "(bool, defalut: False) "
                  "whether to compute reversed GRU.")
        .SetDefault(false);
    AddAttr<bool>("origin_mode",
                  "bool"
                  "use origin mode in article https://arxiv.org/abs/1412.3555")
        .SetDefault(false);
    AddComment(R"DOC(
GRU Operator implements part calculations of the complete GRU as following:

$$
update\_gate: u_t = actGate(xu_t + W_u * h_{t-1} + b_u) \\
reset\_gate: r_t = actGate(xr_t + W_r * h_{t-1} + b_r)  \\
output\_candidate: {h}_t = actNode(xc_t + W_c * dot(r_t, h_{t-1}) + b_c) \\
output: h_t = dot((1 - u_t), h_{t-1}) + dot(u_t, {h}_t)
$$

@note To implement the complete GRU, fully-connected operator must be used
before to feed xu, xr and xc as the Input of GRU operator.
)DOC");
  }
};

class GRUGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(%s) of GRUGradOp should not be null.", "Input");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(%s) of GRUGradOp should not be null.", "Weight");
    PADDLE_ENFORCE(ctx->HasInput("BatchGate"),
                   "Input(%s) of GRUGradOp should not be null.", "BatchGate");
    PADDLE_ENFORCE(ctx->HasInput("BatchResetHiddenPrev"),
                   "Input(%s) of GRUGradOp should not be null.",
                   "BatchResetHiddenPrev");
    PADDLE_ENFORCE(ctx->HasInput("BatchHidden"),
                   "Input(%s) of GRUOp should not be null.", "BatchHidden");
    PADDLE_ENFORCE(ctx->HasInput("Hidden"),
                   "Input(%s) of GRUGradOp should not be null.", "Hidden");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Hidden")),
                   "Input(%s@GRAD) of GRUGradOp should not be null.", "Hidden");
    auto input_dims = ctx->GetInputDim("Input");
    auto weight_dims = ctx->GetInputDim("Weight");
    int input_size = input_dims[1];
    int frame_size = weight_dims[0];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    PADDLE_ENFORCE_EQ(input_size, frame_size * 3,
                      "The input_size must be 3 times of frame_size in GRUOp.");
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    if (ctx->HasInput("H0")) {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(h0_dims[1], frame_size,
                        "The width of H0 must be equal to frame_size.");
      auto h0_grad_name = framework::GradVarName("H0");
      if (ctx->HasOutput(h0_grad_name))
        ctx->SetOutputDim(h0_grad_name, h0_dims);
    }
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(bias_height, 1,
                        "The shape of Bias must be [1, frame_size * 3].");
      PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                        "The shape of Bias must be [1, frame_size * 3].");
      auto bias_grad_name = framework::GradVarName("Bias");
      if (ctx->HasOutput(bias_grad_name))
        ctx->SetOutputDim(bias_grad_name, bias_dims);
    }
    auto input_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(input_grad_name))
      ctx->SetOutputDim(input_grad_name, input_dims);
    auto weight_grad_name = framework::GradVarName("Weight");
    if (ctx->HasOutput(weight_grad_name))
      ctx->SetOutputDim(weight_grad_name, weight_dims);
  }
};

template <typename T>
class GRUCPUKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    bool origin_mode = context.Attr<bool>("origin_mode");
    auto* input = context.Input<LoDTensor>("Input");
    auto* h0 = context.Input<Tensor>("H0");
    auto* weight = context.Input<Tensor>("Weight");
    const T* weight_data = weight->data<T>();
    auto* bias = context.Input<Tensor>("Bias");
    auto* batch_gate = context.Output<LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(context.GetPlace());
    auto* batch_reset_hidden_prev =
        context.Output<LoDTensor>("BatchResetHiddenPrev");
    batch_reset_hidden_prev->mutable_data<T>(context.GetPlace());
    auto* batch_hidden = context.Output<LoDTensor>("BatchHidden");
    batch_hidden->mutable_data<T>(context.GetPlace());
    auto* hidden = context.Output<LoDTensor>("Hidden");
    hidden->mutable_data<T>(context.GetPlace());

    auto hidden_dims = hidden->dims();

    bool is_reverse = context.Attr<bool>("is_reverse");
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    to_batch(dev_ctx, *input, batch_gate, true, is_reverse);

    if (bias) {
      math::RowwiseAdd<DeviceContext, T> add_bias;
      add_bias(dev_ctx, *batch_gate, *bias, batch_gate);
    }

    int frame_size = hidden_dims[1];
    math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);
    Tensor ordered_h0;

    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (h0) {
      // Since the batch computing for GRU reorders the input sequences
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<DeviceContext, T>(
          context.template device_context<DeviceContext>(), *h0, order,
          &ordered_h0, true);
      gru_value.prev_out_value = ordered_h0.data<T>();
    } else {
      gru_value.prev_out_value = nullptr;
    }
    auto batch_starts = batch_gate->lod()[0];
    size_t seq_len = batch_starts.size() - 1;
    auto active_node = math::detail::GetActivationType(
        context.Attr<std::string>("activation"));
    auto active_gate = math::detail::GetActivationType(
        context.Attr<std::string>("gate_activation"));

#ifdef PADDLE_WITH_MKLML
    // use MKL packed to speedup GEMM
    if (FLAGS_paddle_num_threads >= 4) {
      auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
      T* packed_gate = blas.GEMM_ALLOC(CblasBMatrix, 1 /*height of C*/,
                                       frame_size * 2 /*width of weight*/,
                                       frame_size /*height of height*/);
      PADDLE_ENFORCE(packed_gate);
      blas.GEMM_PACK(CblasBMatrix, CblasNoTrans, 1 /*cur bs?*/, frame_size * 2,
                     frame_size, T(1.0), gru_value.gate_weight, frame_size * 2,
                     packed_gate);
      T* packed_state = blas.GEMM_ALLOC(CblasBMatrix, 1 /*height of C*/,
                                        frame_size /*width of weight*/,
                                        frame_size /*height of height*/);
      PADDLE_ENFORCE(packed_state);
      blas.GEMM_PACK(CblasBMatrix, CblasNoTrans, 1 /*cur bs?*/, frame_size,
                     frame_size, T(1.0), gru_value.state_weight, frame_size,
                     packed_state);
      for (size_t n = 0; n < seq_len; n++) {
        int bstart = static_cast<int>(batch_starts[n]);
        int bend = static_cast<int>(batch_starts[n + 1]);
        int cur_batch_size = bend - bstart;

        Tensor gate_t = batch_gate->Slice(bstart, bend);
        Tensor reset_hidden_prev_t =
            batch_reset_hidden_prev->Slice(bstart, bend);
        Tensor hidden_t = batch_hidden->Slice(bstart, bend);
        gru_value.output_value = hidden_t.data<T>();
        gru_value.gate_value = gate_t.data<T>();
        gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

        if (gru_value.prev_out_value) {
          blas.GEMM_COMPUTE(
              CblasNoTrans, CblasPacked, cur_batch_size, frame_size * 2,
              frame_size, gru_value.prev_out_value, frame_size, packed_gate,
              frame_size * 2, T(1), gru_value.gate_value, frame_size * 3);
        }

        math::detail::forward_reset_output(
            math::detail::forward::gru_resetOutput<T>(), gru_value, frame_size,
            cur_batch_size, active_gate);

        if (gru_value.prev_out_value) {
          blas.GEMM_COMPUTE(
              CblasNoTrans, CblasPacked, cur_batch_size, frame_size, frame_size,
              gru_value.reset_output_value, frame_size, packed_state,
              frame_size, T(1), gru_value.gate_value + frame_size * 2,
              frame_size * 3);
        }

        math::detail::forward_final_output(
            math::detail::forward::gru_finalOutput<T>(), gru_value, frame_size,
            cur_batch_size, active_node, origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }

      blas.GEMM_FREE(packed_gate);
      blas.GEMM_FREE(packed_state);
    } else {
#endif
      for (size_t n = 0; n < seq_len; n++) {
        int bstart = static_cast<int>(batch_starts[n]);
        int bend = static_cast<int>(batch_starts[n + 1]);
        int cur_batch_size = bend - bstart;

        Tensor gate_t = batch_gate->Slice(bstart, bend);
        Tensor reset_hidden_prev_t =
            batch_reset_hidden_prev->Slice(bstart, bend);
        Tensor hidden_t = batch_hidden->Slice(bstart, bend);
        gru_value.output_value = hidden_t.data<T>();
        gru_value.gate_value = gate_t.data<T>();
        gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

        math::GRUUnitFunctor<DeviceContext, T>::compute(
            dev_ctx, gru_value, frame_size, cur_batch_size, active_node,
            active_gate, origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }
#ifdef PADDLE_WITH_MKLML
    }
#endif
    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(dev_ctx, *batch_hidden, hidden);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gru, ops::GRUOp, ops::GRUOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(gru_grad, ops::GRUGradOp);
REGISTER_OP_CPU_KERNEL(gru, ops::GRUCPUKernel<float>,
                       ops::GRUCPUKernel<double>);
REGISTER_OP_CPU_KERNEL(
    gru_grad, ops::GRUGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GRUGradKernel<paddle::platform::CPUDeviceContext, double>);
