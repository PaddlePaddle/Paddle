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
#include <memory>
#include <string>
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/detail/gru_cpu_kernel.h"
#include "paddle/phi/kernels/funcs/detail/gru_kernel.h"

DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace operators {

using framework::Tensor;

class GRUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "GRU");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "GRU");
    OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "GRU");
    bool is_test = ctx->Attrs().Get<bool>("is_test");
    if (!is_test) {
      OP_INOUT_CHECK(ctx->HasOutput("BatchGate"), "Output", "BatchGate", "GRU");
      OP_INOUT_CHECK(ctx->HasOutput("BatchResetHiddenPrev"), "Output",
                     "BatchResetHiddenPrev", "GRU");
      OP_INOUT_CHECK(ctx->HasOutput("BatchHidden"), "Output", "BatchHidden",
                     "GRU");
    }
    auto input_dims = ctx->GetInputDim("Input");
    auto weight_dims = ctx->GetInputDim("Weight");
    int input_size = input_dims[1];
    int frame_size = weight_dims[0];
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(input_size, frame_size * 3,
                        platform::errors::InvalidArgument(
                            "The second dimension of Input(Input) must be 3 "
                            "times of frame_size in GRUOp, but received %d "
                            "(Input) vs %d (frame_size).",
                            input_size, frame_size));
    }
    PADDLE_ENFORCE_EQ(
        weight_dims[1], frame_size * 3,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3], but received [%d, %d] (Weight) vs [%d, %d] (frame_size).",
            weight_dims[0], weight_dims[1], frame_size, frame_size * 3));
    if (ctx->HasInput("H0")) {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(
          h0_dims[1], frame_size,
          platform::errors::InvalidArgument(
              "The width of Input(H0) must be equal to frame_size, but "
              "received %d (width of H0) vs %d (frame_size).",
              h0_dims[1], frame_size));
    }
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(
          bias_height, 1,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height, bias_width, frame_size * 3));
      PADDLE_ENFORCE_EQ(
          bias_width, frame_size * 3,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height, bias_width, frame_size * 3));
    }
    if (!is_test) {
      ctx->SetOutputDim("BatchGate", input_dims);
      ctx->SetOutputDim("BatchResetHiddenPrev", {input_dims[0], frame_size});
      ctx->SetOutputDim("BatchHidden", {input_dims[0], frame_size});
    }
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
        .AsIntermediate()
        .AsExtra();
    AddOutput(
        "BatchResetHiddenPrev",
        "(LoDTensor) The reset hidden state LoDTensor organized in batches. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.")
        .AsIntermediate()
        .AsExtra();
    AddOutput(
        "BatchHidden",
        "(LoDTensor) The hidden state LoDTensor organized in batches.  "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.")
        .AsIntermediate()
        .AsExtra();
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
                  "(bool, default: False) "
                  "whether to compute reversed GRU.")
        .SetDefault(false);
    AddAttr<bool>("is_test", "True if in test phase.")
        .SetDefault(false)
        .AsExtra();
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
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "GRU@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "GRU@Grad");
    OP_INOUT_CHECK(ctx->HasInput("BatchGate"), "Input", "BatchGate",
                   "GRU@Grad");
    OP_INOUT_CHECK(ctx->HasInput("BatchResetHiddenPrev"), "Input",
                   "BatchResetHiddenPrev", "GRU@Grad");
    OP_INOUT_CHECK(ctx->HasInput("BatchHidden"), "Input", "BatchHidden",
                   "GRU@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Hidden"), "Input", "Hidden", "GRU@Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Hidden")), "Input",
                   framework::GradVarName("Hidden"), "GRU@Grad");

    auto input_dims = ctx->GetInputDim("Input");
    auto weight_dims = ctx->GetInputDim("Weight");
    int input_size = input_dims[1];
    int frame_size = weight_dims[0];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    PADDLE_ENFORCE_EQ(
        input_size, frame_size * 3,
        platform::errors::InvalidArgument(
            "The second dimension of Input(Input) must be 3 times of "
            "frame_size in GRUOp, but received %d (Input) vs %d (frame_size).",
            input_size, frame_size));
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3], but received [%d, %d] (Weight) vs [%d, %d] (frame_size).",
            weight_height, weight_width, frame_size, frame_size * 3));
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3], but received [%d, %d] (Weight) vs [%d, %d] (frame_size).",
            weight_height, weight_width, frame_size, frame_size * 3));
    if (ctx->HasInput("H0")) {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(
          h0_dims[1], frame_size,
          platform::errors::InvalidArgument(
              "The width of Input(H0) must be equal to frame_size, but "
              "received %d (width of H0) vs %d (frame_size).",
              h0_dims[1], frame_size));
      auto h0_grad_name = framework::GradVarName("H0");
      if (ctx->HasOutput(h0_grad_name))
        ctx->SetOutputDim(h0_grad_name, h0_dims);
    }
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(
          bias_height, 1,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height, bias_width, frame_size * 3));
      PADDLE_ENFORCE_EQ(
          bias_width, frame_size * 3,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height, bias_width, frame_size * 3));
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

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Hidden")),
                                   ctx.device_context());
  }
};

template <typename T>
class GRUCPUKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    using LodTensorPtr = LoDTensor*;
    bool is_test = context.Attr<bool>("is_test");

    bool origin_mode = context.Attr<bool>("origin_mode");
    auto* input = context.Input<LoDTensor>("Input");
    auto* h0 = context.Input<Tensor>("H0");
    auto* weight = context.Input<Tensor>("Weight");
    const T* weight_data = weight->data<T>();
    auto* bias = context.Input<Tensor>("Bias");
    auto* hidden = context.Output<LoDTensor>("Hidden");
    hidden->mutable_data<T>(context.GetPlace());

    auto input_dims = input->dims();
    auto hidden_dims = hidden->dims();

    LodTensorPtr batch_gate, batch_reset_hidden_prev, batch_hidden;
    LoDTensor batch_gate_tmp, batch_reset_hidden_prev_tmp, batch_hidden_tmp;
    if (is_test) {
      batch_gate = &batch_gate_tmp;
      batch_gate->Resize(input_dims);

      batch_reset_hidden_prev = &batch_reset_hidden_prev_tmp;
      batch_reset_hidden_prev->Resize(hidden_dims);

      batch_hidden = &batch_hidden_tmp;
      batch_hidden->Resize(hidden_dims);
    } else {
      batch_gate = context.Output<LoDTensor>("BatchGate");
      batch_hidden = context.Output<LoDTensor>("BatchHidden");
      batch_reset_hidden_prev =
          context.Output<LoDTensor>("BatchResetHiddenPrev");
    }
    batch_gate->mutable_data<T>(context.GetPlace());
    batch_reset_hidden_prev->mutable_data<T>(context.GetPlace());
    batch_hidden->mutable_data<T>(context.GetPlace());

    bool is_reverse = context.Attr<bool>("is_reverse");
    phi::funcs::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    to_batch(dev_ctx, *input, batch_gate, true, is_reverse);

    if (bias) {
      phi::funcs::RowwiseAdd<DeviceContext, T> add_bias;
      add_bias(dev_ctx, *batch_gate, *bias, batch_gate);
    }

    int frame_size = hidden_dims[1];
    phi::funcs::GRUMetaValue<T> gru_value;
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
    auto active_node = phi::funcs::detail::GetActivationType(
        context.Attr<std::string>("activation"));
    auto active_gate = phi::funcs::detail::GetActivationType(
        context.Attr<std::string>("gate_activation"));

#ifdef PADDLE_WITH_MKLML
    // use MKL packed to speedup GEMM
    if (FLAGS_paddle_num_threads >= 4) {
      auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);
      T* packed_gate = blas.GEMM_ALLOC(CblasBMatrix, 1 /*height of C*/,
                                       frame_size * 2 /*width of weight*/,
                                       frame_size /*height of height*/);
      PADDLE_ENFORCE_NOT_NULL(
          packed_gate, platform::errors::NotFound(
                           "The caculation result of packed_gate by "
                           "GEMM_ALLOC should not be null when using MKL."));
      blas.GEMM_PACK(CblasBMatrix, CblasNoTrans, 1 /*cur bs?*/, frame_size * 2,
                     frame_size, T(1.0), gru_value.gate_weight, frame_size * 2,
                     packed_gate);
      T* packed_state = blas.GEMM_ALLOC(CblasBMatrix, 1 /*height of C*/,
                                        frame_size /*width of weight*/,
                                        frame_size /*height of height*/);
      PADDLE_ENFORCE_NOT_NULL(
          packed_state, platform::errors::NotFound(
                            "The caculation result of packed_state by "
                            "GEMM_ALLOC should not be null when using MKL."));
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

        phi::funcs::detail::forward_reset_output(
            phi::funcs::detail::forward::gru_resetOutput<T>(), gru_value,
            frame_size, cur_batch_size, active_gate);

        if (gru_value.prev_out_value) {
          blas.GEMM_COMPUTE(
              CblasNoTrans, CblasPacked, cur_batch_size, frame_size, frame_size,
              gru_value.reset_output_value, frame_size, packed_state,
              frame_size, T(1), gru_value.gate_value + frame_size * 2,
              frame_size * 3);
        }

        phi::funcs::detail::forward_final_output(
            phi::funcs::detail::forward::gru_finalOutput<T>(), gru_value,
            frame_size, cur_batch_size, active_node, origin_mode);

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

        phi::funcs::GRUUnitFunctor<DeviceContext, T>::compute(
            dev_ctx, gru_value, frame_size, cur_batch_size, active_node,
            active_gate, origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }
#ifdef PADDLE_WITH_MKLML
    }
#endif
    phi::funcs::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(dev_ctx, *batch_hidden, hidden);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

template <typename T>
class GRUGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("gru_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput("H0", this->Input("H0"));
    grad_op->SetInput("Bias", this->Input("Bias"));
    grad_op->SetInput("Weight", this->Input("Weight"));

    grad_op->SetInput("BatchGate", this->Output("BatchGate"));
    grad_op->SetInput("BatchResetHiddenPrev",
                      this->Output("BatchResetHiddenPrev"));
    grad_op->SetInput("BatchHidden", this->Output("BatchHidden"));
    grad_op->SetInput("Hidden", this->Output("Hidden"));

    grad_op->SetInput(framework::GradVarName("Hidden"),
                      this->OutputGrad("Hidden"));

    grad_op->SetOutput(framework::GradVarName("H0"), this->InputGrad("H0"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetOutput(framework::GradVarName("Weight"),
                       this->InputGrad("Weight"));
    grad_op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));

    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GRUGradOpNoNeedBufferVarInferer, "Input",
                                    "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gru, ops::GRUOp, ops::GRUOpMaker,
                  ops::GRUGradOpMaker<paddle::framework::OpDesc>,
                  ops::GRUGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(gru_grad, ops::GRUGradOp,
                  ops::GRUGradOpNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(gru, ops::GRUCPUKernel<float>,
                       ops::GRUCPUKernel<double>);
REGISTER_OP_CPU_KERNEL(
    gru_grad, ops::GRUGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GRUGradKernel<paddle::platform::CPUDeviceContext, double>);
