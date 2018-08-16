/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fusion_gru_op.h"
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/detail/gru_cpu_kernel.h"
#include "paddle/fluid/operators/math/detail/gru_kernel.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/operators/math/gru_compute.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence2batch.h"

namespace paddle {
namespace operators {

void FusionGRUOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of GRU should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("WeightX"),
                 "Input(WeightX) of GRU should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("WeightH"),
                 "Input(WeightH) of GRU should not be null.");

  PADDLE_ENFORCE(ctx->HasOutput("XX"), "Output(XX) of GRU should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchedGate"),
                 "Output(BatchedGate) of GRU should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchResetHiddenPrev"),
                 "Output(BatchResetHiddenPrev) of GRU should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchedHidden"),
                 "Output(BatchedHidden) of GRU should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                 "Output(Hidden) of GRU should not be null.");

  auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank must be 2.");

  auto wx_dims = ctx->GetInputDim("WeightX");
  PADDLE_ENFORCE_EQ(wx_dims.size(), 2,
                    "The rank of Input(WeightX) should be 2.");
  PADDLE_ENFORCE_EQ(wx_dims[0], x_dims[1],
                    "The first dimension of Input(WeightX) "
                    "should be %d.",
                    x_dims[1]);

  int frame_size = wx_dims[1] / 3;
  auto wh_dims = ctx->GetInputDim("WeightH");
  PADDLE_ENFORCE_EQ(wh_dims.size(), 2,
                    "The rank of Input(WeightH) should be 2.");
  PADDLE_ENFORCE_EQ(wh_dims[0], frame_size,
                    "The first dimension of Input(WeightH) "
                    "should be %d.",
                    frame_size);
  PADDLE_ENFORCE_EQ(wh_dims[1], 3 * frame_size,
                    "The second dimension of Input(WeightH) "
                    "should be 3 * %d.",
                    frame_size);

  if (ctx->HasInput("H0")) {
    auto h0_dims = ctx->GetInputDim("H0");
    PADDLE_ENFORCE_EQ(h0_dims[1], frame_size,
                      "The width of H0 must be equal to frame_size.");
  }
  if (ctx->HasInput("Bias")) {
    auto b_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(b_dims.size(), 2, "The rank of Input(Bias) should be 2.");
    PADDLE_ENFORCE_EQ(b_dims[0], 1,
                      "The first dimension of Input(Bias) should be 1.");
    PADDLE_ENFORCE_EQ(b_dims[1], frame_size * 3,
                      "The shape of Bias must be [1, frame_size * 3].");
  }
  framework::DDim out_dims({x_dims[0], frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("BatchedGate", {x_dims[0], wx_dims[1]});
  ctx->SetOutputDim("BatchedHidden", out_dims);
  ctx->SetOutputDim("BatchResetHiddenPrev", out_dims);
  ctx->ShareLoD("X", "Hidden");

  int xx_width = x_dims[1] > wx_dims[1] ? wx_dims[1] : x_dims[1];
  ctx->SetOutputDim("XX", {x_dims[0], xx_width});
  ctx->ShareLoD("X", "XX");
}

framework::OpKernelType FusionGRUOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
      ctx.device_context());
}

void FusionGRUOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("H0",
           "(Tensor, optional) The initial hidden state is an optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size, D is the hidden size.")
      .AsDispensable();
  AddInput("WeightX",
           "(Tensor) The FC weight with shape (M x 3D),"
           "where M is the dim size of x, D is the hidden size. ");
  AddInput("WeightH",
           "(Tensor) (D x 3D) Same as GRUOp, where D is the hidden size. ");
  AddInput("Bias",
           "(Tensor, optional) (1 x 3D)."
           "Almost same as GRUOp."
           "Note: if have FC bias it should be added on this bias.")
      .AsDispensable();
  AddOutput("XX",
            "(LoDTensor) the result after X * WeightX (size is T x 4D)"
            " or batched_X (size is T x M), this will be automatically chosen,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size, M is the dim size of x input.")
      .AsIntermediate();
  AddOutput("BatchedGate", "(LoDTensor) Same as GRUOp").AsIntermediate();
  AddOutput("BatchResetHiddenPrev", "(LoDTensor) (T x 3D) Same as GRUOp.")
      .AsIntermediate();
  AddOutput("BatchedHidden", "(LoDTensor) (T X D) Same as GRUOp.")
      .AsIntermediate();
  AddOutput("Hidden", "(LoDTensor) (T x D) Same as GRUOp");
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
  AddComment(R"DOC(
The Fusion complete GRU Operator.
This operator fuse the fully-connected operator into GRU, 
more details can refer to GRU op.
)DOC");
}

template <typename DeviceContext, typename T>
inline void ReorderInitState(const DeviceContext& ctx,
                             const framework::Tensor& src,
                             framework::Vector<size_t> index_lod,
                             framework::Tensor* dst, bool indexed_src) {
  math::CopyMatrixRowsFunctor<DeviceContext, T> row_shuffle;
  dst->mutable_data<T>(src.dims(), ctx.GetPlace());
  row_shuffle(ctx, src, index_lod, dst, indexed_src);
}

template <typename DeviceContext, typename T>
class FusionGRUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* wx = ctx.Input<Tensor>("WeightX");
    auto* wh = ctx.Input<Tensor>("WeightH");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* h0 = ctx.Input<Tensor>("H0");

    auto* xx = ctx.Output<LoDTensor>("XX");
    auto* batched_gate = ctx.Output<LoDTensor>("BatchedGate");
    auto* batch_reset_hidden_prev =
        ctx.Output<LoDTensor>("BatchResetHiddenPrev");
    auto* batch_hidden = ctx.Output<LoDTensor>("BatchedHidden");
    auto* hidden_out = ctx.Output<LoDTensor>("Hidden");
    bool is_reverse = ctx.Attr<bool>("is_reverse");

    T* xx_data = xx->mutable_data<T>(ctx.GetPlace());
    T* batched_gate_data = batched_gate->mutable_data<T>(ctx.GetPlace());
    batch_reset_hidden_prev->mutable_data<T>(ctx.GetPlace());
    batch_hidden->mutable_data<T>(ctx.GetPlace());
    hidden_out->mutable_data<T>(ctx.GetPlace());

    const T* x_data = x->data<T>();
    const T* wx_data = wx->data<T>();
    const T* wh_data = wh->data<T>();
    auto x_dims = x->dims();
    auto wx_dims = wx->dims();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    if (x_dims[1] > wx_dims[1]) {
      math::FCCompute<DeviceContext, T>(blas, x_dims[0], wx_dims[1], x_dims[1],
                                        x_data, wx_data, xx_data,
                                        bias ? bias->data<T>() : NULL);
      to_batch(dev_ctx, *xx, batched_gate, true, is_reverse);
    } else {
      to_batch(dev_ctx, *x, xx, true, is_reverse);
      batched_gate->set_lod(xx->lod());
      math::FCCompute<DeviceContext, T>(blas, x_dims[0], wx_dims[1], x_dims[1],
                                        xx_data, wx_data, batched_gate_data,
                                        bias ? bias->data<T>() : NULL);
    }

    int frame_size = static_cast<int>(wx_dims[1] / 3);
    math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(wh_data);
    gru_value.state_weight =
        const_cast<T*>(wh_data + 2 * frame_size * frame_size);
    Tensor ordered_h0;

    framework::Vector<size_t> order(batched_gate->lod()[2]);

    if (h0) {
      ReorderInitState<DeviceContext, T>(
          ctx.template device_context<DeviceContext>(), *h0, order, &ordered_h0,
          true);
      gru_value.prev_out_value = ordered_h0.data<T>();
    } else {
      gru_value.prev_out_value = nullptr;
    }
    auto batch_starts = batched_gate->lod()[0];
    size_t seq_len = batch_starts.size() - 1;
    auto active_node =
        math::detail::GetActivationType(ctx.Attr<std::string>("activation"));
    auto active_gate = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));

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

        Tensor gate_t = batched_gate->Slice(bstart, bend);
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
            cur_batch_size, active_node);

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

        Tensor gate_t = batched_gate->Slice(bstart, bend);
        Tensor reset_hidden_prev_t =
            batch_reset_hidden_prev->Slice(bstart, bend);
        Tensor hidden_t = batch_hidden->Slice(bstart, bend);
        gru_value.output_value = hidden_t.data<T>();
        gru_value.gate_value = gate_t.data<T>();
        gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

        math::GRUUnitFunctor<DeviceContext, T>::compute(
            dev_ctx, gru_value, frame_size, cur_batch_size, active_node,
            active_gate);

        gru_value.prev_out_value = gru_value.output_value;
      }
#ifdef PADDLE_WITH_MKLML
    }
#endif
    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden->set_lod(batched_gate->lod());
    to_seq(dev_ctx, *batch_hidden, hidden_out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_gru, ops::FusionGRUOp, ops::FusionGRUOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OP_CPU_KERNEL(
    fusion_gru, ops::FusionGRUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FusionGRUKernel<paddle::platform::CPUDeviceContext, double>);
