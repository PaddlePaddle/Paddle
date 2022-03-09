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

#include "paddle/fluid/operators/fused/fusion_gru_op.h"
#include <cstring>  // for memcpy
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

void FusionGRUOp::InferShape(framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fusion_gru");
  OP_INOUT_CHECK(ctx->HasInput("WeightX"), "Input", "WeightX", "fusion_gru");
  OP_INOUT_CHECK(ctx->HasInput("WeightH"), "Input", "WeightH", "fusion_gru");
  OP_INOUT_CHECK(ctx->HasOutput("XX"), "Output", "XX", "fusion_gru");
  OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "fusion_gru");
  auto x_dims = ctx->GetInputDim("X");
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                        ? phi::flatten_to_2d(x_dims, 1)
                        : x_dims;
  PADDLE_ENFORCE_EQ(
      x_mat_dims.size(), 2,
      platform::errors::InvalidArgument("The size of input X dims should be 2, "
                                        "or 3 with second dimension equal to "
                                        "1, but now Input X dim is:[%s] ",
                                        x_dims));

  auto wx_dims = ctx->GetInputDim("WeightX");
  PADDLE_ENFORCE_EQ(wx_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(WeightX) should be 2, but received "
                        "WeightX dim size is:%d, WeightX dim is:[%s] ",
                        wx_dims.size(), wx_dims));
  PADDLE_ENFORCE_EQ(
      wx_dims[0], x_mat_dims[1],
      platform::errors::InvalidArgument(
          "The first dimension of flattened WeightX"
          "should equal to last dimension of flattened input X, but "
          "received fattened WeightX dimension is:%d, flattened X dimension "
          "is:%d",
          wx_dims[0], x_mat_dims[1]));

  int frame_size = wx_dims[1] / 3;
  auto wh_dims = ctx->GetInputDim("WeightH");

  PADDLE_ENFORCE_EQ(wh_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(WeightH) should be 2, but received "
                        "WeightH dim size is:%d, WeightH dim is:[%s]",
                        wh_dims.size(), wh_dims));
  PADDLE_ENFORCE_EQ(wh_dims[0], frame_size,
                    platform::errors::InvalidArgument(
                        "The first dimension of WeightH "
                        "should equal to frame_size, but received WeightH's "
                        "first dimension is: "
                        "%d, frame size is:%d",
                        wh_dims[0], frame_size));
  PADDLE_ENFORCE_EQ(wh_dims[1], 3 * frame_size,
                    platform::errors::InvalidArgument(
                        "The second dimension of Input(WeightH) "
                        "should equal to 3 * frame_size, but received WeightH "
                        "is:%d, frame size is:%d",
                        wh_dims[1], frame_size));

  if (ctx->HasInput("H0")) {
    auto h0_dims = ctx->GetInputDim("H0");
    PADDLE_ENFORCE_EQ(h0_dims[1], frame_size,
                      platform::errors::InvalidArgument(
                          "The width of H0 must be equal to frame_size, but "
                          "receiced the width of H0 is:%d, frame size is:%d",
                          h0_dims[1], frame_size));
  }
  if (ctx->HasInput("Bias")) {
    auto b_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(b_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The rank of Input(Bias) should be 2, but received "
                          "Bias rank is:%d, Bias dim is:[%s]",
                          b_dims.size(), b_dims));
    PADDLE_ENFORCE_EQ(b_dims[0], 1,
                      platform::errors::InvalidArgument(
                          "The first dimension of Input(Bias) should be 1, but "
                          "received Bias first dim is:%d, Bias dim is:[%s]",
                          b_dims[0], b_dims));
    PADDLE_ENFORCE_EQ(b_dims[1], frame_size * 3,
                      platform::errors::InvalidArgument(
                          "The shape of Bias must be [1, frame_size * 3], but "
                          "received bias dim is:[%s], frame size is:%d",
                          b_dims, frame_size));
  }
  framework::DDim out_dims({x_mat_dims[0], frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->ShareLoD("X", "Hidden");
  int xx_width;
  if (ctx->Attrs().Get<bool>("use_seq")) {
    xx_width = wx_dims[1];
  } else {
    xx_width = x_mat_dims[1] > wx_dims[1] ? wx_dims[1] : x_mat_dims[1];
    OP_INOUT_CHECK(ctx->HasOutput("ReorderedH0"), "Output", "ReorderedH0",
                   "fusion_gru");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedInput"), "Output", "BatchedInput",
                   "fusion_gru");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedOut"), "Output", "BatchedOut",
                   "fusion_gru");
    ctx->SetOutputDim("BatchedInput", {x_mat_dims[0], wx_dims[1]});
    ctx->SetOutputDim("BatchedOut", out_dims);
  }
  ctx->SetOutputDim("XX", {x_mat_dims[0], xx_width});
  ctx->ShareLoD("X", "XX");
}

framework::OpKernelType FusionGRUOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
  if (this->CanMKLDNNBeUsed(ctx, data_type)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif
  return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
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
           "(Tensor) (D x 3D) Same as GRUOp, where D is the hidden size. "
           "This weight is not exactly D x 3D as: {W_update, W_reset, W_state}"
           "Acutally they are D x 2D and D x D two part weights."
           "{W_update, W_reset; W_state}"
           "{D x (D + D); D x D}");
  AddInput("Bias",
           "(Tensor, optional) (1 x 3D)."
           "Almost same as GRUOp."
           "Note: if have FC bias it should be added on this bias.")
      .AsDispensable();
  AddOutput("ReorderedH0", "(Tensor) (N x D), which N is the min-batch size.")
      .AsIntermediate();
  AddOutput("XX",
            "(LoDTensor) the result after X * WeightX (size is T x 3D)"
            " or batched_X (size is T x M), this will be automatically chosen,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size, M is the dim size of x input.")
      .AsIntermediate();
  AddOutput("BatchedInput",
            "(LoDTensor) This is the batched result of input X"
            "or the batched result after fc, shape (T x 3D)")
      .AsIntermediate();
  AddOutput("BatchedOut", "(LoDTensor) (T X D) save batched hidden.")
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
                "(bool, default: False) "
                "whether to compute reversed GRU.")
      .SetDefault(false);
  AddAttr<bool>("use_seq",
                "(bool, default: True) "
                "whether to use seq mode to compute GRU.")
      .SetDefault(true);
  AddAttr<bool>("origin_mode",
                "bool"
                "use origin mode in article https://arxiv.org/abs/1412.3555")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<std::string>(
      "mkldnn_data_type",
      "(string, default \"float32\"). Data type of mkldnn kernel")
      .SetDefault("float32")
      .InEnum({"float32", "int8", "bfloat16"});
  AddAttr<float>("Scale_data",
                 "Scale to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(1.0f);
  AddAttr<float>("Shift_data",
                 "Shift to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(0.0f);
  AddAttr<std::vector<float>>("Scale_weights",
                              "Scale_weights to be used for int8 weights data."
                              "Only used with MKL-DNN INT8.")
      .SetDefault({1.0f});
  AddAttr<bool>("force_fp32_output",
                "(bool, default false) Force INT8 kernel output FP32, only "
                "used in MKL-DNN INT8")
      .SetDefault(false);
  AddComment(R"DOC(
The Fusion complete GRU Operator.
This operator fuse the fully-connected operator into GRU, 
more details can refer to GRU op.
)DOC");
}

template <typename T>
class FusionGRUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    if (ctx.Attr<bool>("use_seq")) {
      SeqCompute(ctx);
    } else {
      BatchCompute(ctx);
    }
  }

#define INIT_BASE_DEFINES                                  \
  auto* x = ctx.Input<LoDTensor>("X");                     \
  auto* wh = ctx.Input<Tensor>("WeightH");                 \
  auto* xx = ctx.Output<LoDTensor>("XX");                  \
  auto x_lod = x->lod();                                   \
  auto x_dims = x->dims(); /* T x M*/                      \
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1) \
                        ? phi::flatten_to_2d(x_dims, 1)    \
                        : x_dims;                          \
  auto wh_dims = wh->dims(); /* D x 3D*/                   \
  const int total_T = x_mat_dims[0];                       \
  const int D3 = wh_dims[1]

#define INIT_OTHER_DEFINES                                                   \
  auto* h0 = ctx.Input<Tensor>("H0");                                        \
  auto* wx = ctx.Input<Tensor>("WeightX");                                   \
  auto* bias = ctx.Input<Tensor>("Bias");                                    \
  auto* hidden_out = ctx.Output<LoDTensor>("Hidden");                        \
  bool is_reverse = ctx.Attr<bool>("is_reverse");                            \
  const int M = x_mat_dims[1];                                               \
  const int D = wh_dims[0];                                                  \
  const int D2 = D * 2;                                                      \
  const jit::gru_attr_t attr(                                                \
      D, jit::to_kerneltype(ctx.Attr<std::string>("gate_activation")),       \
      jit::to_kerneltype(ctx.Attr<std::string>("activation")));              \
  jit::gru_t one_step;                                                       \
  auto ComputeH1 =                                                           \
      jit::KernelFuncs<jit::GRUH1Tuple<T>, platform::CPUPlace>::Cache().At(  \
          attr);                                                             \
  auto ComputeHtPart1 =                                                      \
      jit::KernelFuncs<jit::GRUHtPart1Tuple<T>, platform::CPUPlace>::Cache() \
          .At(attr);                                                         \
  auto ComputeHtPart2 =                                                      \
      jit::KernelFuncs<jit::GRUHtPart2Tuple<T>, platform::CPUPlace>::Cache() \
          .At(attr);                                                         \
  const T* x_data = x->data<T>();                                            \
  const T* wx_data = wx->data<T>();                                          \
  const T* wh_data = wh->data<T>();                                          \
  auto place = ctx.GetPlace();                                               \
  T* xx_data = xx->mutable_data<T>(place)

  void SeqCompute(const framework::ExecutionContext& ctx) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    INIT_BASE_DEFINES;
    INIT_OTHER_DEFINES;
    const int N = x_lod[0].size() - 1;
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* wh_state_data = wh_data + D * D2;
    T* hidden_out_data = hidden_out->mutable_data<T>(place);
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::FCFunctor<DeviceContext, T> fc;
    fc(dev_ctx, total_T, D3, M, x_data, wx_data, xx_data,
       bias ? bias->data<T>() : nullptr);

    int xx_offset = D3;
    int gate_offset = D;
    if (is_reverse) {
      const int offset = (total_T - 1) * D;
      xx_data = xx_data + offset * 3;
      hidden_out_data = hidden_out_data + offset;
      xx_offset = -D3;
      gate_offset = -D;
    }
    auto move_step = [&]() {
      xx_data = xx_data + xx_offset;
      hidden_out_data = hidden_out_data + gate_offset;
    };
    for (int i = 0; i < N; ++i) {
      int bid = is_reverse ? N - 1 - i : i;
      int seq_len = x_lod[0][bid + 1] - x_lod[0][bid];
      const T* prev_hidden_data = nullptr;
      int tstart = 0;
      if (h0_data) {
        prev_hidden_data = h0_data + bid * D;
      } else {
        one_step.gates = xx_data;
        one_step.ht = hidden_out_data;
        ComputeH1(&one_step, &attr);
        prev_hidden_data = hidden_out_data;
        tstart = 1;
        move_step();
      }
      for (int step = tstart; step < seq_len; ++step) {
        // gemm prev * (Wu + Wr)
        blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D2, D, static_cast<T>(1),
                  prev_hidden_data, D, wh_data, D2, static_cast<T>(1), xx_data,
                  D3);
        one_step.gates = xx_data;
        one_step.ht_1 = prev_hidden_data;
        one_step.ht = hidden_out_data;
        ComputeHtPart1(&one_step, &attr);
        // gemm rt * Ws
        blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D, D, static_cast<T>(1),
                  hidden_out_data, D, wh_state_data, D, static_cast<T>(1),
                  xx_data + D2, D3);
        ComputeHtPart2(&one_step, &attr);
        // save prev
        prev_hidden_data = hidden_out_data;
        move_step();
      }
    }
  }

  void BatchCompute(const framework::ExecutionContext& ctx) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    INIT_BASE_DEFINES;
    if (x_lod[0].size() == 2) {
      xx->Resize({total_T, D3});
      SeqCompute(ctx);
      return;
    }
    INIT_OTHER_DEFINES;
    auto* reordered_h0 = ctx.Output<Tensor>("ReorderedH0");
    auto* batched_input = ctx.Output<LoDTensor>("BatchedInput");
    auto* batched_out = ctx.Output<LoDTensor>("BatchedOut");
    T* batched_input_data = batched_input->mutable_data<T>(place);
    T* batched_out_data = batched_out->mutable_data<T>(place);
    hidden_out->mutable_data<T>(place);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);
    phi::funcs::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;

    math::FCFunctor<DeviceContext, T> fc;
    if (M > D3) {
      fc(dev_ctx, total_T, D3, M, x_data, wx_data, xx_data,
         bias ? bias->data<T>() : nullptr);
      to_batch(dev_ctx, *xx, batched_input, true, is_reverse);
    } else {
      to_batch(dev_ctx, *x, xx, true, is_reverse);
      batched_input->set_lod(xx->lod());
      fc(dev_ctx, total_T, D3, M, xx_data, wx_data, batched_input_data,
         bias ? bias->data<T>() : nullptr);
    }

    auto batched_lod = batched_input->lod();
    const auto& seq_order = batched_lod[2];
    const int max_bs = seq_order.size();
    reordered_h0->Resize({max_bs, D});

    int tstart = 0;
    T* prev_hidden_data = nullptr;
    if (h0) {
      // reorder h0
      T* reordered_h0_data = reordered_h0->mutable_data<T>(place);
      const T* h0_data = h0->data<T>();
      prev_hidden_data = reordered_h0_data;
      size_t sz = sizeof(T) * D;
      for (int i = 0; i < max_bs; ++i) {
        std::memcpy(reordered_h0_data, h0_data + seq_order[i] * D, sz);
        reordered_h0_data += D;
      }
    } else {
      // compute without h0
      T* cur_in_data = batched_input_data;
      T* cur_out_data = batched_out_data;
      // W: {W_update, W_reset; W_state}
      for (int i = 0; i < max_bs; ++i) {
        one_step.gates = cur_in_data;
        one_step.ht = cur_out_data;
        ComputeH1(&one_step, &attr);
        // add offset
        cur_in_data += D3;
        cur_out_data += D;
      }
      tstart = 1;
      prev_hidden_data = batched_out_data;
    }
    // Then start from next
    const T* wh_state_data = wh_data + D * D2;
    const auto& batch_starts = batched_lod[0];
    const int max_seq_len = batch_starts.size() - 1;
    batched_input_data = batched_input_data + tstart * max_bs * D3;
    batched_out_data = batched_out_data + tstart * max_bs * D;
    for (int step = tstart; step < max_seq_len; ++step) {
      const int cur_bs = batch_starts[step + 1] - batch_starts[step];
      // gemm prev * (Wu + Wr)
      blas.GEMM(CblasNoTrans, CblasNoTrans, cur_bs, D2, D, static_cast<T>(1),
                prev_hidden_data, D, wh_data, D2, static_cast<T>(1),
                batched_input_data, D3);

      T* cur_batched_data = batched_input_data;
      T* cur_out_data = batched_out_data;
      T* cur_prev_hidden_data = prev_hidden_data;
      for (int i = 0; i < cur_bs; ++i) {
        one_step.gates = cur_batched_data;
        one_step.ht_1 = cur_prev_hidden_data;
        one_step.ht = cur_out_data;
        ComputeHtPart1(&one_step, &attr);

        cur_batched_data += D3;
        cur_prev_hidden_data += D;
        cur_out_data += D;
      }

      cur_batched_data = batched_input_data;
      cur_out_data = batched_out_data;
      blas.GEMM(CblasNoTrans, CblasNoTrans, cur_bs, D, D, static_cast<T>(1),
                cur_out_data, D, wh_state_data, D, static_cast<T>(1),
                cur_batched_data + D2, D3);

      cur_prev_hidden_data = prev_hidden_data;
      for (int i = 0; i < cur_bs; ++i) {
        one_step.gates = cur_batched_data;
        one_step.ht_1 = cur_prev_hidden_data;
        one_step.ht = cur_out_data;
        ComputeHtPart2(&one_step, &attr);
        cur_batched_data += D3;
        cur_prev_hidden_data += D;
        cur_out_data += D;
      }
      prev_hidden_data = batched_out_data;
      batched_out_data = cur_out_data;
      batched_input_data = cur_batched_data;
    }

    phi::funcs::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batched_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_out, hidden_out);
  }
#undef INIT_OTHER_DEFINES
#undef INIT_BASE_DEFINES
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_gru, ops::FusionGRUOp, ops::FusionGRUOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_gru, ops::FusionGRUKernel<float>,
                       ops::FusionGRUKernel<double>);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(fusion_gru)
    .AddCheckpoint(
        R"ROC(Upgrade fusion_gru add a new attribute [Scale_weights])ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_weights",
            "The added attribute 'Scale_weights' is not yet "
            "registered.",
            std::vector<float>{1.0f}));
