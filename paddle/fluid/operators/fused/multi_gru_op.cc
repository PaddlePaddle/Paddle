/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/multi_gru_op.h"
// #include "paddle/fluid/operators/fused/fusion_gru_op.h"
#include <cstring>  // for memcpy
#include <string>
#include <vector>

#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace paddle {
namespace operators {

void MultiGRUOp::InferShape(framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "multi_gru");
  OP_INOUT_CHECK(ctx->HasInputs("WeightX"), "Input", "WeightX", "multi_gru");
  OP_INOUT_CHECK(ctx->HasInputs("WeightH"), "Input", "WeightH", "multi_gru");
  OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "multi_gru");
  auto x_dims = ctx->GetInputDim("X");
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                        ? phi::flatten_to_2d(x_dims, 1)
                        : x_dims;
  PADDLE_ENFORCE_EQ(
      x_mat_dims.size(),
      2,
      platform::errors::InvalidArgument("The size of input X dims should be 2, "
                                        "or 3 with second dimension equal to "
                                        "1, but now Input X dim is:[%s] ",
                                        x_dims));

  auto layers = ctx->Attrs().Get<int>("layers");
  auto wx_dims = ctx->GetInputsDim("WeightX");
  for (int i : {0, 1}) {
    PADDLE_ENFORCE_EQ(
        wx_dims[i][0],
        x_mat_dims[1],
        platform::errors::InvalidArgument(
            "The first dimension of flattened WeightX #%d"
            "should equal to last dimension of flattened input X, but "
            "received fattened WeightX dimension is:%d, flattened X dimension "
            "is:%d",
            i,
            wx_dims[i][0],
            x_mat_dims[1]));
  }

  auto wh_dims = ctx->GetInputsDim("WeightH");
  for (int i = 0; i < 2 * layers; ++i) {
    PADDLE_ENFORCE_EQ(wx_dims[i].size(),
                      2,
                      platform::errors::InvalidArgument(
                          "The rank of WeightX #%d should be 2, but received "
                          "WeightX dim size is:%d, WeightX dim is:[%s] ",
                          i,
                          wx_dims[i].size(),
                          wx_dims[i]));
    PADDLE_ENFORCE_EQ(wh_dims[i].size(),
                      2,
                      platform::errors::InvalidArgument(
                          "The rank of WeightH #%d should be 2, but received "
                          "WeightH dim size is:%d, WeightH dim is:[%s] ",
                          i,
                          wh_dims[i].size(),
                          wh_dims[i]));
    int frame_size = wh_dims[i][0];
    PADDLE_ENFORCE_EQ(
        wh_dims[i][1],
        3 * frame_size,
        platform::errors::InvalidArgument(
            "The second dimension of WeightH #%d "
            "should equal to 3 * frame_size, but received WeightH's "
            "second dimension is: %d, frame size is:%d",
            i,
            wh_dims[1],
            frame_size));
    PADDLE_ENFORCE_EQ(
        wx_dims[i][1],
        3 * frame_size,
        platform::errors::InvalidArgument(
            "The second dimension of WeightX #%d "
            "should equal to 3 * frame_size, but received WeightX's "
            "second dimension is: %d, frame size is:%d",
            i,
            wx_dims[i][1],
            frame_size));
  }

  if (ctx->HasInputs("Bias")) {
    auto b_dims = ctx->GetInputsDim("Bias");
    for (int i = 0; i < 2 * layers; ++i) {
      int frame_size = wh_dims[i][0];
      PADDLE_ENFORCE_EQ(b_dims[i].size(),
                        2,
                        platform::errors::InvalidArgument(
                            "The rank of Bias #%d should be 2, but received "
                            "Bias rank is:%d, Bias dim is:[%s]",
                            i,
                            b_dims[i].size(),
                            b_dims[i]));
      PADDLE_ENFORCE_EQ(b_dims[i][0],
                        1,
                        platform::errors::InvalidArgument(
                            "The first dimension of Bias #%d should be 1, but "
                            "received Bias first dim is:%d, Bias dim is:[%s]",
                            i,
                            b_dims[i][0],
                            b_dims[i]));
      PADDLE_ENFORCE_EQ(
          b_dims[i][1],
          frame_size * 3,
          platform::errors::InvalidArgument(
              "The shape of Bias #%d must be [1, frame_size * 3], but "
              "received bias dim is:[%s], frame size is:%d",
              i,
              b_dims[i],
              frame_size));
    }
  }

  int last_frame_size = wh_dims.back()[0];
  framework::DDim out_dims({x_mat_dims[0], 2 * last_frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->ShareLoD("X", "Hidden");
}

framework::OpKernelType MultiGRUOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"),
      ctx.GetPlace(),
      phi::DataLayout::ONEDNN,
      framework::LibraryType::kMKLDNN);
}

void MultiGRUOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is an LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("WeightX",
           "(MultiTensor) The FC weight with shape (M x 3D),"
           "where M is the dim size of x, D is the hidden size. ")
      .AsDuplicable();
  AddInput("WeightH",
           "(MultiTensor) (D x 3D) Same as GRUOp, where D is the hidden size. "
           "This weight is not exactly D x 3D as: {W_update, W_reset, W_state}"
           "Acutally they are D x 2D and D x D two part weights."
           "{W_update, W_reset; W_state}"
           "{D x (D + D); D x D}")
      .AsDuplicable();
  AddInput("Bias",
           "(MultiTensor, optional) (1 x 3D)."
           "Almost same as GRUOp."
           "Note: if have FC bias it should be added on this bias.")
      .AsDuplicable()
      .AsDispensable();
  AddInput(
      "Scale_weights",
      "(MultiTensor, optional) Scale_weights to be used for int8 weights data."
      "Only used with MKL-DNN INT8.")
      .AsDuplicable()
      .AsDispensable();
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
  AddAttr<int>("layers",
               "(int, default: 1) "
               "Number of stacked GRU layers.")
      .SetDefault(1);
  AddAttr<bool>("origin_mode",
                "bool"
                "use origin mode in article https://arxiv.org/abs/1412.3555")
      .SetDefault(false);
  AddAttr<std::string>(
      "mkldnn_data_type",
      "(string, default \"float32\"). Data type of mkldnn kernel")
      .SetDefault("float32")
      .InEnum({"float32", "int8", "bfloat16"});
  AddAttr<float>("Scale_data",
                 "Scales to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault({1.f});
  AddAttr<float>("Shift_data",
                 "Shifts to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault({0.f});
  AddAttr<bool>("force_fp32_output",
                "(bool, default: false) Force INT8 kernel output FP32, only "
                "used in MKL-DNN INT8")
      .SetDefault(false);
  AddComment(R"DOC(
The Fusion complete GRU Operator.
This operator fuse the fully-connected operator into GRU,
more details can refer to GRU op.
)DOC");
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multi_gru, ops::MultiGRUOp, ops::MultiGRUOpMaker);
