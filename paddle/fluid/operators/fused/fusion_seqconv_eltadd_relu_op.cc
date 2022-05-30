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

#include "paddle/fluid/operators/fused/fusion_seqconv_eltadd_relu_op.h"
#include <algorithm>  // for min, max
#include <string>
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace paddle {
namespace operators {

void FusionSeqConvEltAddReluOp::InferShape(
    framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                 "fusion_seqconv_eltadd_relu");
  OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter",
                 "fusion_seqconv_eltadd_relu");
  OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias",
                 "fusion_seqconv_eltadd_relu");

  OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                 "fusion_seqconv_eltadd_relu");
  OP_INOUT_CHECK(ctx->HasOutput("ColMat"), "Output", "ColMat",
                 "fusion_seqconv_eltadd_relu");

  auto x_dims = ctx->GetInputDim("X");
  auto w_dims = ctx->GetInputDim("Filter");
  int context_length = ctx->Attrs().Get<int>("contextLength");
  PADDLE_ENFORCE_EQ(ctx->Attrs().Get<int>("contextStride"), 1,
                    platform::errors::InvalidArgument(
                        "Currently, FusionSeqConvEltAddReluOp only supports "
                        "contextStride=1, but received value is: %d.",
                        ctx->Attrs().Get<int>("contextStride")));

  PADDLE_ENFORCE_EQ(
      x_dims.size(), 2,
      platform::errors::InvalidArgument(
          "Input(X) should be 2-D tensor, but reveiced value is: %d.",
          x_dims.size()));

  PADDLE_ENFORCE_EQ(
      w_dims.size(), 2,
      platform::errors::InvalidArgument(
          "Filter should be 2-D tensor, but reveiced value is: %d.",
          w_dims.size()));

  PADDLE_ENFORCE_EQ(w_dims[0], context_length * x_dims[1],
                    platform::errors::InvalidArgument(
                        "Filter's height should be equal to context_length * "
                        "input_hidden_size, but received Filter height is: %d,"
                        "context_length is: %d, input_hidden_size is: %d.",
                        w_dims[0], context_length, x_dims[1]));

  PADDLE_ENFORCE_GT(
      context_length + ctx->Attrs().Get<int>("contextStart"), 0,
      platform::errors::InvalidArgument(
          "contextStart size should be smaller than contextLength, "
          "but received context_length is: %d, contextStart is: "
          "%d.",
          context_length, ctx->Attrs().Get<int>("contextStart")));

  ctx->SetOutputDim("Out", {x_dims[0], w_dims[1]});
  ctx->SetOutputDim("ColMat", {x_dims[0], w_dims[0]});
  ctx->ShareLoD("X", "Out");
}

framework::OpKernelType FusionSeqConvEltAddReluOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.device_context());
}

void FusionSeqConvEltAddReluOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  // PaddingData only support false yet, should be ensured at pass.
  AddInput("Filter",
           "(Tensor) same as the input(Filter) of sequence conv op is an "
           "learnable parameter."
           "This is a tensor with shape (K, N), where K is the "
           "context_length * dim size of x, N is the output feature size.");
  AddInput("Bias",
           "(Tensor) the learnable weights. shape (1, N), where N is the "
           "output feature size");
  AddOutput(
      "Out",
      "(LoDTensor) the output(Out) is a LodTensor, which support "
      "variable-time length output sequence. The underlying tensor in "
      "this LoDTensor is a matrix with shape (T, N), where, T is the "
      "total time steps in this mini-batch, N is the output feature size.");
  AddOutput("ColMat",
            "(Tensor) (T, K), where T is where T is the "
            "total time steps in this mini-batch, K is height of Filter")
      .AsIntermediate();
  AddAttr<int>("contextLength",
               "(int) the contextLength of FusionSeqConvEltAddReluOp is the "
               "height of the convolution kernel.")
      .GreaterThan(0);
  AddAttr<int>("contextStart",
               "(int, default:0) the contextStart of FusionSeqConvEltAddReluOp "
               "represents the beginning of the convolution of the number of "
               "rows of sequence, which can be negative. The negative number "
               "means to pad contextStart time-steps of zeros or learnable "
               "parameters at the beginning of each instance. The positive "
               "number means to skip contextStart time-steps of each "
               "instance.")
      .SetDefault(0);
  AddAttr<int>(
      "contextStride",
      "(int, default:1) the contextStride of FusionSeqConvEltAddReluOp "
      "represents the stride length of convolution kernel. "
      "Currently, FusionSeqConvEltAddReluOp only supports"
      "contextStride=1.")
      .SetDefault(1)
      .GreaterThan(0);
  AddComment(R"DOC(
Fusion Sequence Conv and ElementwiseAdd Operator.
)DOC");
}

template <typename T>
class FusionSeqConvEltAddReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    auto* x = ctx.Input<LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("Filter");
    auto* b = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<LoDTensor>("Out");
    auto* col = ctx.Output<Tensor>("ColMat");

    auto x_lod = x->lod();
    auto x_dims = x->dims();
    auto w_dims = w->dims();
    PADDLE_ENFORCE_EQ(
        b->numel(), w_dims[1],
        platform::errors::InvalidArgument(
            "bias size should be equal to weights feature size, but received "
            "bias size is: %d, weights feature size is: %d.",
            b->numel(), w_dims[1]));
    PADDLE_ENFORCE_EQ(
        x_lod.size(), 1UL,
        platform::errors::InvalidArgument(
            "Only support one level sequence now, but received value is: %d.",
            x_lod.size()));

    const T* x_data = x->data<T>();
    const T* w_data = w->data<T>();
    const T* b_data = b->data<T>();
    T* y_data = y->mutable_data<T>(ctx.GetPlace());
    T* col_data = col->mutable_data<T>(ctx.GetPlace());

    int context_start = ctx.Attr<int>("contextStart");
    int context_length = ctx.Attr<int>("contextLength");
    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    // im2col
    int src_mat_w = static_cast<int>(x_dims[1]);
    int src_mat_w_sz = src_mat_w * sizeof(T);
    int col_mat_w = static_cast<int>(w_dims[0]);
    int col_mat_w_sz = col_mat_w * sizeof(T);
    for (int i = 0; i < static_cast<int>(x_lod[0].size()) - 1; ++i) {
      int st = x_lod[0][i];
      int ed = x_lod[0][i + 1];
      const T* src_data = x_data + st * src_mat_w;
      T* dst_data = col_data + st * col_mat_w;
      int seq_len = ed - st;
      if (seq_len > up_pad + down_pad) {
        // zero all up_pad and fill data
        std::memset(dst_data, 0, up_pad * col_mat_w_sz);
        dst_data = dst_data + up_pad * src_mat_w;
        int copy_size = col_mat_w_sz - up_pad * src_mat_w_sz;
        for (int j = 0; j < up_pad; ++j) {
          // blas.VCOPY?
          std::memcpy(dst_data, src_data, copy_size);
          dst_data += (col_mat_w - src_mat_w);
          copy_size += src_mat_w_sz;
        }
        // fill data
        if (context_start > 0) {
          src_data += context_start * src_mat_w;
        }
        for (int j = 0; j < seq_len - up_pad - down_pad; ++j) {
          std::memcpy(dst_data, src_data, copy_size);
          dst_data += col_mat_w;
          src_data += src_mat_w;
        }
        // zero all down_pad and fill data
        std::memset(dst_data, 0, down_pad * col_mat_w_sz);
        copy_size -= src_mat_w_sz;
        for (int j = 0; j < down_pad; ++j) {
          if (copy_size < 0) {
            copy_size = 0;
          }
          std::memcpy(dst_data, src_data, copy_size);
          dst_data += col_mat_w;
          src_data += src_mat_w;
          copy_size -= src_mat_w_sz;
        }
      } else {
        std::memset(dst_data, 0, seq_len * col_mat_w_sz);
        dst_data = dst_data + up_pad * src_mat_w;
        int zero_sz = up_pad * src_mat_w_sz;
        int cur_src_sz = seq_len * src_mat_w_sz;
        for (int j = 0; j < std::min(up_pad, seq_len); ++j) {
          int copy_size = std::min(cur_src_sz, col_mat_w_sz - zero_sz);
          std::memcpy(dst_data, src_data, copy_size);
          dst_data += (col_mat_w - src_mat_w);
          zero_sz -= src_mat_w_sz;
        }
        // from bottom
        dst_data = col_data + ed * col_mat_w;
        src_data = x_data + st * src_mat_w;
        if (context_start > 0) {
          src_data += context_start * src_mat_w;
        }
        zero_sz = down_pad * src_mat_w_sz;
        for (int j = 1; j <= std::min(down_pad, seq_len); ++j) {
          int copy_size = std::min(cur_src_sz, col_mat_w_sz - zero_sz);
          if (copy_size < 0) {
            copy_size = 0;
          }
          std::memcpy(dst_data - (zero_sz + copy_size) / sizeof(T),
                      src_data + std::max(seq_len - j - up_pad, 0) * src_mat_w,
                      copy_size);
          dst_data -= col_mat_w;
          zero_sz -= src_mat_w_sz;
        }
      }
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    phi::funcs::FCFunctor<DeviceContext, T> fc;
    fc(dev_ctx, x_dims[0], w_dims[1], w_dims[0], col_data, w_data, y_data,
       b_data, true);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_seqconv_eltadd_relu, ops::FusionSeqConvEltAddReluOp,
                  ops::FusionSeqConvEltAddReluOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_seqconv_eltadd_relu,
                       ops::FusionSeqConvEltAddReluKernel<float>,
                       ops::FusionSeqConvEltAddReluKernel<double>);
