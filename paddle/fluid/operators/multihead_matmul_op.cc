/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/multihead_matmul.h"
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class MultiHeadMatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *q = context.Input<framework::Tensor>("Q");
    auto *k = context.Input<framework::Tensor>("K");
    auto *v = context.Input<framework::Tensor>("V");

    auto &bias_q =
        detail::Ref(context.Input<framework::Tensor>("BiasQ"), "Cannot find X");
    auto &bias_k =
        detail::Ref(context.Input<framework::Tensor>("BiasK"), "Cannot find Y");
    auto &bias_v =
        detail::Ref(context.Input<framework::Tensor>("BiasV"), "Cannot find Y");

    auto &bias_qk = detail::Ref(context.Input<framework::Tensor>("BiasQK"),
                                "Cannot find Y");

    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    T scale = static_cast<T>(context.Attr<float>("alpha"));
    bool transpose_q = context.Attr<bool>("transpose_Q");
    bool transpose_k = context.Attr<bool>("transpose_K");
    bool transpose_v = context.Attr<bool>("transpose_V");

    int head_number = context.Attr<int>("head_number");
    // compute q*k with eltadd
    auto &device_ctx = context.template device_context<DeviceContext>();

    math::MultiHeadGPUCompute<platform::CUDADeviceContext, T>::compute(
        device_ctx, head_number, q->dims(), k->dims(), v->dims(), q->data<T>(),
        k->data<T>(), v->data<T>(), bias_q.data<T>(), bias_k.data<T>(),
        bias_v.data<T>(), bias_qk.data<T>(), out->data<T>(), scale, T(0.0),
        transpose_q, transpose_k, transpose_v);
  }
};

class MultiHeadMatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInput("Q"), true,
                      "Input(Q) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("K"), true,
                      "Input(K) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("V"), true,
                      "Input(V) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasQ"), true,
                      "Input(BiasQ) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasK"), true,
                      "Input(BiasQ) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasV"), true,
                      "Input(BiasQ) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasQK"), true,
                      "Input(BiasQK) of MultiheadOp should not be null.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of MatMulOp should not be null.");

    auto dim_q = context->GetInputDim("Q");
    PADDLE_ENFORCE_GT(dim_q.size(), 2,
                      "Multihead input should be at least 2-D tensor.");

    auto dim_k = context->GetInputDim("K");
    PADDLE_ENFORCE_GT(dim_q.size(), 2,
                      "Multihead input should be at least 2-D tensor.");

    auto dim_v = context->GetInputDim("V");
    PADDLE_ENFORCE_GT(dim_q.size(), 2,
                      "Multihead input should be at least 2-D tensor.");

    PADDLE_ENFORCE_EQ(dim_q[0], dim_k[0],
                      "Multihead input should have same batch size");
    PADDLE_ENFORCE_EQ(dim_q[0], dim_v[0],
                      "Multihead input should have same batch size");

    PADDLE_ENFORCE_EQ(dim_q[1], dim_k[1],
                      "Multihead input should have same size");
    PADDLE_ENFORCE_EQ(dim_q[1], dim_v[1],
                      "Multihead input should have same size");

    PADDLE_ENFORCE_EQ(dim_q[2], dim_k[2],
                      "Multihead input should have same size");
    PADDLE_ENFORCE_EQ(dim_q[2], dim_v[2],
                      "Multihead input should have same size");

    auto dim_bias_q = context->GetInputDim("BiasQ");
    PADDLE_ENFORCE_GT(dim_bias_q.size(), 0,
                      "Multihead input should be at least 2-D tensor.");
    auto dim_bias_k = context->GetInputDim("BiasK");
    PADDLE_ENFORCE_GT(dim_bias_k.size(), 0,
                      "Multihead input should be at least 2-D tensor.");
    auto dim_bias_v = context->GetInputDim("BiasV");
    PADDLE_ENFORCE_GT(dim_bias_v.size(), 0,
                      "Multihead input should be at least 2-D tensor.");

    PADDLE_ENFORCE_EQ(dim_bias_q[0], dim_bias_k[0],
                      "Multihead input bias should have same batch size");
    PADDLE_ENFORCE_EQ(dim_bias_q[0], dim_bias_v[0],
                      "Multihead input bias should have same batch size");

    PADDLE_ENFORCE_EQ(dim_bias_q[1], dim_bias_k[1],
                      "Multihead input bias should have same size");
    PADDLE_ENFORCE_EQ(dim_bias_q[1], dim_bias_v[1],
                      "Multihead input bias should have same size");

    auto dim_bias_qk = context->GetInputDim("BiasQK");
    PADDLE_ENFORCE_GT(dim_bias_qk.size(), 3,
                      "Multihead input bias qk should be at least 3-D tensor.");

    int head_number = context->Attrs().Get<int>("head_number");
    PADDLE_ENFORCE_GT(head_number, 1,
                      "Multihead input head number should be at least 1.");

    context->SetOutputDim("Out", dim_q);
    context->ShareLoD("Q", /*->*/ "Out");
  }
};

class MultiHeadMatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "The first input of MultiHeadMatMul op");
    AddInput("K", "The second input of MMultiHeadMatMul op");
    AddInput("V", "The third input of MultiHeadMatMul op");
    AddInput("BiasQ", "The first bias input of MultiHeadMatMul op");
    AddInput("BiasK", "The second bias input of MultiHeadMatMul op");
    AddInput("BiasV", "The third  bias input of MultiHeadMatMul op");
    AddInput("BiasQK", "The QK bias input of MultiHeadMatMul op");
    AddOutput("Out", "The output of MultiHeadMatMul op");
    AddAttr<bool>("transpose_Q",
                  R"DOC(If true, use the transpose of `Q`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_K",
                  R"DOC(If true, use the transpose of `K`.
        )DOC")
        .SetDefault(true);
    AddAttr<bool>("transpose_V",
                  R"DOC(If true, use the transpose of `V`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
    AddComment(R"DOC(
MultiHeadMatMul Operator.

If a transpose flag is specified, the last two dimensions of the
tensor are transposed. If the tensor is rank-1 of shape [D], then
for `X` it is treated as [1, D] in nontransposed form and as [D, 1]
in transposed form, whereas for `Y` it is the opposite: It is treated
as [D, 1] in nontransposed form and as [1, D] in transposed form.


Example of matrix multiplication with head_number of H
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, H * N]

The behavior is designed to be similar to the `numpy.matmul` function.
The differences are:
- When the rank of the input data is less than or equal to 3, it
  is similar to the `numpy.matmul` function.
- When the rank of the input is greater than 3, the rank of X and
  Y must be equal, and the first `rank - 2` dimensions must be equal.
- We add `transpose_X` and `transpose_Y` flags.
- We add `head_number` attribute, which is used to multiple two matrixes head
  by head, and eventually concatenates the output of several (head_number)
  small matrixes multiplication.

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `X`.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(multihead_matmul, ops::MultiHeadMatMulOp,
                             ops::MultiHeadMatMulOpMaker);

REGISTER_OP_CUDA_KERNEL(
    multihead_matmul,
    ops::MultiHeadMatMulKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MultiHeadMatMulKernel<paddle::platform::CUDADeviceContext, double>);
