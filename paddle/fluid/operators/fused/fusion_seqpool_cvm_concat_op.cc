/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/fused/fusion_seqpool_cvm_concat_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/operators/jit/kernels.h"

namespace paddle {
namespace operators {

void FusionSeqPoolCVMConcatOp::InferShape(
    framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->Inputs("X").size(), 1UL,
                    "Inputs(X) of FusionSeqPoolCVMConcatOp must be 1.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Output(Out) of FusionSeqPoolCVMConcatOp should not be null.");
  int axis = ctx->Attrs().Get<int>("axis");
  PADDLE_ENFORCE_EQ(
      axis, 1, "FusionSeqPoolCVMConcatOp only supports concat axis=1 yet.");
  bool use_cvm = ctx->Attrs().Get<bool>("use_cvm");
  PADDLE_ENFORCE_EQ(
      use_cvm, true,
      "FusionSeqPoolCVMConcatOp only supports use_cvm is true yet.");
  int n = ctx->Attrs().Get<int>("slots_num");
  PADDLE_ENFORCE_GT(
      n, 1UL,
      "slots_num of FusionSeqPoolCVMConcatOp should be greater than 1.");
  auto ins_dims = ctx->GetInputsDim("X");
  PADDLE_ENFORCE_GT(ins_dims.size(), 0UL, "Input tensors count should > 0.");

  // The output height should be confirmed in Compute,
  // since input lod is not accessible here.
  PADDLE_ENFORCE_EQ(ins_dims[0].size(), 2,
                    "The dims size of first input should be 2.");
  ctx->SetOutputDim("Out", {-1, ins_dims[0][axis] * n});
}

framework::OpKernelType FusionSeqPoolCVMConcatOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::GetDataTypeOfVar(ctx.MultiInputVar("X")[0]), ctx.GetPlace());
}

void FusionSeqPoolCVMConcatOpMaker::Make() {
  AddInput("X", "(LoDTensor) Input tensors of this operator.").AsDuplicable();
  AddInput("CVM",
           "(Tensor),  a 2-D Tensor with shape [N x 2], where N is the batch "
           "size, 2 is show and click.");
  AddOutput("Out", "(LoDTensor) Output tensor of concat operator.");
  AddAttr<std::string>("pooltype",
                       "(string, default 'SUM') some of the pooling "
                       "pooltype of SequencePoolOp.")
      .SetDefault("SUM")
      .InEnum({"AVERAGE", "SUM", "SQRT"});
  AddAttr<bool>("use_cvm", "bool, use cvm or not").SetDefault(true);
  AddAttr<int>("axis",
               "The axis along which the input tensors will be concatenated. "
               "Only supports concat axis=1 yet.")
      .SetDefault(1);
  AddAttr<int>("slots_num",
               "The number of slots input to this layer. "
               "This value is determined by pass when fusing.");
  AddComment(R"DOC(
Fusion Sequence Pool of pooltype(sum, average and sqrt), CVM and Concat Operator.
)DOC");
}

template <typename T>
class FusionSeqPoolCVMConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto in = ctx.MultiInput<LoDTensor>("X")[0];
    auto* out = ctx.Output<LoDTensor>("Out");
    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto slots_num = ctx.Attr<int>("slots_num");
    PADDLE_ENFORCE_GT(slots_num, 1, "slots_num should be greater than 1.");
    auto x_lod = in->lod()[0];
    auto x_dims = in->dims();
    auto y_dims = out->dims();
    size_t bs = (x_lod.size() - 1) / slots_num;
    out->Resize({static_cast<int64_t>(bs), y_dims[1]});
    framework::LoD y_lod(1);
    y_lod[0].resize(bs + 1);
    for (size_t i = 0; i <= bs; ++i) {
      y_lod[0][i] = i;
    }
    out->set_lod(y_lod);
    auto place = ctx.GetPlace();
    T* y_data = out->mutable_data<T>(place);
    PADDLE_ENFORCE_GT(
        x_dims[0], 0,
        "X[0]->dims()[0] of FusionSeqpoolCVMConcat should be greater than 0.");
    int w = in->numel() / x_dims[0];
    PADDLE_ENFORCE_EQ(y_dims[1] % w, 0,
                      "The output of dims[1] should be dividable of w");
    jit::seq_pool_attr_t attr(w, jit::SeqPoolType::kSum);
    if (pooltype == "AVERAGE") {
      attr.type = jit::SeqPoolType::kAvg;
    } else if (pooltype == "SQRT") {
      attr.type = jit::SeqPoolType::kSqrt;
    }
    auto seqpool =
        jit::KernelFuncs<jit::SeqPoolTuple<T>, platform::CPUPlace>::Cache().At(
            attr);
    for (size_t i = 0; i < bs * slots_num; ++i) {
      const T* src = in->data<T>() + x_lod[i] * w;
      T* dst = y_data + i * w;
      auto tmp_attr = attr;
      tmp_attr.h = static_cast<int>(x_lod[i + 1] - x_lod[i]);
      seqpool(src, dst, &tmp_attr);
      dst[0] = log(dst[0] + 1);
      dst[1] = log(dst[1] + 1) - dst[0];
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_seqpool_cvm_concat, ops::FusionSeqPoolCVMConcatOp,
                  ops::FusionSeqPoolCVMConcatOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_seqpool_cvm_concat,
                       ops::FusionSeqPoolCVMConcatKernel<float>,
                       ops::FusionSeqPoolCVMConcatKernel<double>);
