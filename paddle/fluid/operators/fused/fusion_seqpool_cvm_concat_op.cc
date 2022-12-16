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
  PADDLE_ENFORCE_GE(
      ctx->Inputs("X").size(),
      1UL,
      paddle::platform::errors::InvalidArgument(
          "Inputs(X) of FusionSeqPoolCVMConcatOp should not be empty."));
  PADDLE_ENFORCE(
      ctx->HasOutput("Out"),
      paddle::platform::errors::InvalidArgument(
          "Output(Out) of FusionSeqPoolCVMConcatOp should not be null."));
  int axis = ctx->Attrs().Get<int>("axis");
  PADDLE_ENFORCE_EQ(axis,
                    1,
                    paddle::platform::errors::InvalidArgument(
                        "FusionSeqPoolCVMConcatOp only supports "
                        "concat axis=1 yet, but received %d.",
                        axis));
  bool use_cvm = ctx->Attrs().Get<bool>("use_cvm");
  PADDLE_ENFORCE_EQ(use_cvm,
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "FusionSeqPoolCVMConcatOp only supports "
                        "use_cvm is true yet, but received %d.",
                        use_cvm));

  auto ins_dims = ctx->GetInputsDim("X");
  const size_t n = ins_dims.size();
  PADDLE_ENFORCE_GT(n,
                    0UL,
                    paddle::platform::errors::InvalidArgument(
                        "Input tensors count should > 0."));
  if (n == 1) {
    LOG(WARNING) << "Only have one input, may waste memory";
  }

  // The output height should be confirmed in Compute,
  // since input lod is not accessible here.
  PADDLE_ENFORCE_EQ(ins_dims[0].size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "The dims size of first input should be 2."));
  ctx->SetOutputDim("Out", {-1, ins_dims[0][axis] * static_cast<int>(n)});
}

framework::OpKernelType FusionSeqPoolCVMConcatOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

void FusionSeqPoolCVMConcatOpMaker::Make() {
  AddInput("X", "(phi::DenseTensor) Input tensors of this operator.")
      .AsDuplicable();
  AddInput("CVM",
           "(phi::DenseTensor),  a 2-D phi::DenseTensor with shape [N x 2], "
           "where N is the batch "
           "size, 2 is show and click.");
  AddOutput("Out", "(phi::DenseTensor) Output tensor of concat operator.");
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
  AddComment(R"DOC(
Fusion Sequence Pool of pooltype(sum, average and sqrt) and Concat Operator.
)DOC");
}

template <typename T>
class FusionSeqPoolCVMConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto x0_lod = ins[0]->lod();
    const auto& x0_dims = ins[0]->dims();
    const auto& y_dims = out->dims();
    size_t bs = x0_lod[0].size() - 1;
    out->Resize({static_cast<int64_t>(bs), y_dims[1]});
    framework::LoD y_lod(1);
    y_lod[0].resize(bs + 1);
    for (size_t i = 0; i <= bs; ++i) {
      y_lod[0][i] = i;
    }
    out->set_lod(y_lod);
    auto place = ctx.GetPlace();
    T* y_data = out->mutable_data<T>(place);

    int w = ins[0]->numel() / x0_dims[0];
    PADDLE_ENFORCE_EQ(y_dims[1] % w,
                      0,
                      paddle::platform::errors::InvalidArgument(
                          "The output of dims[1] should be dividable of w"));
    jit::seq_pool_attr_t attr(w, jit::SeqPoolType::kSum);
    if (pooltype == "AVERAGE") {
      attr.type = jit::SeqPoolType::kAvg;
    } else if (pooltype == "SQRT") {
      attr.type = jit::SeqPoolType::kSqrt;
    }
    auto seqpool =
        jit::KernelFuncs<jit::SeqPoolTuple<T>, platform::CPUPlace>::Cache().At(
            attr);
    size_t n = ins.size();
    size_t dst_step_size = n * w;
    for (size_t i = 0; i < n; ++i) {
      const auto& x_dims = ins[i]->dims();
      auto x_lod = ins[i]->lod()[0];
      const T* src = ins[i]->data<T>();
      T* dst = y_data + i * w;
      PADDLE_ENFORCE_EQ(static_cast<int>(ins[i]->numel() / x_dims[0]),
                        w,
                        paddle::platform::errors::InvalidArgument(
                            "Width of all inputs should be equal."));
      PADDLE_ENFORCE_EQ(x_lod.size(),
                        bs + 1,
                        paddle::platform::errors::InvalidArgument(
                            "Batchsize of all inputs should be equal."));
      for (size_t j = 0; j < bs; ++j) {
        attr.h = static_cast<int>(x_lod[j + 1] - x_lod[j]);
        seqpool(src, dst, &attr);

        // Currently only use_cvm is true.
        dst[0] = log(dst[0] + 1);
        dst[1] = log(dst[1] + 1) - dst[0];

        dst += dst_step_size;
        src += attr.h * attr.w;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fusion_seqpool_cvm_concat,
    ops::FusionSeqPoolCVMConcatOp,
    ops::FusionSeqPoolCVMConcatOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(fusion_seqpool_cvm_concat,
                       ops::FusionSeqPoolCVMConcatKernel<float>,
                       ops::FusionSeqPoolCVMConcatKernel<double>);
