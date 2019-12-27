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

#include "paddle/fluid/operators/fill_constant_batch_size_like_op.h"
#include "paddle/fluid/operators/batch_size_like.h"

namespace paddle {
namespace operators {

class FillConstantBatchSizeLikeOp : public BatchSizeLikeOp {
 protected:
  using BatchSizeLikeOp::BatchSizeLikeOp;
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.device_context());
  }
};

class FillConstantBatchSizeLikeOpMaker : public BatchSizeLikeOpMaker {
 protected:
  void Apply() override {
    AddAttr<int>(
        "dtype",
        "It could be numpy.dtype. Output data type. Default is float32")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<float>("value", "default 0. The value to be filled")
        .SetDefault(0.0f);
    AddAttr<std::string>("str_value", "default empty. The value to be filled")
        .SetDefault("");
    AddAttr<bool>("force_cpu",
                  "(bool, default false) Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device")
        .SetDefault(false);
    AddComment(R"DOC(
This function creates a tensor of specified *shape*, *dtype* and batch size,
and initializes this with a constant supplied in *value*. The batch size is
obtained from the `input` tensor.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fill_constant_batch_size_like, ops::FillConstantBatchSizeLikeOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::FillConstantBatchSizeLikeOpMaker,
    ops::BatchSizeLikeNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUDeviceContext,
                                           float>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUDeviceContext,
                                           double>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUDeviceContext,
                                           int>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUDeviceContext,
                                           int64_t>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUDeviceContext,
                                           bool>);
