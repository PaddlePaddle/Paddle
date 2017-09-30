/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/conv_cudnn_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class CudnnConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    ConvInferShape(ctx);
  }
};

class CudnnConvOpMaker : public Conv2DOpMaker {
 public:
  CudnnConvOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : Conv2DOpMaker(proto, op_checker) {
    AddAttr<std::vector<int>>("dilations", "paddings of convolution operator.")
        .SetDefault(std::vector<int>{1, 1});
    AddAttr<int>("workspace_size_MB", "workspace size for cudnn, in MB.")
        .SetDefault(4096);
  }
};

class CudnnConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    ConvGradInferShape(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(conv_cudnn, ops::CudnnConvOp, ops::CudnnConvOpMaker,
            conv_cudnn_grad, ops::CudnnConvGradOp);
REGISTER_OP_CPU_KERNEL(conv_cudnn,
                       ops::CudnnConvKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv_cudnn_grad,
    ops::CudnnConvGradKernel<paddle::platform::CPUPlace, float>);
