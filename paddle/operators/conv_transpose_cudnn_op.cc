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

#include "paddle/operators/conv_transpose_op.h"

namespace paddle {
namespace operators {

class CudnnConv2DTransposeOpMaker : public Conv2DTransposeOpMaker {
 public:
  CudnnConv2DTransposeOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : Conv2DTransposeOpMaker(proto, op_checker) {
    AddAttr<int>("workspace_size_MB",
                 "workspace size for cudnn, in MB, "
                 "workspace is a section of GPU memory which will be "
                 "allocated/freed each time the operator runs, larger "
                 "workspace size can increase performance but also requires "
                 "better hardward. This size should be carefully setted.")
        .SetDefault(4096);
  }
};

class CudnnConv3DTransposeOpMaker : public Conv3DTransposeOpMaker {
 public:
  CudnnConv3DTransposeOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : Conv3DTransposeOpMaker(proto, op_checker) {
    AddAttr<int>("workspace_size_MB",
                 "workspace size for cudnn, in MB, "
                 "workspace is a section of GPU memory which will be "
                 "allocated/freed each time the operator runs, larger "
                 "workspace size can increase performance but also requires "
                 "better hardward. This size should be carefully setted.")
        .SetDefault(4096);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(conv2d_transpose_cudnn, ops::ConvTransposeOp,
            ops::CudnnConv2DTransposeOpMaker, conv2d_transpose_cudnn_grad,
            ops::ConvTransposeOpGrad);

REGISTER_OP_CPU_KERNEL(
    conv2d_transpose_cudnn,
    ops::GemmConvTransposeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvTransposeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv2d_transpose_cudnn_grad,
    ops::GemmConvTransposeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvTransposeGradKernel<paddle::platform::CPUDeviceContext,
                                     double>);

REGISTER_OP(conv3d_transpose_cudnn, ops::ConvTransposeOp,
            ops::CudnnConv3DTransposeOpMaker, conv3d_transpose_cudnn_grad,
            ops::ConvTransposeOpGrad);

REGISTER_OP_CPU_KERNEL(
    conv3d_transpose_cudnn,
    ops::GemmConvTransposeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvTransposeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv3d_transpose_cudnn_grad,
    ops::GemmConvTransposeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvTransposeGradKernel<paddle::platform::CPUDeviceContext,
                                     double>);
