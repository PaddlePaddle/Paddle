/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/framework/op_registry.h"
#include "paddle/operators/conv2d_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// FIXME(typhoonzer): If CudnnConvOp is running on CPU
// reuse the code from gemm_conv2d_op.h.
template <typename Place, typename T>
class CudnnConvKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    GemmConv2DCompute<Place, T>(context);
  }
};

template <typename Place, typename T>
class CudnnConvGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    GemmConvGrad2DCompute<Place, T>(context);
  }
};

}  // namespace operators
}  // namespace paddle
