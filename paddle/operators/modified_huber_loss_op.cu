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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/modified_huber_loss_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ModifiedHuberLossGradGPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // auto* in0 = context.Input<Tensor>("X");
    // auto* in1 = context.Input<Tensor>("Y");
    // auto* in2 = context.Input<Tensor>("intermediate_val");
    // auto* in3 = context.Input<Tensor>(framework::GradVarName("Out"));
    // auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    // auto* out1 = context.Output<Tensor>(framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(
    modified_huber_loss,
    ops::ModifiedHuberLossKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(modified_huber_loss_grad,
                       ops::ModifiedHuberLossGradGPUKernel<float>);
