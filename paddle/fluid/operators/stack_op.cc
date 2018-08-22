// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/stack_op.h"

namespace paddle {
namespace operators {

struct CPUStackFunctor {
  template <typename DeviceContext, typename T>
  void operator()(const DeviceContext& ctx, const std::vector<const T*>& x,
                  T* y, int pre, int n, int post) const {
    int total_num = pre * post * n;
    for (int idx = 0; idx < total_num; ++idx) {
      int i = idx / (n * post);
      int which_x = idx / post - i * n;
      int x_index = i * post + idx % post;
      y[idx] = x[which_x][x_index];
    }
  }
};

struct CPUStackGradFunctor {
  template <typename DeviceContext, typename T>
  void operator()(const DeviceContext& ctx, std::vector<T*>& dx,  // NOLINT
                  const T* dy, int pre, int n, int post) const {
    int total_num = pre * post * n;
    for (int idx = 0; idx < total_num; ++idx) {
      int i = idx / (n * post);
      int which_x = idx / post - i * n;
      int x_index = i * post + idx % post;
      dx[which_x][x_index] = dy[idx];
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;
REGISTER_OPERATOR(stack, ops::StackOp, ops::StackOpMaker,
                  ops::StackGradOpDescMaker);
REGISTER_OPERATOR(stack_grad, ops::StackOpGrad);

REGISTER_OP_CPU_KERNEL(
    stack,
    ops::StackKernel<plat::CPUDeviceContext, float, ops::CPUStackFunctor>,
    ops::StackKernel<plat::CPUDeviceContext, double, ops::CPUStackFunctor>);

REGISTER_OP_CPU_KERNEL(stack_grad,
                       ops::StackGradKernel<plat::CPUDeviceContext, float,
                                            ops::CPUStackGradFunctor>,
                       ops::StackGradKernel<plat::CPUDeviceContext, double,
                                            ops::CPUStackGradFunctor>);
