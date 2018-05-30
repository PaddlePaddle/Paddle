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

#include "paddle/fluid/operators/reduce_op.h"

#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_REDUCE_OP(op_name)                                        \
  class __##op_name##Maker__ : public ops::ReduceOpMaker {                 \
   protected:                                                              \
    virtual std::string GetName() const { return #op_name; }               \
    virtual std::string GetOpType() const { return "Reduce " #op_name; }   \
  };                                                                       \
  REGISTER_OPERATOR(reduce_##op_name, ops::ReduceOp, __##op_name##Maker__, \
                    paddle::framework::DefaultGradOpDescMaker<true>);      \
  REGISTER_OPERATOR(reduce_##op_name##_grad, ops::ReduceGradOp)

REGISTER_REDUCE_OP(mean);
REGISTER_REDUCE_OP(max);
REGISTER_REDUCE_OP(min);
REGISTER_REDUCE_OP(prod);

#define REGISTER_REDUCE_CPU_KERNEL(reduce_type, functor, grad_functor)         \
  REGISTER_OP_CPU_KERNEL(reduce_type,                                          \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           float, ops::functor>,               \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           double, ops::functor>,              \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           int, ops::functor>,                 \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           int64_t, ops::functor>);            \
  REGISTER_OP_CPU_KERNEL(                                                      \
      reduce_type##_grad,                                                      \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, float,         \
                            ops::grad_functor>,                                \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, double,        \
                            ops::grad_functor>,                                \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, int,           \
                            ops::grad_functor>,                                \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, int64_t,       \
                            ops::grad_functor>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_REDUCE_CPU_KERNEL);
