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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = phi::DenseTensor;

template <typename T>
class SequenceSoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(false,"not support");

  }
};

template <typename T>
class SequenceSoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(false,"not support");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#ifdef PADDLE_WITH_HIP
// MIOPEN not support float64
REGISTER_OP_KERNEL(sequence_softmax,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxCUDNNKernel<float>);
REGISTER_OP_KERNEL(sequence_softmax_grad,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxGradCUDNNKernel<float>);
#else
REGISTER_OP_KERNEL(sequence_softmax,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxCUDNNKernel<float>,
                   ops::SequenceSoftmaxCUDNNKernel<double>);
REGISTER_OP_KERNEL(sequence_softmax_grad,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxGradCUDNNKernel<float>,
                   ops::SequenceSoftmaxGradCUDNNKernel<double>);
#endif
