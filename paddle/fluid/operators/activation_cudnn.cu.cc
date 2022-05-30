// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {
using framework::Tensor;
using platform::ActivationDescriptor;
using platform::TensorDescriptor;

template <typename Functor>
class CudnnActivationKernel
    : public framework::OpKernel<Functor::ElEWISE_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    framework::Tensor *X, *Out;
    ExtractActivationTensor(context, X, Out);
    ActivationDescriptor act_desc;
    TensorDescriptor x_desc, out_desc;
    x_desc.set(GET_DATA_SAFELY(X, "Input", "X", "CudnnActivation"));
    out_desc.set(GET_DATA_SAFELY(Out, "Output", "Out", "CudnnActivation");
  }
};

}  // namespace operators
}  // namespace paddle
