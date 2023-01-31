/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/randperm_op.h"

namespace paddle {
namespace operators {

template <typename T>
class RandpermMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int n = ctx.Attr<int>("n");
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    framework::Variable* out_var = ctx.OutputVar("Out");
    phi::DenseTensor* out_tensor =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);

    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(phi::make_ddim({n}));
    T* tmp_data = tmp_tensor.mutable_data<T>(platform::CPUPlace());
    random_permate<T>(tmp_data, n, seed);
    framework::TensorCopySync(tmp_tensor, ctx.GetPlace(), out_tensor);
  }
};

}  // namespace operators
}  // namespace paddle

template <typename T>
using kernel = paddle::operators::RandpermMLUKernel<T>;

REGISTER_OP_MLU_KERNEL(
    randperm, kernel<int64_t>, kernel<int>, kernel<float>, kernel<double>);
