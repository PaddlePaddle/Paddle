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

#pragma once

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/top_utils.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

// only can include the headers in paddle/top/api dirs
#include "paddle/top/api/include/dev/core.h"
#include "paddle/top/api/include/dev/math.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class SignKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto& dev_ctx = context.device_context<DeviceContext>();

    // debug: print all registered sign kernels for check
    VLOG(1) << pt::OpKernelFactory::Instance();

    // TODO(chenweihang): only to test correctness, this will introduce
    // needless context prepare cost
    pt::OpKernelContext op_kernel_ctx(dev_ctx);
    auto pt_x =
        framework::MakeTensorImpl<pt::DenseTensor>(*x, x->place(), x->type());
    auto pt_out =
        framework::MakeTensorImpl<pt::DenseTensor>(*out, x->place(), x->type());
    op_kernel_ctx.EmplaceBackInput(pt_x);
    op_kernel_ctx.EmplaceBackOutput(pt_out);

    auto& op_kernel = pt::OpKernelFactory::Instance().SelectKernel(
        "sign", pt::TransToPtBackend(x->place()),
        pt::TransToPtLayout(x->layout()), pt::TransToPtDataType(x->type()));
    op_kernel(&op_kernel_ctx);

    // share pt_out data to out
    framework::ShareTensorImpl(pt_out.get(), out);
  }
};

}  // namespace operators
}  // namespace paddle
