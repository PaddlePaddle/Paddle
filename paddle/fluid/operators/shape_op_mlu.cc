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

#ifdef PADDLE_WITH_MLU
#include <algorithm>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = phi::SelectedRows;

template <typename T>
class ShapeMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_var = ctx.InputVar("Input");
    framework::DDim in_dims;
    if (in_var->IsType<phi::SelectedRows>()) {
      in_dims = in_var->Get<phi::SelectedRows>().value().dims();
    } else {
      in_dims = in_var->Get<LoDTensor>().dims();
    }
    auto* out_t = ctx.Output<phi::DenseTensor>("Out");
    out_t->Resize({in_dims.size()});
    out_t->mutable_data<int32_t>(ctx.GetPlace());

    // shape op cpu
    Tensor shape_on_cpu(
        framework::TransToPhiDataType(framework::proto::VarType::INT32));
    shape_on_cpu.Resize({in_dims.size()});
    auto cpu_data = shape_on_cpu.mutable_data<int32_t>(platform::CPUPlace());
    for (int i = 0; i < in_dims.size(); ++i) {
      cpu_data[i] = in_dims[i];
    }

    // cpu to mlu
    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    framework::TensorCopy(shape_on_cpu, ctx.GetPlace(), dev_ctx, out_t);
    dev_ctx.Wait();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(shape,
                       ops::ShapeMLUKernel<bool>,
                       ops::ShapeMLUKernel<uint8_t>,
                       ops::ShapeMLUKernel<int8_t>,
                       ops::ShapeMLUKernel<int>,
                       ops::ShapeMLUKernel<int64_t>,
                       ops::ShapeMLUKernel<paddle::platform::float16>,
                       ops::ShapeMLUKernel<float>,
                       ops::ShapeMLUKernel<double>);

#endif
