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

#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseModXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUElementwise<T, XPUType>(ctx, xpu::broadcast_mod<XPUType>);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(elementwise_mod,
                       ops::ElementwiseModXPUKernel<float>,
                       ops::ElementwiseModXPUKernel<paddle::platform::float16>,
                       ops::ElementwiseModXPUKernel<int32_t>,
                       ops::ElementwiseModXPUKernel<int64_t>);

#endif
