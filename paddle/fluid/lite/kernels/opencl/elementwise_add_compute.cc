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

#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/operators/op_params.h"
// NOTE ugly here, hide these.
#include "paddle/fluid/lite/opencl/cl_caller.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ElementwiseAddCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<OpenClContext>();
    CHECK(context.cl_helper() != nullptr);

    elementwise_add(
        context.cl_helper(), static_cast<const float*>(param.X->raw_data()),
        param.X->dims(), static_cast<const float*>(param.Y->raw_data()),
        param.Y->dims(), param.Out->mutable_data<float>(), param.Out->dims());
  }
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add, kOpenCL, kFloat, kNCHW,
                     paddle::lite::kernels::opencl::ElementwiseAddCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
