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

#include "paddle/fluid/lite/kernels/host/reshape_compute.h"
#include <vector>
#include "paddle/fluid/lite/operators/reshape_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void ReshapeCompute::Run() {
  auto& param = Param<operators::ReshapeParam>();
  auto x = param.x;
  auto actual_shape = param.actual_shape;
  auto output = param.output;
  bool inplace = param.inplace;
  auto x_dims = x->dims();
  auto output_dims = output->dims();
  if (actual_shape) {
    auto actual_shape_dims = actual_shape->dims();
    auto* actual_shape_data = actual_shape->data<int>();
#ifdef LITE_WITH_CUDA
    lite::Tensor cpu_actual_shape;
    if (actual_shape->target() == TARGET(kCUDA)) {
      cpu_actual_shape.CopyDataFrom(*actual_shape);
      actual_shape_data = cpu_actual_shape.data<int>();
    }
#endif
    auto shape = std::vector<int>(
        actual_shape_data, actual_shape_data + actual_shape_dims.production());
    output_dims = lite::operators::ValidateShape(shape, x_dims);
    output->Resize(output_dims);
  }
  if (inplace) {
    output->ShareDataWith(*x);
  } else {
    output->CopyDataFrom(*x);
  }
  output->Resize(output_dims);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reshape, kHost, kAny, kAny,
                     paddle::lite::kernels::host::ReshapeCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                           DATALAYOUT(kAny), -1)})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                               DATALAYOUT(kAny), -1)})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                              DATALAYOUT(kAny), -1)})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2, kHost, kAny, kAny,
                     paddle::lite::kernels::host::ReshapeCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                           DATALAYOUT(kAny), -1)})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                               DATALAYOUT(kAny), -1)})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                              DATALAYOUT(kAny), -1)})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny),
                                                 DATALAYOUT(kAny), -1)})
    .Finalize();
