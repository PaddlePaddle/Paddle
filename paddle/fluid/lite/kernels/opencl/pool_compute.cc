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

#include <vector>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/operators/op_params.h"
// NOTE ugly here, hide these.
#include "paddle/fluid/lite/opencl/cl_caller.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class PoolCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::PoolParam;

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.x->dims();
    const auto& out_dims = param.output->dims();
    const std::string pooling_type = param.pooling_type;
    const bool global_pooling = param.global_pooling;
    std::vector<int> paddings = param.paddings;
    std::vector<int> strides = param.strides;
    std::vector<int> ksize = param.ksize;
    if (global_pooling) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_dims[i + 2]);
      }
    }

    auto& context = ctx_->As<OpenClContext>();
    CHECK(context.cl_helper() != nullptr);

    pool(context.cl_helper(), pooling_type, paddings[0], paddings[1],
         strides[0], strides[1], ksize[0], ksize[1],
         static_cast<const float*>(param.x->raw_data()), in_dims,
         param.output->mutable_data<float>(), out_dims);
  }
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pool2d, kOpenCL, kFloat, kNCHW,
                     paddle::lite::kernels::opencl::PoolCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
