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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/operators/jit/kernels.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class UniformRandomCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto &context = ctx_->As<X86Context>();
    auto &param = *param_.get_mutable<operators::UniformRandomParam>();
    CHECK(context.x86_device_context());

    auto *param_out = &param.Out->raw_tensor();

    T *data =
        param_out->mutable_data<T>(context.x86_device_context()->GetPlace());

    unsigned int seed = static_cast<unsigned int>(param.seed);
    std::minstd_rand engine;
    if (seed == 0) {
      seed = std::random_device()();
    }
    engine.seed(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(param.min),
                                           static_cast<T>(param.max));
    int64_t size = param_out->numel();
    for (int64_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
  }

  virtual ~UniformRandomCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(uniform_random, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::UniformRandomCompute<float>,
                     def)
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
