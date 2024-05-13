/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/uniform_random_op.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/generator.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class XPUUniformRandomInplaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_var = ctx.OutputVar("Out");
    auto *tensor = out_var->GetMutable<phi::DenseTensor>();
    T *data = tensor->mutable_data<T>(ctx.GetPlace());

    int64_t size = tensor->numel();
    std::unique_ptr<T[]> data_cpu(new T[size]);
    std::uniform_real_distribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    auto engine = phi::GetCPURandomEngine(seed);
    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = dist(*engine);
    }

    unsigned int diag_num =
        static_cast<unsigned int>(ctx.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(ctx.Attr<int>("diag_step"));
    auto diag_val = static_cast<T>(ctx.Attr<float>("diag_val"));
    if (diag_num > 0) {
      PADDLE_ENFORCE_GT(
          size,
          (diag_num - 1) * (diag_step + 1),
          phi::errors::InvalidArgument(
              "ShapeInvalid: the diagonal's elements is equal (num-1) "
              "* (step-1) with num %d, step %d,"
              "It should be smaller than %d, but received %d",
              diag_num,
              diag_step,
              (diag_num - 1) * (diag_step + 1),
              size));
      for (int64_t i = 0; i < diag_num; ++i) {
        int64_t pos = i * diag_step + i;
        data_cpu[pos] = diag_val;
      }
    }
    phi::memory_utils::Copy(ctx.GetPlace(),
                            data,
                            phi::CPUPlace(),
                            reinterpret_cast<void *>(data_cpu.get()),
                            size * sizeof(T));
  }
};

template <typename T, typename DeviceContext>
class XPUUniformRandomInplaceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    if (dx) {
      T *data = dx->mutable_data<T>(ctx.GetPlace());
      int64_t size = dx->numel();
      std::unique_ptr<T[]> data_cpu(new T[size]);
      for (int64_t i = 0; i < size; ++i) {
        data_cpu[i] = T(0);
      }
      phi::memory_utils::Copy(ctx.GetPlace(),
                              data,
                              phi::CPUPlace(),
                              reinterpret_cast<void *>(data_cpu.get()),
                              size * sizeof(T));
    }
  }
};

}  // namespace operators
}  // namespace paddle

PD_REGISTER_STRUCT_KERNEL(uniform_random_inplace,
                          XPU,
                          ALL_LAYOUT,
                          ops::XPUUniformRandomInplaceKernel,
                          float) {}
PD_REGISTER_STRUCT_KERNEL(uniform_random_inplace_grad,
                          XPU,
                          ALL_LAYOUT,
                          ops::XPUUniformRandomInplaceGradKernel,
                          float) {}

#endif  // PADDLE_WITH_XPU
