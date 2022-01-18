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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/range_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class XPURangeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* start_t = context.Input<framework::Tensor>("Start");
    auto* end_t = context.Input<framework::Tensor>("End");
    auto* step_t = context.Input<framework::Tensor>("Step");
    auto* out = context.Output<framework::Tensor>("Out");

    framework::Tensor n;
    framework::TensorCopySync(*start_t, platform::CPUPlace(), &n);
    T start = n.data<T>()[0];
    framework::TensorCopySync(*end_t, platform::CPUPlace(), &n);
    T end = n.data<T>()[0];
    framework::TensorCopySync(*step_t, platform::CPUPlace(), &n);
    T step = n.data<T>()[0];

    int64_t size = 0;
    GetSize(start, end, step, &size);
    out->Resize(framework::make_ddim({size}));

    T* out_data = out->mutable_data<T>(context.GetPlace());

    framework::Tensor out_cpu;
    T* out_cpu_data_ptr =
        out_cpu.mutable_data<T>(platform::CPUPlace(), out->numel() * sizeof(T));
    T value = start;
    for (int64_t i = 0; i < size; ++i) {
      out_cpu_data_ptr[i] = value;
      value += step;
    }
    memory::Copy(context.GetPlace(), static_cast<void*>(out_data),
                 platform::CPUPlace(), static_cast<void*>(out_cpu_data_ptr),
                 out->numel() * sizeof(T));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(range, ops::XPURangeKernel<int>,
                       ops::XPURangeKernel<int64_t>, ops::XPURangeKernel<float>,
                       ops::XPURangeKernel<double>);
#endif  // PADDLE_WITH_XPU
