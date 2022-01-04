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

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/range_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void RangeKernel(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
}

template <typename T>
class CUDARangeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* start_t = context.Input<framework::Tensor>("Start");
    auto* end_t = context.Input<framework::Tensor>("End");
    auto* step_t = context.Input<framework::Tensor>("Step");
    auto* out = context.Output<framework::Tensor>("Out");

    T start = GetValue<T>(start_t);
    T end = GetValue<T>(end_t);
    T step = GetValue<T>(step_t);

    int64_t size = 0;
    GetSize(start, end, step, &size);
    out->Resize(framework::make_ddim({size}));
    T* out_data = out->mutable_data<T>(context.GetPlace());

    auto stream = context.cuda_device_context().stream();
    int block = std::min(size, static_cast<int64_t>(256));
    int grid = (size + block - 1) / block;
    RangeKernel<T><<<grid, block, 0, stream>>>(start, step, size, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(range, ops::CUDARangeKernel<int>,
                        ops::CUDARangeKernel<int64_t>,
                        ops::CUDARangeKernel<float>,
                        ops::CUDARangeKernel<double>);
