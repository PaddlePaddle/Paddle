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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void SimpleMarkerKernel(T* in, T* out, int ndim) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (; idx < ndim; idx += blockDim.x * gridDim.x) {
    out[idx] = in[idx];
  }
}

template <typename T>
class MarkerOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    auto marker_role = ctx.Attr<std::string>("marker_role");
    auto marker_pos = ctx.Attr<std::string>("marker_pos");
    VLOG(3) << "marker role: " << marker_role
            << " marker position: " << marker_pos;

    phi::DenseTensor A;
    phi::DenseTensor B;
    auto* in_temp = A.mutable_data<T>({32, 1}, ctx.GetPlace());
    auto* out_temp = B.mutable_data<T>({32, 1}, ctx.GetPlace());
    platform::RecordEvent record_event(
        "MarkerCUDA",
        "marker_" + marker_role + "_" + marker_pos,
        platform::TracerEventType::OperatorInner,
        1,
        platform::EventRole::kInnerOp);
    SimpleMarkerKernel<T>
        <<<1, 32, 0, dev_ctx.stream()>>>(in_temp, out_temp, 32);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(marker, ops::MarkerOpCUDAKernel<float>);
