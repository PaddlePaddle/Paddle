// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/trace_op.h"
#include "paddle/pten/kernels/trace_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TraceCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int offset = context.Attr<int>("offset");
    const int dim1 = context.Attr<int>("axis1");
    const int dim2 = context.Attr<int>("axis2");

    auto& dev_ctx = context.device_context<DeviceContext>();
    pten::TraceKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input, offset, dim1, dim2, out);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    trace, ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext,
                         platform::float16>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<float>>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    trace_grad, ops::TraceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TraceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TraceGradKernel<paddle::platform::CUDADeviceContext,
                         platform::float16>,
    ops::TraceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TraceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TraceGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<float>>,
    ops::TraceGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<double>>);
