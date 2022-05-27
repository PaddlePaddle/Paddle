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

#include <string>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fake_dequantize_op.cu.h"
#include "paddle/fluid/operators/fake_quantize_op.cu.h"
#include "paddle/fluid/operators/quantize_linear_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
struct ChannelDequantizeFunctorV2<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, const int quant_axis, framework::Tensor* out) {
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
    int64_t num = in->numel();
    const T* scale_factor = scale->data<T>();
    int64_t block_size = std::min(
        num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
    int64_t max_threads =
        dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (num + block_size - 1) / block_size);

    int quant_stride = 1;
    for (int i = quant_axis + 1; i < in_dims.size(); i++) {
      quant_stride *= in_dims[i];
    }

    DequantizeOneScaleQuantAxisN<
        T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        in_data, scale_factor, max_range, num, in_dims[quant_axis],
        quant_stride, out_data);
  }
};

template struct ChannelDequantizeFunctorV2<platform::CUDADeviceContext, float>;
template struct ChannelDequantizeFunctorV2<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(dequantize_linear,
                        ops::DeQuantizeLinearKernel<CUDA, float, float>,
                        ops::DeQuantizeLinearKernel<CUDA, int8_t, float>,
                        ops::DeQuantizeLinearKernel<CUDA, double, double>);

REGISTER_OP_CUDA_KERNEL(quantize_linear,
                        ops::QuantizeLinearKernel<CUDA, float>);
