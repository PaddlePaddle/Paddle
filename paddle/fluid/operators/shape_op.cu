/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>

#include <vector>

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/shape_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

__global__ void ShapeCopyFunc(const int32_t* dim_arr, int32_t* out_data,
                              size_t dim_size) {
  const int tid = blockIdx.x;
  if (tid < dim_size) {
    out_data[tid] = dim_arr[tid];
  }
}

template <typename DeviceContext, typename T>
class GPUShapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in_t = context.Input<Tensor>("Input");
    auto* out_t = context.Output<Tensor>("Out");
    int32_t* out_data = out_t->mutable_data<int32_t>(context.GetPlace());

    const auto& dev_ctx = context.cuda_device_context();
    // const auto& ctx = context.template device_context<DeviceContext>();
    const auto gplace = boost::get<platform::CUDAPlace>(context.GetPlace());
    auto cplace = platform::CPUPlace();

    const auto& input_dim = in_t->dims();
    const size_t& dim_size = input_dim.size();

    std::vector<int> v_input_dims(dim_size);
    for (size_t i = 0; i < dim_size; ++i) {
      v_input_dims[i] = static_cast<int>(input_dim[i]);
    }
    int bytes = dim_size * sizeof(int);

    auto dim_arr_ptr = memory::Alloc(dev_ctx, bytes);
    int32_t* dim_arr_cuda = reinterpret_cast<int32_t*>(dim_arr_ptr->ptr());

    memory::Copy(gplace, dim_arr_cuda, cplace, v_input_dims.data(), bytes,
                 dev_ctx.stream());

    int grid = input_dim.size();
    int block = 1;

    ShapeCopyFunc<<<grid, block, 0, dev_ctx.stream()>>>(dim_arr_cuda, out_data,
                                                        dim_size);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    shape,
    paddle::operators::GPUShapeKernel<paddle::platform::CUDADeviceContext, int>,
    paddle::operators::GPUShapeKernel<paddle::platform::CUDADeviceContext,
                                      int32_t>,
    paddle::operators::GPUShapeKernel<paddle::platform::CUDADeviceContext,
                                      int64_t>,
    paddle::operators::GPUShapeKernel<paddle::platform::CUDADeviceContext,
                                      float>,
    paddle::operators::GPUShapeKernel<paddle::platform::CUDADeviceContext,
                                      double>);
