/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"
#include "paddle/operators/transpose_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void transpose_kernel(int nthreads, const T* in_data, T* out_data,
                                 int* offset_buffer, int ndims) {
  int* in_offset = offset_buffer;
  int* out_offset = offset_buffer + ndims;
  int* axis = offset_buffer + ndims;

  int to_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (to_index < nthreads) {
    int from_index = 0;
    int temp = to_index;
    for (size_t i = 0; i < ndims; i++) {
      from_index += (temp / out_offset[i]) * in_offset[axis[i]];
      temp = temp % out_offset[i];
    }
    out_data[to_index] = in_data[from_index];
  }
}

template <typename T>
void TransposeCUDA(const framework::ExecutionContext& context,
                   const framework::Tensor& in, framework::Tensor& out,
                   std::vector<int> axis) {
  auto* in_data = in.template data<T>();
  auto* out_data = out.template mutable_data<T>(context.GetPlace());
  auto in_dim = in.dims();
  auto out_dim = out.dims();
  auto data_size = product(in_dim);
  size_t ndims = in_dim.size();
  std::vector<int> in_offset(ndims, 1);
  std::vector<int> out_offset(ndims, 1);
  std::vector<int64_t> buffer_dim_shape(1, ndims * 3);

  auto buffer_dims = framework::make_ddim(buffer_dim_shape);
  framework::Tensor host_buffer;
  platform::CPUPlace cpu_place;
  platform::GPUPlace gpu_place;

  int* host_buffer_data = host_buffer.mutable_data<int>(buffer_dims, cpu_place);

  auto offset_buffer =
      memory::Alloc(context.GetPlace(), ndims * 3 * sizeof(int));

  for (int i = ndims - 2; i >= 0; i--) {
    in_offset[i] = in_offset[i + 1] * in_dim[i + 1];
    out_offset[i] = out_offset[i + 1] * out_dim[i + 1];
  }

  for (int i = 0; i < ndims; i++) {
    host_buffer_data[i] = in_offset[i];
    host_buffer_data[i + ndims] = out_offset[i];
    host_buffer_data[i + ndims * 2] = axis[i];
  }

  memory::Copy(gpu_place, offset_buffer, cpu_place, host_buffer_data,
               ndims * 3 * sizeof(int));
  int block = 512;
  int grid = (data_size + block - 1) / block;
  transpose_kernel<T><<<grid, block>>>(data_size, in_data, out_data,
                                       static_cast<int*>(offset_buffer), ndims);
  memory::Free(gpu_place, offset_buffer);
}

template <typename T>
class TransposeCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use GPUPlace.");
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto axis = context.GetAttr<std::vector<int>>("axis");
    TransposeCUDA<T>(context, *in, *out, axis);
  }
};

template <typename T>
class TransposeGradCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use GPUPlace.");
    auto* in = context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto axis_temp = context.GetAttr<std::vector<int>>("axis");

    std::vector<int> axis(axis_temp);

    for (size_t i = 0; i < axis.size(); i++) {
      axis[axis_temp[i]] = i;
    }
    TransposeCUDA<T>(context, *in, *out, axis);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(transpose, ops::TransposeCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(transpose_grad, ops::TransposeGradCUDAKernel<float>);
