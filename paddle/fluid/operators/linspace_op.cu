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

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/linspace_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void LinspaceKernel(T start, T stop, double step, int64_t size,
                               T* out) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  for (; index < size; index += blockDim.x * gridDim.x) {
    if (index < size / 2) {
      out[index] = static_cast<T>(start + step * index);
    } else {
      out[index] = static_cast<T>(stop - step * (size - index - 1));
    }
  }
}

template <typename T>
__global__ void LinspaceSpecialKernel(T start, T* out) {
  out[0] = static_cast<T>(start);
}

template <typename T>
class CUDALinspaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* pre_start = context.Input<framework::Tensor>("Start");
    auto* pre_stop = context.Input<framework::Tensor>("Stop");
    auto* num_t = context.Input<framework::Tensor>("Num");
    auto* out = context.Output<framework::Tensor>("Out");
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));

    Tensor start_t;
    Tensor stop_t;
    auto start_dtype =
        framework::OpKernelType(pre_start->type(), context.GetPlace());
    auto stop_dtype =
        framework::OpKernelType(pre_stop->type(), context.GetPlace());
    auto out_dtype = framework::OpKernelType(dtype, context.GetPlace());
    framework::TransDataType(start_dtype, out_dtype, *pre_start, &start_t);
    framework::TransDataType(stop_dtype, out_dtype, *pre_stop, &stop_t);

    framework::Tensor n_start;
    framework::Tensor n_stop;
    framework::Tensor n_num;
    framework::TensorCopy(start_t, platform::CPUPlace(), &n_start);
    T start = n_start.data<T>()[0];
    framework::TensorCopy(stop_t, platform::CPUPlace(), &n_stop);
    T stop = n_stop.data<T>()[0];
    framework::TensorCopy(*num_t, platform::CPUPlace(), &n_num);
    int64_t num = static_cast<int64_t>(n_num.data<int32_t>()[0]);

    PADDLE_ENFORCE_GT(num, 0, platform::errors::InvalidArgument(
                                  "The num of linspace op should be larger "
                                  "than 0, but received num is %d",
                                  num));

    out->Resize(framework::make_ddim({num}));
    T* out_data = out->mutable_data<T>(context.GetPlace());

    double step = 0;
    auto stream = context.cuda_device_context().stream();
    int block = 512;
    int grid = (num + block - 1) / block;
    if (num != 1) {
      step = (static_cast<double>(stop - start)) / (num - 1);
      LinspaceKernel<T><<<grid, block, 0, stream>>>(start, stop, step, num,
                                                    out_data);
    } else {
      LinspaceSpecialKernel<T><<<grid, block, 0, stream>>>(start, out_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(linspace, ops::CUDALinspaceKernel<float>,
                        ops::CUDALinspaceKernel<int32_t>,
                        ops::CUDALinspaceKernel<int64_t>,
                        ops::CUDALinspaceKernel<double>);
