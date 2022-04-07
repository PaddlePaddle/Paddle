/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/fluid/platform/device_context.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/partial_sum_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <class T>
__global__ void SumArrayPartialCUDAKernel(T **in, T *out, int64_t lod_length,
                                          size_t in_size, int64_t start_index,
                                          int64_t length, int64_t row_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < lod_length) {
    T total = static_cast<T>(0);
    int b_id = id / length;
    int b_offset = id % length;

    for (int i = 0; i < in_size; ++i) {
      const T *tmp = in[i];
      if (tmp) {
        total += tmp[start_index + b_id * row_length + b_offset];
      }
    }
    out[id] = total;
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void PartialSumGradCUDAKernel(T **res_grad, const T *out_grad,
                                         int64_t lod_length, size_t in_size,
                                         int64_t start_index, int64_t length,
                                         int64_t row_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < lod_length) {
    T total = static_cast<T>(0);
    int b_id = id / length;
    int b_offset = id % length;

    for (int i = 0; i < in_size; ++i) {
      T *tmp = res_grad[i];
      tmp[start_index + b_id * row_length + b_offset] = out_grad[i];
    }
    id += blockDim.x * gridDim.x;
  }
}

template <typename T>
class PartialSumOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_vars = ctx.MultiInput<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");

    PADDLE_ENFORCE_EQ(
        in_vars[0] != nullptr, true,
        platform::errors::InvalidArgument("The input should not be null."));

    auto place = ctx.GetPlace();  // GPUPlace only now
    auto start_index = ctx.Attr<int>("start_index");
    auto length = ctx.Attr<int>("length");
    auto batch_size = in_vars[0]->dims()[0];
    if (length == -1) {
      length = in_vars[0]->dims()[1] - start_index;
    }

    constexpr size_t theory_sm_threads = 1024;
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    auto sm_count = max_threads / theory_sm_threads;
    size_t tile_size = 0;
    dim3 grids;
    dim3 blocks;
    auto ComputeKernelParameter = [&](size_t length) {
      if (length >= max_threads)
        tile_size = 1024;
      else if (length < max_threads && length > sm_count * 128)
        tile_size = 512;
      else if (length <= sm_count * 128)
        tile_size = 256;
      grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
      blocks = dim3(tile_size, 1, 1);
    };

    auto lod_length = length * batch_size;
    auto row_length = in_vars[0]->dims()[1];
    auto in_num = in_vars.size();

    std::vector<const T *> in_data;
    for (int i = 0; i < in_num; ++i) {
      in_data.emplace_back(in_vars[i]->data<T>());
    }

    if (!in_data.empty()) {
      auto tmp_in_array = memory::Alloc(dev_ctx, in_data.size() * sizeof(T *));

      memory::Copy(dev_ctx.GetPlace(), tmp_in_array->ptr(),
                   platform::CPUPlace(),
                   reinterpret_cast<void *>(in_data.data()),
                   in_data.size() * sizeof(T *), dev_ctx.stream());

      T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
      ComputeKernelParameter(lod_length);
      SumArrayPartialCUDAKernel<T><<<grids, blocks, 0, stream>>>(
          in_array_data, out->data<T>(), lod_length, in_data.size(),
          start_index, length, row_length);
    }
  }
};

template <typename T>
class PartialSumGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto outs = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        ins[0] != nullptr, true,
        platform::errors::InvalidArgument("The input should not be null."));
    auto start_index = ctx.Attr<int>("start_index");
    auto length = ctx.Attr<int>("length");
    if (length == -1) {
      length = ins[0]->dims()[1] - start_index;
    }

    // initialize
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    for (size_t i = 0; i < outs.size(); ++i) {
      outs[i]->mutable_data<T>(ctx.GetPlace());
      auto dxt = framework::EigenVector<T>::Flatten(*outs[i]);
      dxt.device(place) = dxt.constant(static_cast<T>(0));
    }

    auto batch_size = ins[0]->dims()[0];
    if (length == -1) {
      length = ins[0]->dims()[1] - start_index;
    }
    auto lod_length = length * batch_size;
    auto row_length = ins[0]->dims()[1];
    auto out_num = outs.size();

    constexpr size_t theory_sm_threads = 1024;
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    auto sm_count = max_threads / theory_sm_threads;
    size_t tile_size = 0;
    dim3 grids;
    dim3 blocks;
    auto ComputeKernelParameter = [&](size_t length) {
      if (length >= max_threads)
        tile_size = 1024;
      else if (length < max_threads && length > sm_count * 128)
        tile_size = 512;
      else if (length <= sm_count * 128)
        tile_size = 256;
      grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
      blocks = dim3(tile_size, 1, 1);
    };

    std::vector<const T *> out_data;
    for (int i = 0; i < out_num; ++i) {
      out_data.emplace_back(outs[i]->data<T>());
    }

    if (!out_data.empty()) {
      auto tmp_out_array =
          memory::Alloc(dev_ctx, out_data.size() * sizeof(T *));

      memory::Copy(dev_ctx.GetPlace(), tmp_out_array->ptr(),
                   platform::CPUPlace(),
                   reinterpret_cast<void *>(out_data.data()),
                   out_data.size() * sizeof(T *), dev_ctx.stream());

      T **out_grad_data = reinterpret_cast<T **>(tmp_out_array->ptr());
      ComputeKernelParameter(lod_length);
      PartialSumGradCUDAKernel<T><<<grids, blocks, 0, stream>>>(
          out_grad_data, out_grad->data<T>(), lod_length, out_data.size(),
          start_index, length, row_length);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(partial_sum, ops::PartialSumOpCUDAKernel<float>,
                        ops::PartialSumOpCUDAKernel<double>,
                        ops::PartialSumOpCUDAKernel<int>,
                        ops::PartialSumOpCUDAKernel<int64_t>,
                        ops::PartialSumOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(partial_sum_grad,
                        ops::PartialSumGradOpCUDAKernel<float>,
                        ops::PartialSumGradOpCUDAKernel<double>,
                        ops::PartialSumGradOpCUDAKernel<int>,
                        ops::PartialSumGradOpCUDAKernel<int64_t>,
                        ops::PartialSumGradOpCUDAKernel<plat::float16>);
